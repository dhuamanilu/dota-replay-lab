"""Symmetric replay-calibrated 5v5 surrogate for reproducible self-play.

This is deliberately an abstract minute-level environment, not a claim that Dota
has been simulated exactly.  Its purpose is to make actions causally affect a
zero-sum opponent while keeping magnitudes anchored to held historical rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


ACTIONS = ("farm", "fight", "push", "unknown")
OBSERVATION_FEATURES = (
    "state_minute",
    "gold",
    "experience",
    "last_hits",
    "gold_change",
    "experience_change",
    "last_hit_change",
    "team_gold_advantage",
    "team_experience_advantage",
    "kills_last_minute",
    "previous_fight",
    "previous_push",
    "previous_farm",
)


@dataclass(frozen=True)
class ReplayCalibration:
    """Action-conditional per-minute statistics fitted on training matches only."""

    gold_mean: tuple[float, ...]
    gold_std: tuple[float, ...]
    experience_mean: tuple[float, ...]
    experience_std: tuple[float, ...]
    last_hit_mean: tuple[float, ...]
    last_hit_std: tuple[float, ...]
    kill_rate: tuple[float, ...]
    push_rate: float
    hero_ids: tuple[int, ...]
    source_matches: int
    source_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ReplayCalibration":
        converted = dict(values)
        for name in (
            "gold_mean",
            "gold_std",
            "experience_mean",
            "experience_std",
            "last_hit_mean",
            "last_hit_std",
            "kill_rate",
            "hero_ids",
        ):
            converted[name] = tuple(converted[name])
        return cls(**converted)


def fit_replay_calibration(frame: Any, match_ids: Iterable[int]) -> ReplayCalibration:
    """Fit magnitudes without treating successful labels as causal effects.

    Farm rows anchor uncontested resource availability. Other actions pay
    explicit opportunity costs; their upside comes from opponent-dependent kills
    or towers in ``step``. This avoids granting free resources merely because a
    historical row was labelled after a successful objective.
    """

    import numpy as np

    selected = frame[frame["match_id"].isin(set(int(value) for value in match_ids))]
    if selected.empty:
        raise ValueError("Calibration split contains no rows")
    required = {
        "label",
        "gold_change",
        "experience_change",
        "last_hit_change",
        "kills_last_minute",
        "hero_id",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Calibration frame is missing: {', '.join(missing)}")

    def anchored_statistics(
        column: str, multipliers: tuple[float, ...]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = selected.loc[selected["label"] == "farm", column].astype(float)
        lower, upper = values.quantile([0.05, 0.95])
        clipped = values.clip(lower=lower, upper=upper)
        mean = float(max(clipped.mean(), 0.0))
        scale = float(max(clipped.std(), 1.0))
        return (
            tuple(mean * value for value in multipliers),
            tuple(max(scale * value, 1.0) for value in multipliers),
        )

    gold_mean, gold_std = anchored_statistics("gold_change", (1.0, 0.58, 0.68, 0.40))
    experience_mean, experience_std = anchored_statistics(
        "experience_change", (1.0, 0.65, 0.75, 0.50)
    )
    last_hit_mean, last_hit_std = anchored_statistics(
        "last_hit_change", (1.0, 0.45, 0.65, 0.20)
    )
    kill_rate = tuple(
        float(
            selected.loc[selected["label"] == action, "kills_last_minute"]
            .clip(lower=0, upper=2)
            .mean()
        )
        for action in ACTIONS
    )
    heroes = tuple(sorted(int(value) for value in selected["hero_id"].unique() if int(value) > 0))
    return ReplayCalibration(
        gold_mean,
        gold_std,
        experience_mean,
        experience_std,
        last_hit_mean,
        last_hit_std,
        kill_rate,
        float((selected["label"] == "push").mean()),
        heroes,
        int(selected["match_id"].nunique()),
        int(len(selected)),
    )


class DotaSelfPlayEnv:
    """Vectorized two-team environment with five shared-policy heroes per side."""

    def __init__(
        self,
        calibration: ReplayCalibration,
        *,
        environments: int = 1,
        seed: int = 42,
        max_minutes: int = 45,
        stochastic: bool = True,
    ) -> None:
        import numpy as np

        if environments < 1 or max_minutes < 1:
            raise ValueError("environments and max_minutes must be positive")
        self.calibration = calibration
        self.environments = environments
        self.max_minutes = max_minutes
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> tuple[Any, Any, Any]:
        import numpy as np

        shape = (self.environments, 2, 5)
        self.minute = np.zeros(self.environments, dtype=np.int32)
        self.gold = np.full(shape, 600.0, dtype=np.float32)
        self.experience = np.zeros(shape, dtype=np.float32)
        self.last_hits = np.zeros(shape, dtype=np.float32)
        self.kills = np.zeros(shape, dtype=np.float32)
        self.deaths = np.zeros(shape, dtype=np.float32)
        self.towers = np.zeros((self.environments, 2), dtype=np.float32)
        self.last_gold_change = np.zeros(shape, dtype=np.float32)
        self.last_experience_change = np.zeros(shape, dtype=np.float32)
        self.last_hit_change = np.zeros(shape, dtype=np.float32)
        self.last_kills = np.zeros(shape, dtype=np.float32)
        self.previous_actions = np.full(shape, 3, dtype=np.int64)
        heroes = np.asarray(self.calibration.hero_ids, dtype=np.int64)
        if len(heroes) < 10:
            raise ValueError("Calibration needs at least ten distinct heroes")
        self.hero_ids = np.empty(shape, dtype=np.int64)
        for environment in range(self.environments):
            self.hero_ids[environment] = self.rng.choice(heroes, size=10, replace=False).reshape(2, 5)
        self.done = np.zeros(self.environments, dtype=bool)
        return self.observations()

    def _score(self) -> Any:
        import numpy as np

        gold = self.gold.sum(axis=2)
        experience = self.experience.sum(axis=2)
        kills = self.kills.sum(axis=2)
        score = gold / 5000.0 + experience / 8000.0 + kills * 0.5 + self.towers * 1.5
        return score[:, 0] - score[:, 1]

    def observations(self) -> tuple[Any, Any, Any]:
        """Return features, hero ids and team ids for all ten agents."""

        import numpy as np

        team_gold = self.gold.sum(axis=2)
        team_experience = self.experience.sum(axis=2)
        gold_advantage = team_gold[:, :, None] - team_gold[:, ::-1, None]
        experience_advantage = (
            team_experience[:, :, None] - team_experience[:, ::-1, None]
        )
        features = np.stack(
            (
                np.broadcast_to(self.minute[:, None, None], self.gold.shape),
                self.gold,
                self.experience,
                self.last_hits,
                self.last_gold_change,
                self.last_experience_change,
                self.last_hit_change,
                np.broadcast_to(gold_advantage, self.gold.shape),
                np.broadcast_to(experience_advantage, self.gold.shape),
                self.last_kills,
                (self.previous_actions == 1).astype(np.float32),
                (self.previous_actions == 2).astype(np.float32),
                (self.previous_actions == 0).astype(np.float32),
            ),
            axis=-1,
        ).astype(np.float32)
        teams = np.broadcast_to(
            np.asarray([0, 1], dtype=np.int64)[None, :, None], self.hero_ids.shape
        ).copy()
        return features, self.hero_ids.copy(), teams

    def step(self, actions: Any) -> tuple[tuple[Any, Any, Any], Any, Any, dict[str, Any]]:
        """Advance one minute and return zero-sum team rewards."""

        import numpy as np

        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != self.gold.shape:
            raise ValueError(f"Expected actions shaped {self.gold.shape}, got {actions.shape}")
        if ((actions < 0) | (actions >= len(ACTIONS))).any():
            raise ValueError("Action index is out of range")
        active = ~self.done
        before = self._score()
        means = (
            np.asarray(self.calibration.gold_mean),
            np.asarray(self.calibration.experience_mean),
            np.asarray(self.calibration.last_hit_mean),
        )
        scales = (
            np.asarray(self.calibration.gold_std),
            np.asarray(self.calibration.experience_std),
            np.asarray(self.calibration.last_hit_std),
        )
        changes = []
        for mean, scale in zip(means, scales):
            expected = mean[actions]
            if self.stochastic:
                expected = expected + self.rng.normal(size=actions.shape) * scale[actions] * 0.35
            changes.append(np.maximum(expected, 0.0).astype(np.float32))
        gold_change, experience_change, last_hit_change = changes

        # Interaction is symmetric: fighting punishes exposed farming/pushing,
        # while retreat reduces exposure at the cost of the lowest replay growth.
        fighter_count = (actions == 1).sum(axis=2).astype(np.float32)
        pusher_count = (actions == 2).sum(axis=2).astype(np.float32)
        exposed = (
            (actions == 0).sum(axis=2)
            + 1.25 * pusher_count
            + 0.5 * fighter_count
        ).astype(np.float32)
        team_gold = self.gold.sum(axis=2)
        advantage = np.clip((team_gold - team_gold[:, ::-1]) / 8000.0, -1.0, 1.0)
        expected_kills = (
            (self.calibration.kill_rate[1] / 5.0)
            * fighter_count
            * exposed[:, ::-1]
            * np.exp(0.35 * advantage)
        )
        if self.stochastic:
            team_kills = self.rng.poisson(expected_kills).astype(np.float32)
        else:
            team_kills = expected_kills.astype(np.float32)
        team_kills = np.minimum(team_kills, 5.0)
        hero_kills = np.zeros_like(self.kills)
        hero_deaths = np.zeros_like(self.deaths)
        for team in (0, 1):
            attackers = (actions[:, team] == 1).astype(np.float32)
            vulnerable = np.where(actions[:, 1 - team] == 3, 0.25, 1.0).astype(np.float32)
            attacker_total = np.maximum(attackers.sum(axis=1, keepdims=True), 1.0)
            vulnerable_total = np.maximum(vulnerable.sum(axis=1, keepdims=True), 1.0)
            hero_kills[:, team] = team_kills[:, team, None] * attackers / attacker_total
            hero_deaths[:, 1 - team] += (
                team_kills[:, team, None] * vulnerable / vulnerable_total
            )
        gold_change += hero_kills * 250.0
        experience_change += hero_kills * 350.0

        # A building event is credited to one player in the replay rows. Convert
        # that per-player rate to a per-team opportunity (five players), while
        # extra pushers add sublinear coordination rather than five free towers.
        push_pressure = (
            (self.calibration.push_rate * 5.0)
            * np.sqrt(pusher_count)
            * np.exp(0.25 * advantage)
            / (1.0 + 0.35 * fighter_count[:, ::-1])
        )
        if self.stochastic:
            tower_gain = (self.rng.random(push_pressure.shape) < (1 - np.exp(-push_pressure))).astype(
                np.float32
            )
        else:
            tower_gain = (1 - np.exp(-push_pressure)).astype(np.float32)
        tower_gain *= active[:, None]
        gold_change += tower_gain[:, :, None] * 200.0
        experience_change += tower_gain[:, :, None] * 120.0

        active_agents = active[:, None, None]
        self.last_gold_change = np.where(active_agents, gold_change, 0)
        self.last_experience_change = np.where(active_agents, experience_change, 0)
        self.last_hit_change = np.where(active_agents, last_hit_change, 0)
        self.last_kills = np.where(active_agents, hero_kills, 0)
        self.gold += self.last_gold_change
        self.experience += self.last_experience_change
        self.last_hits += self.last_hit_change
        self.kills += self.last_kills
        self.deaths += np.where(active_agents, hero_deaths, 0)
        self.towers = np.minimum(self.towers + tower_gain, 11.0)
        self.previous_actions = np.where(active_agents, actions, self.previous_actions)
        self.minute += active.astype(np.int32)
        self.done |= (self.minute >= self.max_minutes) | (self.towers.max(axis=1) >= 11)
        after = self._score()
        radiant_reward = np.where(active, after - before, 0.0).astype(np.float32)
        rewards = np.stack((radiant_reward, -radiant_reward), axis=1)
        info = {
            "score": after.copy(),
            "towers": self.towers.copy(),
            "kills": self.kills.sum(axis=2).copy(),
        }
        return self.observations(), rewards, self.done.copy(), info

    def winners(self) -> Any:
        import numpy as np

        score = self._score()
        return np.where(score > 1e-6, 0, np.where(score < -1e-6, 1, -1))
