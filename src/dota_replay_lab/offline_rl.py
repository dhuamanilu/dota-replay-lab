"""Causal reward construction for offline reinforcement learning on replay timelines.

The dataset row at state minute ``m`` describes the information available before
the decision for minute ``m + 1``.  Outcome counters for that interval therefore
live on the next row of the same hero timeline.  This module performs that shift
explicitly and never treats replay transitions as a counterfactual simulator.
"""

from __future__ import annotations

from typing import Any


OUTCOME_WEIGHTS = {
    "gold_change": 0.006,
    "experience_change": 0.002,
    "last_hit_change": 0.40,
    "deny_change": 0.15,
    "kills_last_minute": 1.0,
}


def add_replay_rewards(frame: Any, *, team_spirit: float = 0.5) -> Any:
    """Return a copy with causal, team-aware rewards for logged transitions.

    Reward weights follow the scale of OpenAI Five's shaped rewards where the
    available replay counters overlap.  A reward is only valid when the next
    minute for the same ``(match_id, player_slot)`` exists.  Team spirit blends
    individual and allied mean reward, after which the enemy mean is subtracted
    to make the result zero-sum at each match-minute.
    """

    import numpy as np

    if not 0.0 <= team_spirit <= 1.0:
        raise ValueError("team_spirit must be between 0 and 1")
    required = {
        "match_id",
        "player_slot",
        "team",
        "state_minute",
        "decision_minute",
        *OUTCOME_WEIGHTS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing reward columns: {', '.join(missing)}")

    rewarded = frame.copy()
    original_index = rewarded.index.copy()
    rewarded["_reward_row_order"] = range(len(rewarded))
    rewarded = rewarded.sort_values(
        ["match_id", "player_slot", "state_minute"], kind="stable"
    )
    timeline = rewarded.groupby(["match_id", "player_slot"], sort=False)
    next_minute = timeline["state_minute"].shift(-1)
    contiguous = next_minute.eq(rewarded["decision_minute"])

    raw = np.zeros(len(rewarded), dtype=np.float64)
    for column, weight in OUTCOME_WEIGHTS.items():
        outcome = timeline[column].shift(-1).fillna(0.0).clip(lower=0.0)
        raw += outcome.to_numpy(dtype=np.float64) * weight
    rewarded["reward_valid"] = contiguous.to_numpy(dtype=bool)
    rewarded["individual_reward"] = np.where(contiguous, raw, 0.0)

    keys = ["match_id", "decision_minute", "team"]
    rewarded["team_reward"] = rewarded.groupby(keys, sort=False)[
        "individual_reward"
    ].transform("mean")
    means = (
        rewarded.groupby(keys, sort=False)["individual_reward"]
        .mean()
        .rename("enemy_reward")
        .reset_index()
    )
    means["team"] = means["team"].map({"Radiant": "Dire", "Dire": "Radiant"})
    rewarded = rewarded.merge(means, on=keys, how="left", sort=False)
    rewarded["enemy_reward"] = rewarded["enemy_reward"].fillna(0.0)

    shared = (1.0 - team_spirit) * rewarded["individual_reward"] + team_spirit * rewarded[
        "team_reward"
    ]
    rewarded["replay_reward"] = np.where(
        rewarded["reward_valid"], shared - rewarded["enemy_reward"], 0.0
    )
    rewarded = rewarded.sort_values("_reward_row_order", kind="stable").drop(
        columns="_reward_row_order"
    )
    rewarded.index = original_index
    return rewarded


def add_discounted_returns(frame: Any, *, gamma: float = 0.99) -> Any:
    """Add return-to-go within each hero timeline without crossing boundaries."""

    import numpy as np

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if "replay_reward" not in frame or "reward_valid" not in frame:
        raise ValueError("Call add_replay_rewards before add_discounted_returns")

    returned = frame.copy()
    values = np.zeros(len(returned), dtype=np.float64)
    positions = {index: position for position, index in enumerate(returned.index)}
    for _, rows in returned.groupby(["match_id", "player_slot"], sort=False):
        running = 0.0
        for index in reversed(rows.sort_values("state_minute").index.tolist()):
            if not bool(returned.at[index, "reward_valid"]):
                running = 0.0
            else:
                running = float(returned.at[index, "replay_reward"]) + gamma * running
            values[positions[index]] = running
    returned["discounted_return"] = values
    return returned


def add_advantage_weights(
    frame: Any,
    train_match_ids: Any,
    *,
    beta: float = 2.0,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> tuple[Any, dict[str, dict[str, float]]]:
    """Add conservative advantage-weighted imitation weights.

    Baseline medians and robust scales are fitted exclusively on training
    matches and separately for each logged action.  This keeps rare actions in
    the data while emphasizing unusually successful examples of that action.
    It is an offline-RL regularizer, not a counterfactual value estimate.
    """

    import numpy as np

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0 < minimum <= maximum:
        raise ValueError("weight bounds must satisfy 0 < minimum <= maximum")
    required = {"match_id", "label", "reward_valid", "replay_reward"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing advantage columns: {', '.join(missing)}")

    weighted = frame.copy()
    fit = weighted[
        weighted["match_id"].isin(set(train_match_ids)) & weighted["reward_valid"]
    ]
    if fit.empty:
        raise ValueError("Training split has no valid replay rewards")

    statistics: dict[str, dict[str, float]] = {}
    for label, rows in fit.groupby("label", sort=True):
        rewards = rows["replay_reward"].to_numpy(dtype=np.float64)
        median = float(np.median(rewards))
        mad = float(np.median(np.abs(rewards - median)))
        scale = max(1.4826 * mad, 1e-3)
        statistics[str(label)] = {"median": median, "scale": scale}
    missing_labels = sorted(set(weighted["label"].astype(str)) - set(statistics))
    if missing_labels:
        raise ValueError(f"Training split has no rewards for labels: {', '.join(missing_labels)}")

    medians = weighted["label"].astype(str).map(
        {label: values["median"] for label, values in statistics.items()}
    )
    scales = weighted["label"].astype(str).map(
        {label: values["scale"] for label, values in statistics.items()}
    )
    advantage = (weighted["replay_reward"] - medians) / scales
    weights = np.exp(np.clip(advantage / beta, -20.0, 20.0)).clip(minimum, maximum)
    weighted["replay_advantage"] = np.where(weighted["reward_valid"], advantage, 0.0)
    weighted["sample_weight"] = np.where(weighted["reward_valid"], weights, 0.0)
    return weighted, statistics
