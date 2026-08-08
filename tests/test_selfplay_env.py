import numpy as np
import pandas as pd

from dota_replay_lab.selfplay_env import (
    DotaSelfPlayEnv,
    ReplayCalibration,
    fit_replay_calibration,
)


def calibration() -> ReplayCalibration:
    return ReplayCalibration(
        gold_mean=(500, 550, 700, 300),
        gold_std=(10, 10, 10, 10),
        experience_mean=(600, 700, 900, 400),
        experience_std=(10, 10, 10, 10),
        last_hit_mean=(7, 5, 8, 1),
        last_hit_std=(1, 1, 1, 1),
        kill_rate=(0.1, 0.2, 0.3, 0.05),
        push_rate=0.02,
        hero_ids=tuple(range(1, 21)),
        source_matches=4,
        source_rows=16,
    )


def test_calibration_uses_only_selected_matches() -> None:
    rows = []
    for match_id, multiplier in ((1, 1), (2, 100)):
        for index, label in enumerate(("farm", "fight", "push", "unknown")):
            rows.append(
                {
                    "match_id": match_id,
                    "label": label,
                    "gold_change": multiplier * (index + 1),
                    "experience_change": multiplier * (index + 2),
                    "last_hit_change": multiplier * (index + 1),
                    "kills_last_minute": 0.1 * index,
                    "hero_id": index + 1,
                }
            )
    fitted = fit_replay_calibration(pd.DataFrame(rows), [1])
    assert fitted.source_matches == 1
    assert fitted.gold_mean == (1.0, 0.58, 0.68, 0.4)


def test_environment_is_zero_sum_and_causal() -> None:
    env = DotaSelfPlayEnv(calibration(), environments=2, stochastic=False, seed=5)
    actions = np.zeros((2, 2, 5), dtype=np.int64)
    actions[:, 0] = 2
    actions[:, 1] = 0
    before = env.towers.copy()
    (features, heroes, teams), rewards, done, _ = env.step(actions)
    assert features.shape == (2, 2, 5, 13)
    assert heroes.shape == teams.shape == (2, 2, 5)
    assert np.allclose(rewards[:, 0], -rewards[:, 1])
    assert np.all(env.towers[:, 0] > before[:, 0])
    assert not done.any()


def test_swapping_teams_negates_deterministic_outcome() -> None:
    first = DotaSelfPlayEnv(calibration(), stochastic=False, seed=7)
    second = DotaSelfPlayEnv(calibration(), stochastic=False, seed=7)
    actions = np.array([[[1, 1, 2, 0, 0], [0, 0, 3, 3, 2]]])
    _, reward_first, _, _ = first.step(actions)
    _, reward_second, _, _ = second.step(actions[:, ::-1])
    assert np.allclose(reward_first[:, 0], -reward_second[:, 0])
