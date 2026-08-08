import pandas as pd

from dota_replay_lab.replay_behavior import build_behavior_frame
from dota_replay_lab.train_replay_behavior import _grouped_bootstrap


def _row(slot: int, team: str, x: float, move: int, time: int = 0) -> dict:
    return {
        "match_id": 1,
        "time": time,
        "slot": slot,
        "hero_id": slot + 1,
        "team": team,
        "x": x,
        "y": 0,
        "alive": 1,
        "level": 1,
        "gold": 0,
        "xp": 0,
        "networth": 600,
        "lh": 0,
        "denies": 0,
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "movement_distance": 0,
        "move_orders": move,
        "attack_orders": 0,
        "cast_orders": 0,
    }


def test_behavior_features_use_previous_action_and_simultaneous_positions() -> None:
    frame = pd.DataFrame(
        [
            _row(0, "Radiant", 0, 1),
            _row(1, "Radiant", 10, 0),
            _row(5, "Dire", 30, 0),
            _row(0, "Radiant", 1, 0, time=1),
            _row(1, "Radiant", 11, 0, time=1),
            _row(5, "Dire", 31, 0, time=1),
        ]
    )
    result = build_behavior_frame(frame, nearby_distance=20)
    hero = result[(result.slot == 0) & (result.time == 1)].iloc[0]
    assert hero.previous_move == 1
    assert hero.label_move == 0
    assert hero.nearest_ally_distance == 10
    assert hero.nearest_enemy_distance == 30
    assert hero.allies_nearby == 1
    assert hero.enemies_nearby == 0


def test_grouped_bootstrap_resamples_match_deltas() -> None:
    result = _grouped_bootstrap([0.1, 0.2, 0.3], seed=7, samples=1000)
    assert result["matches"] == 3
    assert result["mean_delta"] == 0.20000000000000004
    assert result["probability_positive"] == 1.0
