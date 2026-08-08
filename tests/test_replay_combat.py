import pandas as pd

from dota_replay_lab.replay_combat import build_combat_frame


def test_combat_target_is_next_contiguous_second() -> None:
    rows = []
    for time, dealt in ((0, 0), (1, 25), (3, 10)):
        rows.append(
            {
                "match_id": 1,
                "time": time,
                "slot": 0,
                "hero_id": 1,
                "team": "Radiant",
                "x": 0,
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
                "move_orders": 0,
                "attack_orders": 0,
                "cast_orders": 0,
                "hero_damage_dealt": dealt,
                "hero_damage_received": 0,
            }
        )
    result = build_combat_frame(pd.DataFrame(rows), horizon=1)
    assert result.time.tolist() == [0]
    assert result.iloc[0].recent_damage_dealt == 0
    assert result.iloc[0].label_engage == 1
