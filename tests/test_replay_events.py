from dota_replay_lab.replay_events import action_group, extract_second_rows


def test_action_groups_cover_high_level_orders() -> None:
    assert action_group("1") == "move"
    assert action_group(4) == "attack"
    assert action_group(8) == "cast"
    assert action_group(10) == "hold"
    assert action_group("bad") is None


def test_second_rows_join_actions_and_compute_contiguous_movement() -> None:
    events = [
        {"time": 0, "type": "actions", "key": "1", "slot": 0},
        {"time": 0, "type": "actions", "key": "4", "slot": 0},
        {
            "time": 0,
            "type": "interval",
            "unit": "CDOTA_Unit_Hero_Axe",
            "slot": 0,
            "hero_id": 1,
            "x": 10,
            "y": 20,
            "life_state": 0,
        },
        {
            "time": 1,
            "type": "interval",
            "unit": "CDOTA_Unit_Hero_Axe",
            "slot": 0,
            "hero_id": 1,
            "x": 13,
            "y": 24,
            "life_state": 1,
        },
        {
            "time": 3,
            "type": "interval",
            "unit": "CDOTA_Unit_Hero_Axe",
            "slot": 0,
            "hero_id": 1,
            "x": 30,
            "y": 40,
            "life_state": 0,
        },
        {
            "time": 1,
            "type": "DOTA_COMBATLOG_DAMAGE",
            "value": 75,
            "attackername": "npc_dota_hero_axe",
            "targetname": "npc_dota_hero_lina",
            "attackerhero": True,
            "targethero": True,
        },
        {
            "time": 1,
            "type": "DOTA_COMBATLOG_DAMAGE",
            "value": 20,
            "attackername": "npc_dota_creep_badguys_melee",
            "targetname": "npc_dota_hero_axe",
            "attackerhero": False,
            "targethero": True,
        },
    ]
    rows = extract_second_rows(events)
    assert rows[0]["orders"] == 2
    assert rows[0]["move_orders"] == 1
    assert rows[0]["attack_orders"] == 1
    assert rows[1]["movement_distance"] == 5
    assert rows[1]["alive"] == 0
    assert rows[1]["hero_damage_dealt"] == 75
    assert rows[1]["hero_damage_received"] == 20
    assert rows[2]["movement_distance"] == 0
