from dota_replay_lab.valve_telemetry import parse_lines, summarize


def test_parse_and_summarize_valve_console_telemetry() -> None:
    lines = [
        "unrelated engine output\n",
        'L 08/08 DRL_TELEMETRY {"schema":1,"event":"policy_loaded","game_time":0,"minute":0}\n',
        'DRL_TELEMETRY {"schema":2,"event":"decision","game_time":1.2,"minute":0,"player_id":2,"team_id":2,"hero_name":"npc_dota_hero_axe","action":"fight","last_hits":4,"kills":1,"deaths":0,"missing_features":"previous_fight"}\n',
        'DRL_TELEMETRY {"schema":1,"event":"order_issued","game_time":1.3,"minute":0,"order":"attack","target":"npc_dota_hero_axe","fallback":"fight_no_target"}\n',
        'DRL_TELEMETRY {"schema":1,"event":"query_error","game_time":61.3,"minute":1,"action":"farm","error":"forced"}\n',
        'DRL_TELEMETRY {"schema":3,"event":"activity","game_time":61.3,"minute":1,"player_id":2,"team_id":2,"hero_name":"npc_dota_hero_axe","action_type":1,"idle":true,"idle_seconds":5,"activity_seconds":60}\n',
        "DRL_TELEMETRY not-json\n",
    ]

    records, invalid = parse_lines(lines)
    result = summarize(records, invalid)

    assert result["records"] == 5
    assert result["invalid_telemetry_lines"] == 1
    assert result["policy_loaded"] == 1
    assert result["decision_counts"] == {"fight": 1}
    assert result["fallback_counts"] == {"fight_no_target": 1}
    assert result["order_counts"] == {"attack": 1}
    assert result["error_event_counts"] == {"query_error": 1}
    assert result["observed_game_seconds"] == 61.3
    assert result["minutes_seen"] == [0, 1]
    assert result["schema_versions"] == [1, 2, 3]
    assert result["counter_maxima"]["last_hits"] == 4
    assert result["bot_snapshots"]["2:npc_dota_hero_axe"]["last_action"] == "fight"
    assert result["activity_by_bot"]["2:npc_dota_hero_axe"]["idle_seconds"] == 5
    assert result["aggregate_idle_ratio"] == 5 / 60
    assert result["tower_attack_orders"] == 0
