from dota_replay_lab.valve_telemetry import parse_lines, summarize


def test_parse_and_summarize_valve_console_telemetry() -> None:
    lines = [
        "unrelated engine output\n",
        'L 08/08 DRL_TELEMETRY {"schema":1,"event":"policy_loaded","game_time":0,"minute":0}\n',
        'DRL_TELEMETRY {"schema":1,"event":"decision","game_time":1.2,"minute":0,"action":"fight","missing_features":"previous_fight"}\n',
        'DRL_TELEMETRY {"schema":1,"event":"order_issued","game_time":1.3,"minute":0,"order":"attack","target":"npc_dota_hero_axe","fallback":"fight_no_target"}\n',
        'DRL_TELEMETRY {"schema":1,"event":"query_error","game_time":61.3,"minute":1,"action":"farm","error":"forced"}\n',
        "DRL_TELEMETRY not-json\n",
    ]

    records, invalid = parse_lines(lines)
    result = summarize(records, invalid)

    assert result["records"] == 4
    assert result["invalid_telemetry_lines"] == 1
    assert result["policy_loaded"] == 1
    assert result["decision_counts"] == {"fight": 1}
    assert result["fallback_counts"] == {"fight_no_target": 1}
    assert result["order_counts"] == {"attack": 1}
    assert result["error_event_counts"] == {"query_error": 1}
    assert result["observed_game_seconds"] == 61.3
    assert result["minutes_seen"] == [0, 1]
