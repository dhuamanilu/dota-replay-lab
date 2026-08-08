from dota_replay_lab.summary import render_match_summary


def test_render_match_summary_includes_match_result_and_players() -> None:
    match = {
        "match_id": 42,
        "duration": 125,
        "start_time": 1_700_000_000,
        "radiant_name": "Radiant Team",
        "dire_name": "Dire Team",
        "radiant_win": True,
        "players": [
            {
                "isRadiant": True,
                "hero_id": 1,
                "personaname": "tester",
                "kills": 4,
                "deaths": 2,
                "assists": 8,
                "gold_per_min": 420,
                "xp_per_min": 510,
            }
        ],
    }

    report = render_match_summary(match, {1: "Anti-Mage"})

    assert "# Match 42" in report
    assert "Radiant Team ganó" in report
    assert "Anti-Mage" in report
    assert "tester" in report
