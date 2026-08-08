from dota_replay_lab.audit_team_selfplay import _aggregate


def test_multiseed_aggregation_recomputes_wilson_and_actions() -> None:
    rows = [
        {"games": 10, "wins": 7, "losses": 3, "draws": 0, "action_distribution": {"farm": 0.4, "fight": 0.3, "push": 0.2, "unknown": 0.1}},
        {"games": 10, "wins": 9, "losses": 1, "draws": 0, "action_distribution": {"farm": 0.6, "fight": 0.2, "push": 0.1, "unknown": 0.1}},
    ]
    result = _aggregate(rows)
    assert result["games"] == 20
    assert result["wins"] == 16
    assert result["decisive_win_rate"] == 0.8
    assert result["action_distribution"]["farm"] == 0.5
