from dota_replay_lab.hero_state import render_states, state_at_minute


def test_state_at_minute_uses_hero_series_and_team_perspective() -> None:
    match = {"radiant_gold_adv": [0, 500], "radiant_xp_adv": [0, -200]}
    player = {
        "isRadiant": False,
        "hero_id": 1,
        "gold_t": [100, 300],
        "xp_t": [20, 90],
        "lh_t": [0, 3],
        "kills_log": [{"time": 45}],
    }

    state = state_at_minute(match, player, 1, {1: "Anti-Mage"})

    assert state.hero == "Anti-Mage"
    assert state.gold_change == 200
    assert state.team_gold_advantage == -500
    assert state.team_experience_advantage == 200
    assert state.kills_last_minute == 1


def test_render_states_explains_data_limitations() -> None:
    report = render_states(
        {"players": [{"isRadiant": True, "hero_id": 1, "gold_t": [100]}]}, 0, {1: "Anti-Mage"}
    )

    assert "Anti-Mage" in report
    assert "Límites conscientes" in report
