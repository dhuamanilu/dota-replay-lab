from dota_replay_lab.timeline import render_advantage_svg, render_timeline


def test_timeline_renders_advantages_and_objectives() -> None:
    match = {
        "radiant_gold_adv": [0, 1200],
        "radiant_xp_adv": [0, -300],
        "players": [{"player_slot": 0, "hero_id": 1}],
        "objectives": [
            {"time": 65, "type": "CHAT_MESSAGE_FIRSTBLOOD", "player_slot": 0},
            {"time": 125, "type": "building_kill", "key": "npc_dota_goodguys_tower1_mid"},
        ],
    }

    timeline = render_timeline(match, {1: "Anti-Mage"})

    assert "| 1 | +1 200 | -300 |" in timeline
    assert "First blood por Anti-Mage" in timeline
    assert "tower1 mid" in timeline


def test_advantage_svg_contains_two_series() -> None:
    chart = render_advantage_svg({"radiant_gold_adv": [0, 20], "radiant_xp_adv": [0, -20]})

    assert "<svg" in chart
    assert "Oro" in chart
    assert "Experiencia" in chart
