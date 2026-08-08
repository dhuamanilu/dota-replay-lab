import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_valve_adapter_loads_policy_and_exposes_think() -> None:
    source = (ROOT / "bots" / "bot_generic.lua").read_text(encoding="utf-8")
    assert 'local policy_path = GetScriptDirectory() .. "/decision_policy"' in source
    assert "pcall(require, policy_path)" in source
    assert "function Think()" in source
    assert "pcall(policy.predict, state)" in source
    assert 'print("DRL_TELEMETRY {"' in source


def test_generated_lua_and_adapter_parse_when_luaparser_is_installed() -> None:
    ast = pytest.importorskip("luaparser.ast")
    for filename in (
        "decision_policy.lua",
        "replay_combat_policy.lua",
        "bot_generic.lua",
    ):
        source = (ROOT / "bots" / filename).read_text(encoding="utf-8")
        ast.parse(source)


def test_large_team_policy_compiles_in_the_target_lua_runtime() -> None:
    lupa = pytest.importorskip("lupa")
    source = (ROOT / "bots" / "team_selfplay_policy.lua").read_text(encoding="utf-8")
    policy = lupa.LuaRuntime(unpack_returned_tuples=True).execute(source)
    assert policy.hero_id("npc_dota_hero_axe") > 0
    metrics = json.loads(
        (ROOT / "bots" / "team_selfplay_policy.metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["training"]["promotion_gate"]["passed"] is True
    assert metrics["multi_seed_audit"]["promotion_gate"]["passed"] is True
    assert metrics["lua_parity"]["fidelity"] == 1.0
    assert metrics["lua_parity"]["largest_probability_error"] <= 1e-6
