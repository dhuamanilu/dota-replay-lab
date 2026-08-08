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
    for filename in ("decision_policy.lua", "bot_generic.lua"):
        source = (ROOT / "bots" / filename).read_text(encoding="utf-8")
        ast.parse(source)
