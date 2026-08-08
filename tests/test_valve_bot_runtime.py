import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_valve_adapter_executes_policy_and_issues_an_action() -> None:
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    bots_path = (ROOT / "bots").as_posix()
    lua.execute(
        """
        CURRENT_TIME = 0
        BOT_MODE_NONE = 0
        local creep = { health = 300 }
        function creep:IsAlive() return true end
        function creep:GetHealth() return self.health end

        mockBot = {}
        function mockBot:IsAlive() return true end
        function mockBot:GetGold() return 600 end
        function mockBot:GetXP() return 500 end
        function mockBot:GetLastHits() return 3 end
        function mockBot:GetKills() return 0 end
        function mockBot:GetTeam() return 2 end
        function mockBot:GetUnitName() return "npc_dota_hero_life_stealer" end
        function mockBot:GetNearbyLaneCreeps(radius, enemies) return { creep } end
        function mockBot:GetNearbyHeroes(radius, enemies, mode) return {} end
        function mockBot:GetNearbyTowers(radius, enemies) return {} end
        function mockBot:Action_AttackUnit(target, once) self.last_action = "attack" end
        function mockBot:Action_MoveToLocation(location) self.last_action = "move" end

        function GetBot() return mockBot end
        function DotaTime() return CURRENT_TIME end
        function GetAncient(team)
          return { GetLocation = function() return { x = 0, y = 0 } end }
        end
        """
    )
    lua.execute(f"function GetScriptDirectory() return {json.dumps(bots_path)} end")
    module_name = f"{bots_path}/decision_policy"
    policy_source = (ROOT / "bots" / "decision_policy.lua").read_text(encoding="utf-8")
    lua.execute(f"package.preload[{json.dumps(module_name)}] = function()\n{policy_source}\nend")
    lua.execute((ROOT / "bots" / "bot_generic.lua").read_text(encoding="utf-8"))
    lua.globals().Think()
    assert lua.globals().mockBot.last_action == "attack"
