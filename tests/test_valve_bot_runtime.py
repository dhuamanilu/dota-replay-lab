import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def build_runtime(prediction: str | None = "farm", *, policy_error: bool = False):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    bots_path = (ROOT / "bots").as_posix()
    lua.execute(
        """
        CURRENT_TIME = 0
        BOT_MODE_NONE = 0
        BOT_ACTION_TYPE_IDLE = 1
        TELEMETRY = {}
        function print(message) table.insert(TELEMETRY, message) end

        function makeUnit(name, health)
          local unit = { name = name, health = health }
          function unit:IsAlive() return true end
          function unit:GetHealth() return self.health end
          function unit:GetUnitName() return self.name end
          return unit
        end

        mockBot = { creeps = {}, heroes = {}, towers = {} }
        function mockBot:IsAlive() return true end
        function mockBot:GetGold() return 600 end
        function mockBot:GetXP() return 500 end
        function mockBot:GetLastHits() return 3 end
        function mockBot:GetKills() return 0 end
        function mockBot:GetTeam() return 2 end
        function mockBot:GetPlayerID() return 0 end
        function mockBot:GetCurrentXP() return 500 end
        function mockBot:GetXPNeededToLevel() return 140 end
        function mockBot:GetCurrentActionType() return BOT_ACTION_TYPE_IDLE end
        function mockBot:GetUnitName() return "npc_dota_hero_life_stealer" end
        function mockBot:GetAssignedLane() return 2 end
        function mockBot:GetNearbyLaneCreeps(radius, enemies) return self.creeps end
        function mockBot:GetNearbyHeroes(radius, enemies, mode) return self.heroes end
        function mockBot:GetNearbyTowers(radius, enemies) return self.towers end
        function mockBot:Action_AttackUnit(target, once)
          self.last_action = "attack"
          self.last_target = target.name
        end
        function mockBot:Action_MoveToLocation(location) self.last_action = "move" end

        function GetBot() return mockBot end
        function GetHeroKills(player_id) return 0 end
        function GetHeroDeaths(player_id) return 0 end
        function GetHeroLevel(player_id) return 2 end
        function DotaTime() return CURRENT_TIME end
        function GetLaneFrontLocation(team, lane, offset)
          return { x = 100, y = 200 }
        end
        function GetAncient(team)
          return { GetLocation = function() return { x = 0, y = 0 } end }
        end
        """
    )
    lua.execute(f"function GetScriptDirectory() return {json.dumps(bots_path)} end")
    module_name = f"{bots_path}/decision_policy"
    if prediction is None:
        policy_source = (ROOT / "bots" / "decision_policy.lua").read_text(encoding="utf-8")
        lua.execute(
            f"package.preload[{json.dumps(module_name)}] = function()\n{policy_source}\nend"
        )
    elif policy_error:
        predict = 'error("forced policy failure")'
    else:
        predict = f"return {json.dumps(prediction)}"
    if prediction is not None:
        lua.execute(
            f"""
            package.preload[{json.dumps(module_name)}] = function()
              return {{
                hero_id = function(name) return 54 end,
                predict = function(state) {predict} end,
              }}
            end
            """
        )
    lua.execute((ROOT / "bots" / "bot_generic.lua").read_text(encoding="utf-8"))
    return lua


def test_valve_adapter_executes_policy_and_issues_an_action() -> None:
    lua = build_runtime("farm")
    lua.execute('mockBot.creeps = { makeUnit("npc_dota_creep_badguys_melee", 300) }')
    lua.globals().Think()
    assert lua.globals().mockBot.last_action == "attack"
    assert lua.globals().mockBot.last_target == "npc_dota_creep_badguys_melee"
    messages = [lua.globals().TELEMETRY[index] for index in range(1, len(lua.globals().TELEMETRY) + 1)]
    decision = next(json.loads(message.removeprefix("DRL_TELEMETRY ")) for message in messages if '"event":"decision"' in message)
    assert decision["schema"] == 3
    assert decision["player_id"] == 0
    assert decision["hero_name"] == "npc_dota_hero_life_stealer"
    assert decision["last_hits"] == 3
    assert decision["level"] == 2
    assert decision["xp_to_next_level"] == 140
    assert "experience" in decision["missing_features"].split(",")
    assert "experience_change" in decision["missing_features"].split(",")


def test_valve_adapter_measures_idle_alive_time() -> None:
    lua = build_runtime("farm")
    lua.globals().Think()
    lua.execute("CURRENT_TIME = 5")
    lua.globals().Think()
    messages = [
        lua.globals().TELEMETRY[index]
        for index in range(1, len(lua.globals().TELEMETRY) + 1)
    ]
    activity = [
        json.loads(message.removeprefix("DRL_TELEMETRY "))
        for message in messages
        if '"event":"activity"' in message
    ]
    assert activity[-1]["idle"] is True
    assert activity[-1]["idle_seconds"] == 5
    assert activity[-1]["activity_seconds"] == 5


def test_generated_policy_executes_through_the_valve_adapter() -> None:
    lua = build_runtime(None)
    lua.execute(
        """
        mockBot.creeps = { makeUnit("npc_dota_creep_badguys_melee", 300) }
        mockBot.heroes = { makeUnit("npc_dota_hero_axe", 900) }
        mockBot.towers = { makeUnit("npc_dota_badguys_tower1_mid", 1600) }
        """
    )
    lua.globals().Think()
    assert lua.globals().mockBot.last_action in {"attack", "move"}


@pytest.mark.parametrize(
    ("prediction", "setup", "expected_action", "expected_target"),
    [
        (
            "fight",
            'mockBot.heroes = { makeUnit("npc_dota_hero_axe", 900) }',
            "attack",
            "npc_dota_hero_axe",
        ),
        (
            "push",
            'mockBot.towers = { makeUnit("npc_dota_badguys_tower1_mid", 1600) }',
            "attack",
            "npc_dota_badguys_tower1_mid",
        ),
        (
            "unknown",
            'mockBot.heroes = { makeUnit("npc_dota_hero_axe", 900) }',
            "move",
            None,
        ),
    ],
)
def test_each_policy_action_has_a_safe_order(
    prediction: str, setup: str, expected_action: str, expected_target: str | None
) -> None:
    lua = build_runtime(prediction)
    lua.execute(setup)
    lua.globals().Think()
    assert lua.globals().mockBot.last_action == expected_action
    if expected_target is not None:
        assert lua.globals().mockBot.last_target == expected_target


def test_fight_without_target_falls_back_to_farm_and_records_it() -> None:
    lua = build_runtime("fight")
    lua.execute('mockBot.creeps = { makeUnit("npc_dota_creep_badguys_melee", 300) }')
    lua.globals().Think()
    assert lua.globals().mockBot.last_action == "attack"
    messages = [lua.globals().TELEMETRY[index] for index in range(1, len(lua.globals().TELEMETRY) + 1)]
    assert any('"event":"order_issued"' in message for message in messages)
    assert any('"fallback":"fight_no_target"' in message for message in messages)


def test_farm_without_nearby_creep_moves_to_lane_front() -> None:
    lua = build_runtime("farm")
    lua.globals().Think()
    assert lua.globals().mockBot.last_action == "move"
    messages = [lua.globals().TELEMETRY[index] for index in range(1, len(lua.globals().TELEMETRY) + 1)]
    assert any('"target":"lane_front"' in message for message in messages)
    assert any('"fallback":"farm_no_creep"' in message for message in messages)


def test_policy_error_degrades_to_unknown_and_emits_structured_telemetry() -> None:
    lua = build_runtime(policy_error=True)
    lua.execute('mockBot.heroes = { makeUnit("npc_dota_hero_axe", 900) }')
    lua.globals().Think()
    assert lua.globals().mockBot.last_action == "move"
    messages = [lua.globals().TELEMETRY[index] for index in range(1, len(lua.globals().TELEMETRY) + 1)]
    assert all(message.startswith("DRL_TELEMETRY {") for message in messages)
    assert any('"event":"policy_loaded"' in message for message in messages)
    assert any('"event":"decision_error"' in message for message in messages)
    assert any('"fallback":"policy_error"' in message for message in messages)
    decision_error = next(
        json.loads(message.removeprefix("DRL_TELEMETRY "))
        for message in messages
        if '"event":"decision_error"' in message
    )
    assert "team_gold_advantage" in decision_error["missing_features"].split(",")
