-- Experimental Valve bot adapter for the distilled Dota Replay Lab policy.
local policy = require(GetScriptDirectory() .. "/decision_policy")
local bot = GetBot()

local tracker = {
  minute = -1,
  gold = 0,
  experience = 0,
  last_hits = 0,
  kills = 0,
  gold_change = 0,
  experience_change = 0,
  last_hit_change = 0,
  kills_last_minute = 0,
  previous_action = "unknown",
}
local selected_action = "unknown"

local function safe_number(callback, fallback)
  local ok, value = pcall(callback)
  if ok and type(value) == "number" then return value end
  return fallback
end

local function game_minute()
  return math.max(0, math.floor(safe_number(function() return DotaTime() end, 0) / 60))
end

local function read_counters()
  return {
    gold = safe_number(function() return bot:GetGold() end, tracker.gold),
    experience = safe_number(function() return bot:GetXP() end, tracker.experience),
    last_hits = safe_number(function() return bot:GetLastHits() end, tracker.last_hits),
    kills = safe_number(function() return bot:GetKills() end, tracker.kills),
  }
end

local function checkpoint_state()
  local minute = game_minute()
  local counters = read_counters()
  if minute ~= tracker.minute then
    if tracker.minute >= 0 then
      tracker.gold_change = counters.gold - tracker.gold
      tracker.experience_change = counters.experience - tracker.experience
      tracker.last_hit_change = counters.last_hits - tracker.last_hits
      tracker.kills_last_minute = counters.kills - tracker.kills
      tracker.previous_action = selected_action
    end
    tracker.minute = minute
    tracker.gold = counters.gold
    tracker.experience = counters.experience
    tracker.last_hits = counters.last_hits
    tracker.kills = counters.kills
  end
  local team = safe_number(function() return bot:GetTeam() end, 2) == 2 and "Radiant" or "Dire"
  return {
    hero_id = policy.hero_id(bot:GetUnitName()),
    team = team,
    state_minute = minute,
    gold = counters.gold,
    experience = counters.experience,
    last_hits = counters.last_hits,
    gold_change = tracker.gold_change,
    experience_change = tracker.experience_change,
    last_hit_change = tracker.last_hit_change,
    team_gold_advantage = 0,
    team_experience_advantage = 0,
    kills_last_minute = tracker.kills_last_minute,
    previous_fight = tracker.previous_action == "fight" and 1 or 0,
    previous_push = tracker.previous_action == "push" and 1 or 0,
    previous_farm = tracker.previous_action == "farm" and 1 or 0,
  }
end

local function weakest(units)
  local target = nil
  for _, unit in pairs(units or {}) do
    if unit ~= nil and unit:IsAlive() and (target == nil or unit:GetHealth() < target:GetHealth()) then
      target = unit
    end
  end
  return target
end

local function attack(target)
  if target == nil then return false end
  bot:Action_AttackUnit(target, true)
  return true
end

local function farm()
  local ok, creeps = pcall(function() return bot:GetNearbyLaneCreeps(1400, true) end)
  return ok and attack(weakest(creeps))
end

local function fight()
  local mode = BOT_MODE_NONE or 0
  local ok, heroes = pcall(function() return bot:GetNearbyHeroes(1800, true, mode) end)
  return ok and attack(weakest(heroes))
end

local function push()
  local ok, towers = pcall(function() return bot:GetNearbyTowers(2200, true) end)
  if ok and attack(weakest(towers)) then return true end
  return farm()
end

local function retreat_if_threatened()
  local mode = BOT_MODE_NONE or 0
  local ok, enemies = pcall(function() return bot:GetNearbyHeroes(1200, true, mode) end)
  if not ok or enemies == nil or #enemies == 0 then return farm() end
  local moved = pcall(function()
    local ancient = GetAncient(bot:GetTeam())
    bot:Action_MoveToLocation(ancient:GetLocation())
  end)
  return moved
end

function Think()
  if bot == nil or not bot:IsAlive() then return end
  local minute = game_minute()
  if minute ~= tracker.minute then
    selected_action = policy.predict(checkpoint_state())
  end
  if selected_action == "fight" and fight() then return end
  if selected_action == "push" and push() then return end
  if selected_action == "farm" and farm() then return end
  retreat_if_threatened()
end
