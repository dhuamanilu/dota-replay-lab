-- Experimental Valve bot adapter for the distilled Dota Replay Lab policy.
local policy_path = GetScriptDirectory() .. "/decision_policy"
local policy_ok, policy_or_error = pcall(require, policy_path)
local policy = policy_ok and policy_or_error or nil
local bot_ok, bot = pcall(GetBot)
if not bot_ok then bot = nil end

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
}
local selected_action = "unknown"
local last_order = { name = "", target = "", time = -100 }

local function safe_number(callback, fallback)
  local ok, value = pcall(callback)
  if ok and type(value) == "number" then return value, true end
  return fallback, false
end

local function game_time()
  local value = safe_number(function() return DotaTime() end, 0)
  return value
end

local function game_minute()
  return math.max(0, math.floor(game_time() / 60))
end

local function json_string(value)
  local escaped = tostring(value)
    :gsub("\\", "\\\\")
    :gsub('"', '\\"')
    :gsub("\r", "\\r")
    :gsub("\n", "\\n")
  return '"' .. escaped .. '"'
end

local function json_value(value)
  if type(value) == "number" then return tostring(value) end
  if type(value) == "boolean" then return value and "true" or "false" end
  return json_string(value)
end

local telemetry_keys = {
  "action", "fallback", "order", "target", "features_available",
  "missing_features", "error",
}

local function telemetry(event, fields)
  local parts = {
    '"schema":1',
    '"event":' .. json_string(event),
    '"game_time":' .. tostring(math.floor(game_time() * 1000) / 1000),
    '"minute":' .. tostring(game_minute()),
  }
  for _, key in ipairs(telemetry_keys) do
    if fields ~= nil and fields[key] ~= nil then
      local value = fields[key]
      if key == "error" then value = tostring(value):sub(1, 240) end
      table.insert(parts, json_string(key) .. ":" .. json_value(value))
    end
  end
  print("DRL_TELEMETRY {" .. table.concat(parts, ",") .. "}")
end

if policy_ok then
  telemetry("policy_loaded", {})
else
  telemetry("policy_load_error", { error = policy_or_error })
end

local function append(values, value)
  table.insert(values, value)
end

local function read_counter(available, missing, name, callback, fallback)
  local value, ok = safe_number(callback, fallback)
  append(ok and available or missing, name)
  return value
end

local function read_counters()
  local available = {}
  local missing = {}
  local counters = {
    available = available,
    missing = missing,
  }
  counters.gold = read_counter(available, missing, "gold", function() return bot:GetGold() end, tracker.gold)
  counters.experience = read_counter(available, missing, "experience", function() return bot:GetCurrentXP() end, tracker.experience)
  counters.last_hits = read_counter(available, missing, "last_hits", function() return bot:GetLastHits() end, tracker.last_hits)
  counters.kills = read_counter(available, missing, "kills", function()
    return GetHeroKills(bot:GetPlayerID())
  end, tracker.kills)
  return counters
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
    end
    tracker.minute = minute
    tracker.gold = counters.gold
    tracker.experience = counters.experience
    tracker.last_hits = counters.last_hits
    tracker.kills = counters.kills
  end

  local team_number, team_ok = safe_number(function() return bot:GetTeam() end, 2)
  append(team_ok and counters.available or counters.missing, "team")
  local unit_ok, unit_name = pcall(function() return bot:GetUnitName() end)
  append(unit_ok and counters.available or counters.missing, "hero_id")
  local hero_id = 0
  if policy ~= nil and unit_ok then
    local hero_ok, value = pcall(policy.hero_id, unit_name)
    if hero_ok and type(value) == "number" then
      hero_id = value
    else
      append(counters.missing, "policy.hero_id")
    end
  end

  append(counters.available, "state_minute")
  append(counters.available, "gold_change")
  append(counters.available, "experience_change")
  append(counters.available, "last_hit_change")
  append(counters.available, "kills_last_minute")
  append(counters.missing, "team_gold_advantage")
  append(counters.missing, "team_experience_advantage")
  append(counters.missing, "previous_fight")
  append(counters.missing, "previous_push")
  append(counters.missing, "previous_farm")

  return {
    hero_id = hero_id,
    team = team_number == 2 and "Radiant" or "Dire",
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
    previous_fight = 0,
    previous_push = 0,
    previous_farm = 0,
  }, table.concat(counters.available, ","), table.concat(counters.missing, ",")
end

local function unit_name(unit)
  local ok, value = pcall(function() return unit:GetUnitName() end)
  if ok and value ~= nil then return value end
  return "unknown"
end

local function weakest(units)
  local target = nil
  local target_health = nil
  for _, unit in pairs(units or {}) do
    local ok, alive, health = pcall(function()
      return unit:IsAlive(), unit:GetHealth()
    end)
    if ok and alive and type(health) == "number" and (target_health == nil or health < target_health) then
      target = unit
      target_health = health
    end
  end
  return target
end

local function record_order(order, target, fallback)
  local now = game_time()
  if order ~= last_order.name or target ~= last_order.target or now - last_order.time >= 5 then
    telemetry("order_issued", { order = order, target = target, fallback = fallback })
    last_order.name = order
    last_order.target = target
    last_order.time = now
  end
end

local function attack(target, fallback)
  if target == nil then return false end
  local ok, err = pcall(function() bot:Action_AttackUnit(target, true) end)
  if not ok then
    telemetry("order_error", { order = "attack", target = unit_name(target), fallback = fallback, error = err })
    return false
  end
  record_order("attack", unit_name(target), fallback)
  return true
end

local function move_to_lane(fallback)
  local lane_ok, lane = pcall(function() return bot:GetAssignedLane() end)
  if not lane_ok or type(lane) ~= "number" then
    telemetry("query_error", { action = "farm", fallback = fallback, error = lane })
    return false
  end
  local front_ok, location = pcall(function()
    return GetLaneFrontLocation(bot:GetTeam(), lane, -600)
  end)
  if not front_ok or location == nil then
    telemetry("query_error", { action = "farm", fallback = fallback, error = location })
    return false
  end
  local moved, err = pcall(function() bot:Action_MoveToLocation(location) end)
  if not moved then
    telemetry("order_error", { order = "move", target = "lane_front", fallback = fallback, error = err })
    return false
  end
  record_order("move", "lane_front", fallback)
  return true
end

local function farm(fallback)
  local ok, creeps = pcall(function() return bot:GetNearbyLaneCreeps(1400, true) end)
  if not ok then
    telemetry("query_error", { action = "farm", fallback = fallback, error = creeps })
    return false
  end
  if attack(weakest(creeps), fallback) then return true end
  return move_to_lane(fallback or "farm_no_creep")
end

local function fight()
  local mode = BOT_MODE_NONE or 0
  local ok, heroes = pcall(function() return bot:GetNearbyHeroes(1800, true, mode) end)
  if not ok then
    telemetry("query_error", { action = "fight", error = heroes })
    return false
  end
  return attack(weakest(heroes), nil)
end

local function push()
  local ok, towers = pcall(function() return bot:GetNearbyTowers(2200, true) end)
  if not ok then
    telemetry("query_error", { action = "push", error = towers })
    return farm("push_tower_query_error")
  end
  if attack(weakest(towers), nil) then return true end
  return farm("push_no_tower")
end

local function retreat_if_threatened(fallback)
  local mode = BOT_MODE_NONE or 0
  local ok, enemies = pcall(function() return bot:GetNearbyHeroes(1200, true, mode) end)
  if not ok then
    telemetry("query_error", { action = "unknown", fallback = fallback, error = enemies })
    return farm("enemy_query_error")
  end
  if enemies == nil or #enemies == 0 then return farm(fallback or "unknown_no_threat") end
  local ancient_ok, ancient = pcall(function() return GetAncient(bot:GetTeam()) end)
  if not ancient_ok or ancient == nil then
    telemetry("query_error", { action = "unknown", fallback = fallback, error = ancient })
    return farm("ancient_unavailable")
  end
  local moved, err = pcall(function() bot:Action_MoveToLocation(ancient:GetLocation()) end)
  if not moved then
    telemetry("order_error", { order = "move", target = "ancient", fallback = fallback, error = err })
    return farm("retreat_order_error")
  end
  record_order("move", "ancient", fallback)
  return true
end

local function choose_action()
  local state, available, missing = checkpoint_state()
  if policy == nil then
    telemetry("decision", {
      action = "unknown", fallback = "policy_unavailable",
      features_available = available, missing_features = missing,
    })
    return "unknown"
  end
  local ok, action = pcall(policy.predict, state)
  if not ok then
    telemetry("decision_error", {
      action = "unknown", fallback = "policy_error", error = action,
      features_available = available, missing_features = missing,
    })
    return "unknown"
  end
  if action ~= "farm" and action ~= "fight" and action ~= "push" and action ~= "unknown" then
    telemetry("decision_error", {
      action = "unknown", fallback = "invalid_policy_action", error = action,
      features_available = available, missing_features = missing,
    })
    return "unknown"
  end
  telemetry("decision", {
    action = action, features_available = available, missing_features = missing,
  })
  return action
end

function Think()
  if bot == nil then return end
  local alive_ok, alive = pcall(function() return bot:IsAlive() end)
  if not alive_ok or not alive then return end
  local minute = game_minute()
  if minute ~= tracker.minute then selected_action = choose_action() end
  if selected_action == "fight" then
    if fight() then return end
    if farm("fight_no_target") then return end
  elseif selected_action == "push" then
    if push() then return end
  elseif selected_action == "farm" then
    if farm(nil) then return end
  end
  retreat_if_threatened(selected_action .. "_no_order")
end
