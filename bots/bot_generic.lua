-- Experimental Valve bot adapter for the distilled Dota Replay Lab policy.
local policy_path = GetScriptDirectory() .. "/decision_policy"
local policy_ok, policy_or_error = pcall(require, policy_path)
local policy = policy_ok and policy_or_error or nil
local combat_policy_path = GetScriptDirectory() .. "/replay_combat_policy"
local combat_policy_ok, combat_policy_or_error = pcall(require, combat_policy_path)
local combat_policy = combat_policy_ok and combat_policy_or_error or nil
local team_policy_path = GetScriptDirectory() .. "/team_selfplay_policy"
local team_policy_ok, team_policy_or_error = pcall(require, team_policy_path)
local team_policy = team_policy_ok and team_policy_or_error or nil
local bot_ok, bot = pcall(GetBot)
if not bot_ok then bot = nil end

local tracker = {
  minute = -1,
  gold = 0,
  experience = 0,
  last_hits = 0,
  denies = 0,
  kills = 0,
  gold_change = 0,
  experience_change = 0,
  last_hit_change = 0,
  deny_change = 0,
  kills_last_minute = 0,
}
local selected_action = "unknown"
local selected_minute = -1
local last_order = { name = "", target = "", time = -100 }
local activity = { last_sample = nil, idle_seconds = 0, observed_seconds = 0 }
local observed = { fight = 0, push = 0, farm = 0 }
local previous_observed = { fight = 0, push = 0, farm = 0 }
local combat_tracker = { time = -100, prediction = nil, x = nil, y = nil }
local team_tracker = { gold = {}, last_hits = {}, kills = {}, previous_actions = {} }

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
  "player_id", "team_id", "hero_name", "action", "fallback", "order", "target",
  "gold", "experience", "level", "xp_to_next_level", "last_hits", "denies", "kills", "deaths",
  "gold_change", "experience_change", "last_hit_change", "deny_change", "kills_last_minute",
  "previous_fight", "previous_push", "previous_farm",
  "action_type", "idle", "idle_seconds", "activity_seconds",
  "features_available", "missing_features", "error",
  "engage_probability", "threat_probability",
}

local function telemetry(event, fields)
  fields = fields or {}
  fields.player_id = safe_number(function() return bot:GetPlayerID() end, -1)
  fields.team_id = safe_number(function() return bot:GetTeam() end, -1)
  local hero_ok, hero_name = pcall(function() return bot:GetUnitName() end)
  fields.hero_name = hero_ok and hero_name or "unknown"
  local parts = {
    '"schema":4',
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

local function sample_activity()
  local now = game_time()
  if activity.last_sample ~= nil and now - activity.last_sample < 5 then return end
  local action_type, action_ok = safe_number(function()
    return bot:GetCurrentActionType()
  end, -1)
  local elapsed = activity.last_sample ~= nil and math.max(0, now - activity.last_sample) or 0
  if action_ok then
    activity.observed_seconds = activity.observed_seconds + elapsed
    if action_type == (BOT_ACTION_TYPE_IDLE or 1) then
      activity.idle_seconds = activity.idle_seconds + elapsed
    end
  end
  activity.last_sample = now
  telemetry("activity", {
    action_type = action_type,
    idle = action_ok and action_type == (BOT_ACTION_TYPE_IDLE or 1),
    idle_seconds = math.floor(activity.idle_seconds * 1000) / 1000,
    activity_seconds = math.floor(activity.observed_seconds * 1000) / 1000,
  })
end

if policy_ok then
  telemetry("policy_loaded", {})
else
  telemetry("policy_load_error", { error = policy_or_error })
end
if combat_policy_ok then
  telemetry("combat_policy_loaded", {})
else
  telemetry("combat_policy_load_error", { error = combat_policy_or_error })
end
if team_policy_ok then
  telemetry("team_selfplay_policy_loaded", {})
else
  telemetry("team_selfplay_policy_load_error", { error = team_policy_or_error })
end

local function append(values, value)
  table.insert(values, value)
end

local function read_counter(available, missing, name, callback, fallback)
  local value, ok = safe_number(callback, fallback)
  append(ok and available or missing, name)
  return value, ok
end

local function read_counters()
  local available = {}
  local missing = {}
  local counters = {
    available = available,
    missing = missing,
  }
  counters.gold, counters.gold_ok = read_counter(
    available, missing, "gold", function() return bot:GetGold() end, tracker.gold
  )
  -- GetCurrentXP belongs to the server entity API, not CDOTA_Bot_Script.
  -- Preserve the trained input default and report the domain gap honestly.
  counters.experience = tracker.experience
  counters.experience_ok = false
  append(missing, "experience")
  counters.last_hits, counters.last_hits_ok = read_counter(
    available, missing, "last_hits", function() return bot:GetLastHits() end, tracker.last_hits
  )
  counters.denies, counters.denies_ok = read_counter(
    available, missing, "denies", function() return bot:GetDenies() end, tracker.denies
  )
  counters.kills, counters.kills_ok = read_counter(available, missing, "kills", function()
    return GetHeroKills(bot:GetPlayerID())
  end, tracker.kills)
  counters.deaths = safe_number(function() return GetHeroDeaths(bot:GetPlayerID()) end, 0)
  counters.level = safe_number(function() return GetHeroLevel(bot:GetPlayerID()) end, 1)
  counters.xp_to_next_level = safe_number(function() return bot:GetXPNeededToLevel() end, -1)
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
      tracker.deny_change = counters.denies - tracker.denies
      tracker.kills_last_minute = counters.kills - tracker.kills
      previous_observed.fight = observed.fight
      previous_observed.push = observed.push
      previous_observed.farm = observed.farm
    end
    observed.fight = 0
    observed.push = 0
    observed.farm = 0
    tracker.minute = minute
    tracker.gold = counters.gold
    tracker.experience = counters.experience
    tracker.last_hits = counters.last_hits
    tracker.denies = counters.denies
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
  append(counters.gold_ok and counters.available or counters.missing, "gold_change")
  append(counters.experience_ok and counters.available or counters.missing, "experience_change")
  append(counters.last_hits_ok and counters.available or counters.missing, "last_hit_change")
  append(counters.denies_ok and counters.available or counters.missing, "deny_change")
  append(counters.kills_ok and counters.available or counters.missing, "kills_last_minute")
  append(counters.missing, "team_gold_advantage")
  append(counters.missing, "team_experience_advantage")
  append(counters.available, "previous_fight")
  append(counters.available, "previous_push")
  append(counters.available, "previous_farm")

  return {
    hero_id = hero_id,
    team = team_number == 2 and "Radiant" or "Dire",
    state_minute = minute,
    gold = counters.gold,
    experience = counters.experience,
    level = counters.level,
    xp_to_next_level = counters.xp_to_next_level,
    last_hits = counters.last_hits,
    denies = counters.denies,
    kills = counters.kills,
    deaths = counters.deaths,
    gold_change = tracker.gold_change,
    experience_change = tracker.experience_change,
    last_hit_change = tracker.last_hit_change,
    deny_change = tracker.deny_change,
    team_gold_advantage = 0,
    team_experience_advantage = 0,
    kills_last_minute = tracker.kills_last_minute,
    previous_fight = previous_observed.fight,
    previous_push = previous_observed.push,
    previous_farm = previous_observed.farm,
  }, table.concat(counters.available, ","), table.concat(counters.missing, ",")
end

local function add_state_fields(fields, state)
  for _, key in ipairs({
    "gold", "experience", "level", "xp_to_next_level", "last_hits", "denies", "kills", "deaths",
    "gold_change", "experience_change", "last_hit_change", "deny_change", "kills_last_minute",
    "previous_fight", "previous_push", "previous_farm",
  }) do
    fields[key] = state[key]
  end
  return fields
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

local function weakest_killable(units, damage, maximum_health_ratio)
  if type(damage) ~= "number" or damage <= 0 then return nil end
  local target = nil
  local target_health = nil
  for _, unit in pairs(units or {}) do
    local ok, alive, health, max_health = pcall(function()
      return unit:IsAlive(), unit:GetHealth(), unit:GetMaxHealth()
    end)
    local ratio_ok = maximum_health_ratio == nil
      or (type(max_health) == "number" and max_health > 0 and health / max_health <= maximum_health_ratio)
    if ok and alive and ratio_ok and type(health) == "number" and health <= damage * 1.15
      and (target_health == nil or health < target_health) then
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
  local attack_damage = safe_number(function() return bot:GetAttackDamage() end, 0)
  local target = weakest_killable(creeps, attack_damage, nil)
  if attack(target, fallback or "last_hit") then
    observed.farm = 1
    return true
  end
  local allies_ok, allied_creeps = pcall(function() return bot:GetNearbyLaneCreeps(900, false) end)
  if allies_ok then
    local deny_target = weakest_killable(allied_creeps, attack_damage, 0.5)
    if attack(deny_target, fallback or "deny") then
      observed.farm = 1
      return true
    end
  end
  if attack(weakest(creeps), fallback) then
    observed.farm = 1
    return true
  end
  local moved = move_to_lane(fallback or "farm_no_creep")
  if moved then observed.farm = 1 end
  return moved
end

local function fight()
  local mode = BOT_MODE_NONE or 0
  local ok, heroes = pcall(function() return bot:GetNearbyHeroes(1800, true, mode) end)
  if not ok then
    telemetry("query_error", { action = "fight", error = heroes })
    return false
  end
  local attacked = attack(weakest(heroes), nil)
  if attacked then observed.fight = 1 end
  return attacked
end

local function spatial_context(units, own_x, own_y)
  local nearest = 256
  local nearby = 0
  for _, unit in pairs(units or {}) do
    local ok, location = pcall(function() return unit:GetLocation() end)
    if ok and location ~= nil and type(location.x) == "number" and type(location.y) == "number" then
      local distance = math.sqrt((location.x - own_x) ^ 2 + (location.y - own_y) ^ 2) / 64
      nearest = math.min(nearest, distance)
      if distance <= 20 then nearby = nearby + 1 end
    end
  end
  return nearest, nearby
end

local function combat_prediction()
  if combat_policy == nil or policy == nil then return nil end
  local now = game_time()
  if combat_tracker.prediction ~= nil and now - combat_tracker.time < 1 then
    return combat_tracker.prediction
  end
  local location_ok, location = pcall(function() return bot:GetLocation() end)
  if not location_ok or location == nil or type(location.x) ~= "number" or type(location.y) ~= "number" then
    return nil
  end
  local mode = BOT_MODE_NONE or 0
  local enemies_ok, enemies = pcall(function() return bot:GetNearbyHeroes(16000, true, mode) end)
  local allies_ok, allies = pcall(function() return bot:GetNearbyHeroes(16000, false, mode) end)
  if not enemies_ok or not allies_ok then return nil end
  local enemy_distance, enemies_nearby = spatial_context(enemies, location.x, location.y)
  local ally_distance, allies_nearby = spatial_context(allies, location.x, location.y)
  local counters = read_counters()
  local hero_ok, hero_id = pcall(policy.hero_id, bot:GetUnitName())
  if not hero_ok then hero_id = 0 end
  local movement = 0
  if combat_tracker.x ~= nil then
    movement = math.sqrt((location.x - combat_tracker.x) ^ 2 + (location.y - combat_tracker.y) ^ 2) / 64
  end
  local state = {
    time_minutes = now / 60,
    x = location.x / 64 + 128,
    y = location.y / 64 + 128,
    alive = 1,
    level = counters.level,
    gold = counters.gold,
    lh = counters.last_hits,
    denies = counters.denies,
    kills = counters.kills,
    deaths = counters.deaths,
    assists = safe_number(function() return GetHeroAssists(bot:GetPlayerID()) end, 0),
    movement_distance = movement,
    previous_move = last_order.name == "move" and 1 or 0,
    previous_attack = last_order.name == "attack" and 1 or 0,
    previous_cast = last_order.name == "cast" and 1 or 0,
    nearest_ally_distance = ally_distance,
    nearest_enemy_distance = enemy_distance,
    allies_nearby = allies_nearby,
    enemies_nearby = enemies_nearby,
    team_id = safe_number(function() return bot:GetTeam() end, 2) == 2 and 0 or 1,
    hero_id = hero_id,
  }
  local predicted, result = pcall(combat_policy.predict, state)
  if not predicted or type(result) ~= "table" then
    telemetry("combat_prediction_error", { error = result })
    return nil
  end
  combat_tracker.time = now
  combat_tracker.x = location.x
  combat_tracker.y = location.y
  combat_tracker.prediction = result
  telemetry("combat_prediction", {
    engage_probability = result.engage_probability,
    threat_probability = result.threat_probability,
  })
  return result
end

local function conservative_combat_opportunity(prediction)
  if prediction == nil or type(prediction.engage_probability) ~= "number"
    or prediction.engage_probability < 0.90 then return false end
  local mode = BOT_MODE_NONE or 0
  local ok, enemies = pcall(function() return bot:GetNearbyHeroes(900, true, mode) end)
  if not ok or enemies == nil or #enemies == 0 then return false end
  local attacked = attack(weakest(enemies), "replay_combat_high_confidence")
  if attacked then observed.fight = 1 end
  return attacked
end

local function push()
  local ok, towers = pcall(function() return bot:GetNearbyTowers(2200, true) end)
  if not ok then
    telemetry("query_error", { action = "push", error = towers })
    return farm("push_tower_query_error")
  end
  if attack(weakest(towers), nil) then
    observed.push = 1
    return true
  end
  return farm("push_no_tower")
end

local function safe_push_opportunity()
  local towers_ok, towers = pcall(function() return bot:GetNearbyTowers(900, true) end)
  if not towers_ok or towers == nil or #towers == 0 then return false end
  local mode = BOT_MODE_NONE or 0
  local enemies_ok, enemies = pcall(function() return bot:GetNearbyHeroes(1200, true, mode) end)
  if not enemies_ok or enemies == nil or #enemies > 0 then return false end
  local creeps_ok, allied_creeps = pcall(function()
    return bot:GetNearbyLaneCreeps(900, false)
  end)
  if not creeps_ok or allied_creeps == nil or #allied_creeps == 0 then return false end
  if not attack(weakest(towers), "safe_push_opportunity") then return false end
  observed.push = 1
  return true
end

local function move_to_ancient(fallback)
  local ancient_ok, ancient = pcall(function() return GetAncient(bot:GetTeam()) end)
  if not ancient_ok or ancient == nil then
    telemetry("query_error", { action = "unknown", fallback = fallback, error = ancient })
    return false
  end
  local moved, err = pcall(function() bot:Action_MoveToLocation(ancient:GetLocation()) end)
  if not moved then
    telemetry("order_error", { order = "move", target = "ancient", fallback = fallback, error = err })
    return false
  end
  record_order("move", "ancient", fallback)
  return true
end

local function retreat_for_survival()
  local health, health_ok = safe_number(function() return bot:GetHealth() end, 0)
  local max_health, max_health_ok = safe_number(function() return bot:GetMaxHealth() end, 0)
  if not health_ok or not max_health_ok or max_health <= 0 then return false end
  local health_ratio = health / max_health
  if health_ratio > 0.25 then return false end
  local mode = BOT_MODE_NONE or 0
  local enemies_ok, enemies = pcall(function() return bot:GetNearbyHeroes(1600, true, mode) end)
  if health_ratio > 0.15 and (not enemies_ok or enemies == nil or #enemies == 0) then return false end
  return move_to_ancient("low_health")
end

local function retreat_if_threatened(fallback)
  local mode = BOT_MODE_NONE or 0
  local ok, enemies = pcall(function() return bot:GetNearbyHeroes(1200, true, mode) end)
  if not ok then
    telemetry("query_error", { action = "unknown", fallback = fallback, error = enemies })
    return farm("enemy_query_error")
  end
  if enemies == nil or #enemies == 0 then return farm(fallback or "unknown_no_threat") end
  if move_to_ancient(fallback) then return true end
  return farm("retreat_unavailable")
end

local function choose_action()
  local state, available, missing = checkpoint_state()
  if policy == nil then
    telemetry("decision", add_state_fields({
      action = "unknown", fallback = "policy_unavailable",
      features_available = available, missing_features = missing,
    }, state))
    return "unknown"
  end
  local ok, action = pcall(policy.predict, state)
  if not ok then
    telemetry("decision_error", add_state_fields({
      action = "unknown", fallback = "policy_error", error = action,
      features_available = available, missing_features = missing,
    }, state))
    return "unknown"
  end
  if action ~= "farm" and action ~= "fight" and action ~= "push" and action ~= "unknown" then
    telemetry("decision_error", add_state_fields({
      action = "unknown", fallback = "invalid_policy_action", error = action,
      features_available = available, missing_features = missing,
    }, state))
    return "unknown"
  end
  telemetry("decision", add_state_fields({
    action = action, features_available = available, missing_features = missing,
  }, state))
  return action
end

local function member_number(member, callback, fallback)
  local ok, value = pcall(callback, member)
  if ok and type(value) == "number" then return value end
  return fallback
end

local function choose_team_action()
  if team_policy == nil then return nil end
  local states = {}
  local own_index = nil
  local own_player = safe_number(function() return bot:GetPlayerID() end, -1)
  local minute = game_minute()
  local team_number = safe_number(function() return bot:GetTeam() end, 2)
  for index = 1, 5 do
    local ok, member = pcall(function() return GetTeamMember(index) end)
    if not ok or member == nil then return nil end
    local player_id = member_number(member, function(unit) return unit:GetPlayerID() end, -1)
    if player_id == own_player then own_index = index end
    local gold = member_number(member, function(unit) return unit:GetGold() end, 0)
    local last_hits = member_number(member, function(unit) return unit:GetLastHits() end, 0)
    local kills = safe_number(function() return GetHeroKills(player_id) end, 0)
    local previous = team_tracker.previous_actions[index] or "unknown"
    local unit_ok, unit_name = pcall(function() return member:GetUnitName() end)
    local hero_id = 0
    if unit_ok then
      local hero_ok, value = pcall(team_policy.hero_id, unit_name)
      if hero_ok and type(value) == "number" then hero_id = value end
    end
    states[index] = {
      hero_id = hero_id,
      team = team_number == 2 and "Radiant" or "Dire",
      state_minute = minute,
      gold = gold,
      experience = 0,
      last_hits = last_hits,
      gold_change = team_tracker.gold[index] ~= nil and gold - team_tracker.gold[index] or 0,
      experience_change = 0,
      last_hit_change = team_tracker.last_hits[index] ~= nil
        and last_hits - team_tracker.last_hits[index] or 0,
      team_gold_advantage = 0,
      team_experience_advantage = 0,
      kills_last_minute = team_tracker.kills[index] ~= nil
        and kills - team_tracker.kills[index] or 0,
      previous_fight = previous == "fight" and 1 or 0,
      previous_push = previous == "push" and 1 or 0,
      previous_farm = previous == "farm" and 1 or 0,
    }
    team_tracker.gold[index] = gold
    team_tracker.last_hits[index] = last_hits
    team_tracker.kills[index] = kills
  end
  if own_index == nil then return nil end
  local modulus = 2147483647
  local random_value = ((minute * 1103515245 + team_number * 12345) % modulus) / modulus
  local predicted, result = pcall(team_policy.predict, states, random_value)
  if not predicted or type(result) ~= "table" or type(result.actions) ~= "table" then
    telemetry("team_selfplay_prediction_error", { error = result })
    return nil
  end
  local action = result.actions[own_index]
  if action ~= "farm" and action ~= "fight" and action ~= "push" and action ~= "unknown" then
    telemetry("team_selfplay_prediction_error", { error = "invalid team action" })
    return nil
  end
  for index = 1, 5 do team_tracker.previous_actions[index] = result.actions[index] end
  telemetry("team_selfplay_decision", { action = action })
  return action
end

function Think()
  if bot == nil then return end
  local minute = game_minute()
  if minute ~= selected_minute then
    selected_action = choose_team_action() or choose_action()
    selected_minute = minute
  end
  local alive_ok, alive = pcall(function() return bot:IsAlive() end)
  if not alive_ok or not alive then
    activity.last_sample = nil
    return
  end
  sample_activity()
  if retreat_for_survival() then return end
  local combat = combat_prediction()
  if selected_action ~= "fight" and conservative_combat_opportunity(combat) then return end
  if selected_action ~= "fight" and safe_push_opportunity() then return end
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
