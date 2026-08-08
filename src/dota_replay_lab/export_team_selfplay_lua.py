"""Export the coordinated recurrent self-play policy to standalone Lua."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .export_sequence_lua import _matrix, _number, _vector
from .train_policy import LABELS, NUMERIC_FEATURES
from .train_team_selfplay import COMPOSITIONS, assign_compositions, composition_distribution, make_team_policy


def render_team_selfplay_lua(bundle: dict[str, Any], hero_names: dict[int, str]) -> str:
    state = bundle["state_dict"]

    def array(name: str) -> Any:
        return state[name].detach().cpu().numpy()

    hero_lines = ["local hero_ids = {"]
    for hero_id, name in sorted(hero_names.items()):
        hero_lines.append(f"  [{json.dumps(name)}] = {hero_id},")
    hero_lines.append("}")
    blocks = [
        *hero_lines,
        f"local feature_names = {json.dumps(list(NUMERIC_FEATURES)).replace('[', '{').replace(']', '}')}",
        f"local labels = {json.dumps(list(LABELS)).replace('[', '{').replace(']', '}')}",
        f"local means = {_vector(bundle['means'])}",
        f"local scales = {_vector(bundle['scales'])}",
        f"local hero_embedding = {_matrix(array('hero_embedding.weight'))}",
        f"local team_embedding = {_matrix(array('team_embedding.weight'))}",
    ]
    for layer in (0, 1):
        blocks.extend(
            [
                f"local w_ih_{layer} = {_matrix(array(f'recurrent.weight_ih_l{layer}'))}",
                f"local w_hh_{layer} = {_matrix(array(f'recurrent.weight_hh_l{layer}'))}",
                f"local b_ih_{layer} = {_vector(array(f'recurrent.bias_ih_l{layer}'))}",
                f"local b_hh_{layer} = {_vector(array(f'recurrent.bias_hh_l{layer}'))}",
            ]
        )
    blocks.extend(
        [
            f"local hero_norm_weight = {_vector(array('head.0.weight'))}",
            f"local hero_norm_bias = {_vector(array('head.0.bias'))}",
            f"local hero_head1_weight = {_matrix(array('head.1.weight'))}",
            f"local hero_head1_bias = {_vector(array('head.1.bias'))}",
            f"local hero_head2_weight = {_matrix(array('head.4.weight'))}",
            f"local hero_head2_bias = {_vector(array('head.4.bias'))}",
            f"local composition_norm_weight = {_vector(array('composition_head.0.weight'))}",
            f"local composition_norm_bias = {_vector(array('composition_head.0.bias'))}",
            f"local composition_head1_weight = {_matrix(array('composition_head.1.weight'))}",
            f"local composition_head1_bias = {_vector(array('composition_head.1.bias'))}",
            f"local composition_head2_weight = {_matrix(array('composition_head.3.weight'))}",
            f"local composition_head2_bias = {_vector(array('composition_head.3.bias'))}",
            f"local composition_prior = {_vector(array('composition_prior'))}",
            "local compositions = {"
            + ",".join("{" + ",".join(str(value) for value in row) + "}" for row in COMPOSITIONS)
            + "}",
        ]
    )
    logic = r'''
local M = {}
local hidden_0 = {}
local hidden_1 = {}

local function zeros(size)
  local values = {}
  for index = 1, size do values[index] = 0 end
  return values
end

function M.reset()
  hidden_0 = {}
  hidden_1 = {}
  for slot = 1, 5 do hidden_0[slot] = zeros(96); hidden_1[slot] = zeros(96) end
end

function M.hero_id(unit_name) return hero_ids[unit_name] or 0 end

local function dot(row, values)
  local total = 0
  for index = 1, #values do total = total + row[index] * values[index] end
  return total
end

local function sigmoid(value)
  if value >= 0 then local z = math.exp(-value); return 1 / (1 + z) end
  local z = math.exp(value); return z / (1 + z)
end

local function tanh(value)
  if value > 20 then return 1 end
  if value < -20 then return -1 end
  local doubled = math.exp(2 * value)
  return (doubled - 1) / (doubled + 1)
end

local function gru(input, hidden, w_ih, w_hh, b_ih, b_hh)
  local output = {}
  for index = 1, 96 do
    local reset = sigmoid(dot(w_ih[index], input) + b_ih[index] + dot(w_hh[index], hidden) + b_hh[index])
    local update_index = 96 + index
    local update = sigmoid(dot(w_ih[update_index], input) + b_ih[update_index] + dot(w_hh[update_index], hidden) + b_hh[update_index])
    local candidate_index = 192 + index
    local candidate = tanh(dot(w_ih[candidate_index], input) + b_ih[candidate_index] + reset * (dot(w_hh[candidate_index], hidden) + b_hh[candidate_index]))
    output[index] = (1 - update) * candidate + update * hidden[index]
  end
  return output
end

local function linear(input, weight, bias, relu)
  local output = {}
  for index = 1, #weight do
    local value = dot(weight[index], input) + bias[index]
    output[index] = relu and math.max(value, 0) or value
  end
  return output
end

local function layer_norm(input, weight, bias)
  local mean = 0
  for index = 1, #input do mean = mean + input[index] end
  mean = mean / #input
  local variance = 0
  for index = 1, #input do local centered = input[index] - mean; variance = variance + centered * centered end
  variance = variance / #input
  local denominator = math.sqrt(variance + 0.00001)
  local output = {}
  for index = 1, #input do output[index] = ((input[index] - mean) / denominator) * weight[index] + bias[index] end
  return output
end

local function build_input(state)
  local input = {}
  for index, name in ipairs(feature_names) do input[index] = ((tonumber(state[name]) or 0) - means[index]) / scales[index] end
  local hero_index = math.max(0, math.min(#hero_embedding - 1, tonumber(state.hero_id) or 0)) + 1
  for _, value in ipairs(hero_embedding[hero_index]) do table.insert(input, value) end
  local team_index = state.team == "Dire" and 2 or 1
  for _, value in ipairs(team_embedding[team_index]) do table.insert(input, value) end
  return input
end

local function softmax_with_prior(logits)
  local maximum = logits[1]
  for index = 2, #logits do maximum = math.max(maximum, logits[index]) end
  local total, exponentials = 0, {}
  for index = 1, #logits do exponentials[index] = math.exp(logits[index] - maximum); total = total + exponentials[index] end
  local probabilities = {}
  for index = 1, #logits do probabilities[index] = 0.75 * exponentials[index] / total + 0.25 * composition_prior[index] end
  return probabilities
end

local function choose_composition(probabilities, random_value)
  local target, cumulative = math.max(0, math.min(0.999999999999, tonumber(random_value) or 0)), 0
  for index, probability in ipairs(probabilities) do cumulative = cumulative + probability; if target < cumulative then return index end end
  return #probabilities
end

local function best_assignment(hero_logits, composition_index)
  local target = compositions[composition_index]
  local best, best_score = nil, nil
  for a = 1, 4 do for b = 1, 4 do for c = 1, 4 do for d = 1, 4 do for e = 1, 4 do
    local actions = {a, b, c, d, e}
    local counts = {0, 0, 0, 0}
    for slot = 1, 5 do counts[actions[slot]] = counts[actions[slot]] + 1 end
    if counts[1] == target[1] and counts[2] == target[2] and counts[3] == target[3] and counts[4] == target[4] then
      local score = 0
      for slot = 1, 5 do score = score + hero_logits[slot][actions[slot]] end
      if best_score == nil or score > best_score then best_score, best = score, actions end
    end
  end end end end end
  return best
end

function M.predict(states, random_value)
  if #states ~= 5 then error("team policy requires exactly five hero states") end
  local hero_logits, team_encoded = {}, {}
  for slot = 1, 5 do
    hidden_0[slot] = gru(build_input(states[slot]), hidden_0[slot], w_ih_0, w_hh_0, b_ih_0, b_hh_0)
    hidden_1[slot] = gru(hidden_0[slot], hidden_1[slot], w_ih_1, w_hh_1, b_ih_1, b_hh_1)
    for _, value in ipairs(hidden_1[slot]) do table.insert(team_encoded, value) end
    local normalized = layer_norm(hidden_1[slot], hero_norm_weight, hero_norm_bias)
    hero_logits[slot] = linear(linear(normalized, hero_head1_weight, hero_head1_bias, true), hero_head2_weight, hero_head2_bias, false)
  end
  local normalized_team = layer_norm(team_encoded, composition_norm_weight, composition_norm_bias)
  local logits = linear(linear(normalized_team, composition_head1_weight, composition_head1_bias, true), composition_head2_weight, composition_head2_bias, false)
  local probabilities = softmax_with_prior(logits)
  local composition_index = choose_composition(probabilities, random_value)
  local assignment = best_assignment(hero_logits, composition_index)
  local actions = {}
  for slot = 1, 5 do actions[slot] = labels[assignment[slot]] end
  return { actions = actions, probabilities = probabilities, composition_index = composition_index }
end

M.reset()
return M
'''
    return "-- Generated by dota_replay_lab.export_team_selfplay_lua. Do not edit.\n" + "\n\n".join(blocks) + "\n" + logic


def verify_team_lua_parity(
    source: str,
    bundle: dict[str, Any],
    dataset: Path,
    test_match_ids: Iterable[int],
    *,
    limit: int = 250,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from lupa import LuaRuntime

    frame = pd.read_csv(dataset)
    selected = frame[frame["match_id"].isin(set(test_match_ids))]
    model = make_team_policy(int(bundle["maximum_hero_id"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    lua = LuaRuntime(unpack_returned_tuples=True)
    policy = lua.execute(source)
    rng = np.random.default_rng(20260808)
    compared, action_matches, largest_error = 0, 0, 0.0
    with torch.no_grad():
        for _, timeline in selected.groupby(["match_id", "team"], sort=True):
            policy.reset()
            hidden = torch.zeros(2, 5, 96)
            for _, rows in timeline.groupby("state_minute", sort=True):
                rows = rows.sort_values("player_slot")
                if len(rows) != 5:
                    continue
                raw = rows.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
                numeric = torch.from_numpy((raw - bundle["means"]) / bundle["scales"])
                heroes = torch.from_numpy(rows["hero_id"].to_numpy(dtype=np.int64).copy())
                team_ids = torch.full((5,), 0 if rows.iloc[0]["team"] == "Radiant" else 1)
                encoded, hidden = model.encode(numeric[:, None], heroes, team_ids, hidden)
                hero_logits = model.head(encoded[:, 0]).unsqueeze(0)
                team_encoded = encoded[:, 0].reshape(1, 480)
                logits = model.composition_head(team_encoded)
                distribution = composition_distribution(model, logits)
                probabilities = distribution.probs[0].numpy()
                random_value = float(rng.random())
                composition_index = int(np.searchsorted(np.cumsum(probabilities), random_value, side="right"))
                expected = assign_compositions(hero_logits, torch.tensor([composition_index]))[0].tolist()
                states = []
                for row_index, (_, row) in enumerate(rows.iterrows()):
                    state = {
                        name: float(raw[row_index, feature_index])
                        for feature_index, name in enumerate(NUMERIC_FEATURES)
                    }
                    state.update({"hero_id": int(row["hero_id"]), "team": str(row["team"])})
                    states.append(lua.table_from(state))
                actual = policy.predict(lua.table_from(states), random_value)
                lua_probabilities = np.asarray(
                    [actual["probabilities"][index] for index in range(1, len(COMPOSITIONS) + 1)]
                )
                largest_error = max(largest_error, float(np.max(np.abs(probabilities - lua_probabilities))))
                actual_actions = [actual["actions"][slot] for slot in range(1, 6)]
                expected_actions = [LABELS[index] for index in expected]
                action_matches += int(actual_actions == expected_actions)
                compared += 1
                if compared >= limit:
                    return {
                        "team_minutes": compared,
                        "action_matches": action_matches,
                        "fidelity": action_matches / compared,
                        "largest_probability_error": largest_error,
                    }
    return {
        "team_minutes": compared,
        "action_matches": action_matches,
        "fidelity": action_matches / compared if compared else 0,
        "largest_probability_error": largest_error,
    }


def export_team_policy(
    checkpoint: Path,
    training_metrics: Path,
    audit_path: Path,
    dataset: Path,
    manifest: Path,
    output: Path,
    *,
    parity_rows: int = 250,
) -> dict[str, Any]:
    import torch

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["promotion_gate"]["passed"]:
        raise ValueError("Refusing to export a team policy that failed the multi-seed gate")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    hero_names = {
        int(hero_id): str(name)
        for hero_id, name in manifest_data.get("hero_internal_names", {}).items()
    }
    source = render_team_selfplay_lua(bundle, hero_names)
    training = json.loads(training_metrics.read_text(encoding="utf-8"))
    parity = verify_team_lua_parity(
        source, bundle, dataset, training["match_ids"]["test"], limit=parity_rows
    )
    if parity["fidelity"] != 1.0 or parity["largest_probability_error"] > 1e-6:
        raise RuntimeError(f"Team Lua parity failed: {parity}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    result = {
        "policy_type": "coordinated_recurrent_ppo_selfplay",
        "source_model": str(checkpoint),
        "training": training,
        "multi_seed_audit": audit,
        "lua_parity": parity,
        "lua_bytes": output.stat().st_size,
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/team-selfplay-models/team-selfplay-policy-v1.pt"))
    parser.add_argument("--training-metrics", type=Path, default=Path("artifacts/team-selfplay-models/team-selfplay-policy-v1.metrics.json"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/team-selfplay-models/team-selfplay-policy-v1.audit.json"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v3.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/corpora/pro-matches-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("bots/team_selfplay_policy.lua"))
    parser.add_argument("--parity-rows", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_team_policy(
        args.checkpoint, args.training_metrics, args.audit, args.dataset, args.manifest,
        args.output, parity_rows=args.parity_rows,
    )
    print(json.dumps({"parity": result["lua_parity"], "bytes": result["lua_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
