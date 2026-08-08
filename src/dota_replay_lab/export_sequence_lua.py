"""Export a trained recurrent policy as standalone Lua and verify parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .train_policy import LABELS, NUMERIC_FEATURES
from .train_sequence_policy import build_sequences, make_recurrent_policy


def _number(value: Any) -> str:
    return f"{float(value):.17g}"


def _vector(values: Iterable[Any]) -> str:
    return "{" + ",".join(_number(value) for value in values) + "}"


def _matrix(values: Any) -> str:
    return "{\n" + "\n".join(f"  {_vector(row)}," for row in values) + "\n}"


def render_recurrent_lua(bundle: dict[str, Any], hero_internal_names: dict[int, str]) -> str:
    """Render PyTorch GRU weights using the exact inference equations."""

    state = bundle["state_dict"]

    def array(name: str) -> Any:
        return state[name].detach().cpu().numpy()

    hero_lines = ["local hero_ids = {"]
    for hero_id, internal_name in sorted(hero_internal_names.items()):
        hero_lines.append(f"  [{json.dumps(internal_name)}] = {hero_id},")
    hero_lines.append("}")
    data_blocks = [
        *hero_lines,
        f"local feature_names = {json.dumps(list(NUMERIC_FEATURES), ensure_ascii=False).replace('[', '{').replace(']', '}')}",
        f"local means = {_vector(bundle['means'])}",
        f"local scales = {_vector(bundle['scales'])}",
        f"local hero_embedding = {_matrix(array('hero_embedding.weight'))}",
        f"local team_embedding = {_matrix(array('team_embedding.weight'))}",
    ]
    for layer in (0, 1):
        data_blocks.extend(
            [
                f"local w_ih_{layer} = {_matrix(array(f'recurrent.weight_ih_l{layer}'))}",
                f"local w_hh_{layer} = {_matrix(array(f'recurrent.weight_hh_l{layer}'))}",
                f"local b_ih_{layer} = {_vector(array(f'recurrent.bias_ih_l{layer}'))}",
                f"local b_hh_{layer} = {_vector(array(f'recurrent.bias_hh_l{layer}'))}",
            ]
        )
    data_blocks.extend(
        [
            f"local norm_weight = {_vector(array('head.0.weight'))}",
            f"local norm_bias = {_vector(array('head.0.bias'))}",
            f"local head1_weight = {_matrix(array('head.1.weight'))}",
            f"local head1_bias = {_vector(array('head.1.bias'))}",
            f"local head2_weight = {_matrix(array('head.4.weight'))}",
            f"local head2_bias = {_vector(array('head.4.bias'))}",
            f"local labels = {json.dumps(list(LABELS)).replace('[', '{').replace(']', '}')}",
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
  hidden_0 = zeros(96)
  hidden_1 = zeros(96)
end

function M.hero_id(unit_name)
  return hero_ids[unit_name] or 0
end

local function dot(row, values)
  local total = 0
  for index = 1, #values do total = total + row[index] * values[index] end
  return total
end

local function sigmoid(value)
  if value >= 0 then
    local z = math.exp(-value)
    return 1 / (1 + z)
  end
  local z = math.exp(value)
  return z / (1 + z)
end

local function hyperbolic_tangent(value)
  if value > 20 then return 1 end
  if value < -20 then return -1 end
  local doubled = math.exp(2 * value)
  return (doubled - 1) / (doubled + 1)
end

local function gru_step(input, hidden, w_ih, w_hh, b_ih, b_hh)
  local next_hidden = {}
  for index = 1, 96 do
    local reset = sigmoid(
      dot(w_ih[index], input) + b_ih[index]
      + dot(w_hh[index], hidden) + b_hh[index]
    )
    local update_index = 96 + index
    local update = sigmoid(
      dot(w_ih[update_index], input) + b_ih[update_index]
      + dot(w_hh[update_index], hidden) + b_hh[update_index]
    )
    local candidate_index = 192 + index
    local candidate = hyperbolic_tangent(
      dot(w_ih[candidate_index], input) + b_ih[candidate_index]
      + reset * (dot(w_hh[candidate_index], hidden) + b_hh[candidate_index])
    )
    next_hidden[index] = (1 - update) * candidate + update * hidden[index]
  end
  return next_hidden
end

local function linear(input, weight, bias, relu)
  local output = {}
  for index = 1, #weight do
    local value = dot(weight[index], input) + bias[index]
    output[index] = relu and math.max(0, value) or value
  end
  return output
end

local function layer_norm(input)
  local mean = 0
  for index = 1, #input do mean = mean + input[index] end
  mean = mean / #input
  local variance = 0
  for index = 1, #input do
    local centered = input[index] - mean
    variance = variance + centered * centered
  end
  variance = variance / #input
  local denominator = math.sqrt(variance + 0.00001)
  local output = {}
  for index = 1, #input do
    output[index] = ((input[index] - mean) / denominator) * norm_weight[index] + norm_bias[index]
  end
  return output
end

local function build_input(state)
  local input = {}
  for index, name in ipairs(feature_names) do
    input[index] = ((tonumber(state[name]) or 0) - means[index]) / scales[index]
  end
  local hero_index = math.max(0, math.min(#hero_embedding - 1, tonumber(state.hero_id) or 0)) + 1
  for _, value in ipairs(hero_embedding[hero_index]) do table.insert(input, value) end
  local team_index = state.team == "Dire" and 2 or 1
  for _, value in ipairs(team_embedding[team_index]) do table.insert(input, value) end
  return input
end

function M.predict(state)
  hidden_0 = gru_step(build_input(state), hidden_0, w_ih_0, w_hh_0, b_ih_0, b_hh_0)
  hidden_1 = gru_step(hidden_0, hidden_1, w_ih_1, w_hh_1, b_ih_1, b_hh_1)
  local normalized = layer_norm(hidden_1)
  local intermediate = linear(normalized, head1_weight, head1_bias, true)
  local logits = linear(intermediate, head2_weight, head2_bias, false)
  local best = 1
  for index = 2, #logits do
    if logits[index] > logits[best] then best = index end
  end
  return labels[best]
end

M.reset()
return M
'''
    return "-- Generated by dota_replay_lab.export_sequence_lua. Do not edit.\n" + "\n\n".join(data_blocks) + "\n" + logic


def verify_lua_parity(
    source: str, bundle: dict[str, Any], dataset: Path, test_match_ids: Iterable[int], limit: int = 5000
) -> dict[str, Any]:
    """Compare causal PyTorch and Lua predictions on frozen test sequences."""

    import numpy as np
    import pandas as pd
    import torch

    try:
        import lupa
    except ImportError as error:
        raise RuntimeError("Lua parity verification requires the optional dev dependencies.") from error
    frame = pd.read_csv(dataset)
    sequences = build_sequences(frame, test_match_ids, bundle["means"], bundle["scales"])
    model = make_recurrent_policy(int(bundle["maximum_hero_id"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    policy = lua.execute(source)
    compared = 0
    matches = 0
    with torch.no_grad():
        for sequence in sequences:
            numeric = torch.from_numpy(sequence["numeric"]).unsqueeze(0)
            heroes = torch.tensor([sequence["hero_id"]], dtype=torch.long)
            teams = torch.tensor([sequence["team_id"]], dtype=torch.long)
            expected = model(numeric, heroes, teams).argmax(dim=-1).squeeze(0).tolist()
            policy.reset()
            raw_rows = frame[
                (frame["match_id"] == sequence["match_id"])
                & (frame["player_slot"] == sequence["player_slot"])
            ].sort_values("state_minute")
            for expected_index, (_, row) in zip(expected, raw_rows.iterrows()):
                state = {name: float(row[name]) for name in NUMERIC_FEATURES}
                state.update(
                    {
                        "hero_id": int(row["hero_id"]),
                        "team": str(row["team"]),
                    }
                )
                actual = policy.predict(lua.table_from(state))
                matches += int(actual == LABELS[expected_index])
                compared += 1
                if compared >= limit:
                    return {"rows": compared, "matches": matches, "fidelity": matches / compared}
    return {"rows": compared, "matches": matches, "fidelity": matches / compared if compared else 0}


def export_sequence_policy(
    model_path: Path,
    metrics_path: Path,
    dataset: Path,
    manifest_path: Path,
    output: Path,
    parity_rows: int = 5000,
) -> dict[str, Any]:
    import torch

    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hero_names = {
        int(hero_id): str(name)
        for hero_id, name in manifest.get("hero_internal_names", {}).items()
    }
    source = render_recurrent_lua(bundle, hero_names)
    training = json.loads(metrics_path.read_text(encoding="utf-8"))
    parity = verify_lua_parity(
        source, bundle, dataset, training["match_ids"]["test"], parity_rows
    )
    result = {
        "policy_type": "recurrent_gru",
        "source_model": str(model_path),
        "best_epoch": training["best_epoch"],
        "device": training["device"],
        "gpu": training.get("gpu"),
        "train_seconds": training["train_seconds"],
        "validation_macro_f1": training["best_validation_macro_f1"],
        "test": training["test"],
        "lua_parity": parity,
        "lua_bytes": len(source.encode("utf-8")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/sequence-models/sequence-policy-v1.pt")
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/sequence-models/sequence-policy-v1.metrics.json"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v3.csv")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/corpora/pro-matches-v1.json")
    )
    parser.add_argument("--output", type=Path, default=Path("bots/decision_policy.lua"))
    parser.add_argument("--parity-rows", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_sequence_policy(
        args.model, args.metrics, args.dataset, args.manifest, args.output, args.parity_rows
    )
    print(f"Saved recurrent Lua policy: {args.output}")
    print(
        f"Lua parity: {result['lua_parity']['matches']}/{result['lua_parity']['rows']} "
        f"({result['lua_parity']['fidelity']:.6f})"
    )
    print(f"Lua bytes: {result['lua_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
