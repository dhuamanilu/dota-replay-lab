"""Multi-seed audit for a coordinated self-play checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .selfplay_env import ACTIONS, ReplayCalibration
from .train_selfplay import _wilson
from .train_team_selfplay import evaluate_team_matchup, load_team_initialization, make_team_policy


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(item["wins"] for item in results)
    losses = sum(item["losses"] for item in results)
    draws = sum(item["draws"] for item in results)
    games = wins + losses + draws
    decisive = wins + losses
    action_distribution = {
        action: sum(item["action_distribution"][action] * item["games"] for item in results)
        / games
        for action in ACTIONS
    }
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": (wins + 0.5 * draws) / games,
        "decisive_win_rate": wins / decisive if decisive else 0.5,
        "wilson_95": _wilson(wins, decisive),
        "action_distribution": action_distribution,
        "per_seed": results,
    }


def audit_team_checkpoint(
    checkpoint: Path,
    initial_checkpoint: Path,
    training_metrics: Path,
    output: Path,
    *,
    seeds: list[int],
    games_per_seed: int = 400,
    device: str = "auto",
) -> dict[str, Any]:
    import numpy as np
    import torch

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested but unavailable")
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    model = make_team_policy(int(bundle["maximum_hero_id"])).to(torch_device)
    model.load_state_dict(bundle["state_dict"])
    model.recurrent.flatten_parameters()
    model.eval()
    original, _ = load_team_initialization(initial_checkpoint, torch_device)
    original.eval()
    symmetry_opponent = copy.deepcopy(model).to(torch_device)
    symmetry_opponent.recurrent.flatten_parameters()
    symmetry_opponent.eval()
    calibration = ReplayCalibration.from_dict(bundle["calibration"])
    means = np.asarray(bundle["means"], dtype=np.float32)
    scales = np.asarray(bundle["scales"], dtype=np.float32)
    metrics = json.loads(training_metrics.read_text(encoding="utf-8"))
    max_minutes = int(metrics["max_minutes"])
    opponents: dict[str, Any] = {
        "original_imitation": (original, "independent"),
        "scripted_balanced": "balanced",
        "scripted_aggressive": "aggressive",
        "scripted_farm": "farm",
        "symmetry_control": symmetry_opponent,
    }
    evaluation = {}
    for opponent_index, (name, opponent) in enumerate(opponents.items()):
        runs = [
            evaluate_team_matchup(
                model,
                opponent,
                calibration,
                means,
                scales,
                games=games_per_seed,
                seed=seed + opponent_index * 1_000_003,
                max_minutes=max_minutes,
                device=torch_device,
            )
            for seed in seeds
        ]
        evaluation[name] = _aggregate(runs)
    minimum_share = metrics["minimum_action_share"]
    performance = all(
        evaluation[name]["wilson_95"][0] > 0.5
        for name in (
            "original_imitation",
            "scripted_balanced",
            "scripted_aggressive",
            "scripted_farm",
        )
    )
    symmetry_interval = evaluation["symmetry_control"]["wilson_95"]
    symmetry = symmetry_interval[0] <= 0.5 <= symmetry_interval[1]
    diversity = all(
        all(
            evaluation[name]["action_distribution"][action] >= minimum_share[action]
            for action in ACTIONS
        )
        for name in (
            "original_imitation",
            "scripted_balanced",
            "scripted_aggressive",
            "scripted_farm",
        )
    )
    retention = metrics["replay_imitation"]["macro_f1_delta"] >= -0.01
    result = {
        "checkpoint": str(checkpoint),
        "device": str(torch_device),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "seeds": seeds,
        "games_per_seed": games_per_seed,
        "evaluation": evaluation,
        "replay_imitation": metrics["replay_imitation"],
        "minimum_action_share": minimum_share,
        "promotion_gate": {
            "performance_passed": performance,
            "symmetry_passed": symmetry,
            "noncollapse_passed": diversity,
            "replay_retention_passed": retention,
            "passed": performance and symmetry and diversity and retention,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/team-selfplay-models/team-selfplay-policy-v1.pt"))
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("artifacts/sequence-models/sequence-policy-v1.pt"))
    parser.add_argument("--training-metrics", type=Path, default=Path("artifacts/team-selfplay-models/team-selfplay-policy-v1.metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/team-selfplay-models/team-selfplay-policy-v1.audit.json"))
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--games-per-seed", type=int, default=400)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_team_checkpoint(
        args.checkpoint,
        args.initial_checkpoint,
        args.training_metrics,
        args.output,
        seeds=[int(value) for value in args.seeds.split(",")],
        games_per_seed=args.games_per_seed,
        device=args.device,
    )
    print(json.dumps({
        "promotion": result["promotion_gate"]["passed"],
        "win_rates": {
            name: values["decisive_win_rate"]
            for name, values in result["evaluation"].items()
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
