"""Coordinated recurrent PPO self-play over all 56 five-hero compositions."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from .selfplay_env import ACTIONS, DotaSelfPlayEnv, fit_replay_calibration
from .train_policy import LABELS, split_match_ids
from .train_selfplay import (
    _wilson,
    calibration_audit,
    evaluate_replay_imitation,
    replay_regularization_update,
    scripted_actions,
)


COMPOSITIONS = tuple(
    (farm, fight, push, 5 - farm - fight - push)
    for farm in range(6)
    for fight in range(6 - farm)
    for push in range(6 - farm - fight)
)
JOINT_ACTIONS = tuple(itertools.product(range(4), repeat=5))
JOINT_COMPOSITION = tuple(
    COMPOSITIONS.index(tuple(actions.count(index) for index in range(4)))
    for actions in JOINT_ACTIONS
)


def make_team_policy(maximum_hero_id: int) -> Any:
    import torch
    from torch import nn

    class TeamPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hero_embedding = nn.Embedding(maximum_hero_id + 1, 16)
            self.team_embedding = nn.Embedding(2, 2)
            self.recurrent = nn.GRU(31, 96, num_layers=2, dropout=0.0, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(96),
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(64, 4),
            )
            self.value_head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1))
            self.composition_head = nn.Sequential(
                nn.LayerNorm(480), nn.Linear(480, 192), nn.ReLU(), nn.Linear(192, len(COMPOSITIONS))
            )
            self.team_value_head = nn.Sequential(
                nn.LayerNorm(480), nn.Linear(480, 96), nn.ReLU(), nn.Linear(96, 1)
            )
            self.register_buffer(
                "composition_prior", torch.full((len(COMPOSITIONS),), 1 / len(COMPOSITIONS))
            )

        def encode(
            self, numeric: Any, heroes: Any, teams: Any, hidden: Any | None = None
        ) -> tuple[Any, Any]:
            steps = numeric.shape[1]
            hero = self.hero_embedding(heroes).unsqueeze(1).expand(-1, steps, -1)
            team = self.team_embedding(teams).unsqueeze(1).expand(-1, steps, -1)
            return self.recurrent(torch.cat((numeric, hero, team), dim=-1), hidden)

        def forward(
            self, numeric: Any, heroes: Any, teams: Any, hidden: Any | None = None
        ) -> tuple[Any, Any, Any]:
            encoded, next_hidden = self.encode(numeric, heroes, teams, hidden)
            return self.head(encoded), self.value_head(encoded).squeeze(-1), next_hidden

        def team_sequence(
            self, numeric: Any, heroes: Any, teams: Any
        ) -> tuple[Any, Any]:
            batch, steps, _, features = numeric.shape
            flattened = numeric.permute(0, 2, 1, 3).reshape(batch * 5, steps, features)
            encoded, _ = self.encode(flattened, heroes.reshape(-1), teams.reshape(-1))
            team_encoded = (
                encoded.reshape(batch, 5, steps, 96)
                .permute(0, 2, 1, 3)
                .reshape(batch, steps, 480)
            )
            return self.composition_head(team_encoded), self.team_value_head(team_encoded).squeeze(-1)

    return TeamPolicy()


def load_team_initialization(checkpoint: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_team_policy(int(bundle["maximum_hero_id"]))
    missing, unexpected = model.load_state_dict(bundle["state_dict"], strict=False)
    allowed = ("value_head.", "composition_head.", "team_value_head.", "composition_prior")
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.recurrent.flatten_parameters()
    return model.to(device), bundle


def composition_prior(frame: Any, match_ids: list[int]) -> Any:
    """Laplace-smoothed historical frequency of five-hero team plans."""

    import numpy as np

    selected = frame[frame["match_id"].isin(match_ids)]
    counts = np.ones(len(COMPOSITIONS), dtype=np.float64)
    for _, rows in selected.groupby(["match_id", "team", "state_minute"], sort=False):
        if len(rows) != 5:
            continue
        composition = tuple(int((rows["label"] == action).sum()) for action in ACTIONS)
        counts[COMPOSITIONS.index(composition)] += 1
    return (counts / counts.sum()).astype(np.float32)


def initialize_composition_prior(model: Any, prior: Any) -> None:
    import torch

    output = model.composition_head[-1]
    torch.nn.init.zeros_(output.weight)
    with torch.no_grad():
        output.bias.copy_(torch.log(torch.as_tensor(prior, device=output.bias.device)))
        model.composition_prior.copy_(torch.as_tensor(prior, device=output.bias.device))


def composition_distribution(model: Any, logits: Any) -> Any:
    """Keep 25% historical support while PPO learns the other 75%."""

    import torch

    learned = torch.softmax(logits, dim=-1)
    probabilities = 0.75 * learned + 0.25 * model.composition_prior
    return torch.distributions.Categorical(probs=probabilities)


def assign_compositions(hero_logits: Any, composition_indices: Any) -> Any:
    """Choose the maximum-score hero-to-role assignment for each plan."""

    import torch

    joint = torch.as_tensor(JOINT_ACTIONS, dtype=torch.long, device=hero_logits.device)
    joint_composition = torch.as_tensor(
        JOINT_COMPOSITION, dtype=torch.long, device=hero_logits.device
    )
    scores = torch.zeros(hero_logits.shape[0], len(JOINT_ACTIONS), device=hero_logits.device)
    for slot in range(5):
        scores += hero_logits[:, slot][:, joint[:, slot]]
    scores = scores.masked_fill(
        joint_composition.unsqueeze(0) != composition_indices.unsqueeze(1), float("-inf")
    )
    return joint[scores.argmax(dim=1)]


def team_step(
    model: Any,
    numeric: Any,
    heroes: Any,
    teams: Any,
    hidden: Any,
    *,
    deterministic: bool,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    batch = numeric.shape[0]
    encoded, next_hidden = model.encode(
        numeric.reshape(batch * 5, 1, 13), heroes.reshape(-1), teams.reshape(-1), hidden
    )
    hero_encoded = encoded[:, 0].reshape(batch, 5, 96)
    hero_logits = model.head(hero_encoded)
    team_encoded = hero_encoded.reshape(batch, 480)
    composition_logits = model.composition_head(team_encoded)
    distribution = composition_distribution(model, composition_logits)
    composition = (
        composition_logits.argmax(dim=-1) if deterministic else distribution.sample()
    )
    actions = assign_compositions(hero_logits, composition)
    value = model.team_value_head(team_encoded).squeeze(-1)
    return actions, composition, distribution.log_prob(composition), value, next_hidden


def independent_step(
    model: Any, numeric: Any, heroes: Any, teams: Any, hidden: Any, *, deterministic: bool
) -> tuple[Any, Any]:
    import torch

    batch = numeric.shape[0]
    logits, _, next_hidden = model(
        numeric.reshape(batch * 5, 1, 13), heroes.reshape(-1), teams.reshape(-1), hidden
    )
    distribution = torch.distributions.Categorical(logits=logits[:, 0])
    actions = logits[:, 0].argmax(-1) if deterministic else distribution.sample()
    return actions.reshape(batch, 5), next_hidden


def collect_team_rollout(
    model: Any,
    opponent: Any | str,
    calibration: Any,
    means: Any,
    scales: Any,
    *,
    environments: int,
    seed: int,
    max_minutes: int,
    learner_side: int,
    device: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch

    env = DotaSelfPlayEnv(
        calibration, environments=environments, seed=seed, max_minutes=max_minutes
    )
    observations, heroes, teams = env.observations()
    own_hidden = torch.zeros(2, environments * 5, 96, device=device)
    enemy_hidden = torch.zeros_like(own_hidden)
    model.eval()
    storage = {name: [] for name in (
        "numeric", "heroes", "teams", "composition", "log_prob", "value", "reward", "done", "valid"
    )}
    action_counts = np.zeros(4, dtype=np.int64)
    for _ in range(max_minutes):
        active = ~env.done
        enemy_side = 1 - learner_side
        own_np = (observations[:, learner_side] - means) / scales
        enemy_np = (observations[:, enemy_side] - means) / scales
        with torch.no_grad():
            own_actions, composition, log_prob, value, own_hidden = team_step(
                model,
                torch.as_tensor(own_np, dtype=torch.float32, device=device),
                torch.as_tensor(heroes[:, learner_side], device=device),
                torch.as_tensor(teams[:, learner_side], device=device),
                own_hidden,
                deterministic=False,
            )
            if isinstance(opponent, str):
                enemy_actions = scripted_actions(observations[:, enemy_side], opponent)
            else:
                enemy_actions_tensor, _, _, _, enemy_hidden = team_step(
                    opponent,
                    torch.as_tensor(enemy_np, dtype=torch.float32, device=device),
                    torch.as_tensor(heroes[:, enemy_side], device=device),
                    torch.as_tensor(teams[:, enemy_side], device=device),
                    enemy_hidden,
                    deterministic=False,
                )
                enemy_actions = enemy_actions_tensor.cpu().numpy()
        own_actions_np = own_actions.cpu().numpy()
        actions = np.empty((environments, 2, 5), dtype=np.int64)
        actions[:, learner_side] = own_actions_np
        actions[:, enemy_side] = enemy_actions
        (next_observations, heroes, teams), rewards, done, _ = env.step(actions)
        newly_done = done & active
        winners = env.winners()
        terminal = np.where(winners == learner_side, 3.0, np.where(winners < 0, 0.0, -3.0))
        rewards[:, learner_side] += newly_done * terminal
        storage["numeric"].append(torch.from_numpy(own_np.copy()))
        storage["heroes"].append(torch.from_numpy(heroes[:, learner_side].copy()))
        storage["teams"].append(torch.from_numpy(teams[:, learner_side].copy()))
        storage["composition"].append(composition.cpu())
        storage["log_prob"].append(log_prob.cpu())
        storage["value"].append(value.cpu())
        storage["reward"].append(torch.from_numpy(rewards[:, learner_side].copy()))
        storage["done"].append(torch.from_numpy(done.copy()))
        storage["valid"].append(torch.from_numpy(active.copy()))
        for action in range(4):
            action_counts[action] += int(((own_actions_np == action) & active[:, None]).sum())
        observations = next_observations
        if done.all():
            break
    result = {name: torch.stack(values) for name, values in storage.items()}
    result["action_counts"] = action_counts
    return result


def team_ppo_update(
    model: Any,
    optimizer: Any,
    rollout: dict[str, Any],
    *,
    device: Any,
    epochs: int = 4,
    batch_size: int = 32,
) -> dict[str, float]:
    import numpy as np
    import torch

    rewards, values = rollout["reward"].float(), rollout["value"].float()
    done = rollout["done"].float()
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        nonterminal = 1 - done[step]
        delta = rewards[step] + 0.99 * next_value * nonterminal - values[step]
        next_advantage = delta + 0.99 * 0.95 * nonterminal * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    returns = advantages + values
    numeric = rollout["numeric"].permute(1, 0, 2, 3)
    heroes = rollout["heroes"][0]
    teams = rollout["teams"][0]
    composition = rollout["composition"].transpose(0, 1)
    old_log_prob = rollout["log_prob"].transpose(0, 1)
    valid = rollout["valid"].transpose(0, 1).bool()
    advantages = advantages.transpose(0, 1)
    returns = returns.transpose(0, 1)
    active_advantages = advantages[valid]
    advantages = (advantages - active_advantages.mean()) / (active_advantages.std() + 1e-8)
    indices = np.arange(len(numeric))
    metrics = {name: [] for name in ("loss", "policy_loss", "value_loss", "entropy", "kl")}
    model.train()
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            index = indices[start : start + batch_size]
            mask = valid[index].to(device)
            logits, predicted_value = model.team_sequence(
                numeric[index].to(device), heroes[index].to(device), teams[index].to(device)
            )
            distribution = composition_distribution(model, logits)
            new_log_prob = distribution.log_prob(composition[index].to(device))
            old = old_log_prob[index].to(device)
            ratio = torch.exp(new_log_prob - old)
            advantage = advantages[index].to(device)
            policy_loss = -torch.minimum(
                ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage
            )[mask].mean()
            value_loss = torch.nn.functional.smooth_l1_loss(
                predicted_value[mask], returns[index].to(device)[mask]
            )
            entropy = distribution.entropy()[mask].mean()
            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics["loss"].append(float(loss.detach().cpu()))
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.detach().cpu()))
            metrics["kl"].append(float((old - new_log_prob)[mask].mean().detach().cpu()))
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def evaluate_team_matchup(
    model: Any,
    opponent: Any | str | tuple[Any, str],
    calibration: Any,
    means: Any,
    scales: Any,
    *,
    games: int,
    seed: int,
    max_minutes: int,
    device: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    outcomes, action_counts = [], np.zeros(4, dtype=np.int64)
    for side, count in ((0, games // 2), (1, games - games // 2)):
        env = DotaSelfPlayEnv(
            calibration, environments=count, seed=seed + side * 100003, max_minutes=max_minutes
        )
        observations, heroes, teams = env.observations()
        own_hidden = torch.zeros(2, count * 5, 96, device=device)
        enemy_hidden = torch.zeros_like(own_hidden)
        for _ in range(max_minutes):
            enemy_side = 1 - side
            own_np = (observations[:, side] - means) / scales
            enemy_np = (observations[:, enemy_side] - means) / scales
            with torch.no_grad():
                own_actions, _, _, _, own_hidden = team_step(
                    model,
                    torch.as_tensor(own_np, dtype=torch.float32, device=device),
                    torch.as_tensor(heroes[:, side], device=device),
                    torch.as_tensor(teams[:, side], device=device),
                    own_hidden,
                    deterministic=False,
                )
                if isinstance(opponent, str):
                    enemy_actions = scripted_actions(observations[:, enemy_side], opponent)
                elif isinstance(opponent, tuple) and opponent[1] == "independent":
                    enemy_tensor, enemy_hidden = independent_step(
                        opponent[0],
                        torch.as_tensor(enemy_np, dtype=torch.float32, device=device),
                        torch.as_tensor(heroes[:, enemy_side], device=device),
                        torch.as_tensor(teams[:, enemy_side], device=device),
                        enemy_hidden,
                        deterministic=False,
                    )
                    enemy_actions = enemy_tensor.cpu().numpy()
                else:
                    enemy_tensor, _, _, _, enemy_hidden = team_step(
                        opponent,
                        torch.as_tensor(enemy_np, dtype=torch.float32, device=device),
                        torch.as_tensor(heroes[:, enemy_side], device=device),
                        torch.as_tensor(teams[:, enemy_side], device=device),
                        enemy_hidden,
                        deterministic=False,
                    )
                    enemy_actions = enemy_tensor.cpu().numpy()
            own_np_actions = own_actions.cpu().numpy()
            actions = np.empty((count, 2, 5), dtype=np.int64)
            actions[:, side], actions[:, enemy_side] = own_np_actions, enemy_actions
            (observations, heroes, teams), _, done, _ = env.step(actions)
            for action in range(4):
                action_counts[action] += int((own_np_actions == action).sum())
            if done.all():
                break
        winners = env.winners()
        outcomes.extend(1 if value == side else 0.5 if value < 0 else 0 for value in winners)
    wins = sum(value == 1 for value in outcomes)
    decisive = sum(value != 0.5 for value in outcomes)
    total_actions = max(int(action_counts.sum()), 1)
    return {
        "games": games,
        "wins": wins,
        "losses": decisive - wins,
        "draws": games - decisive,
        "score_rate": float(sum(outcomes) / games),
        "decisive_win_rate": float(wins / decisive) if decisive else 0.5,
        "wilson_95": _wilson(wins, decisive),
        "action_distribution": {
            action: float(action_counts[index] / total_actions)
            for index, action in enumerate(ACTIONS)
        },
    }


def train_team_selfplay(
    dataset: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    device: str = "auto",
    iterations: int = 60,
    environments: int = 128,
    max_minutes: int = 45,
    evaluation_games: int = 1000,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    frame = pd.read_csv(dataset)
    splits = split_match_ids(frame["match_id"].astype(int).tolist(), seed)
    calibration = fit_replay_calibration(frame, splits["train"])
    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested but unavailable")
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    model, initial = load_team_initialization(initial_checkpoint, torch_device)
    prior = composition_prior(frame, splits["train"])
    initialize_composition_prior(model, prior)
    initial_state = copy.deepcopy(model.state_dict())
    means, scales = np.asarray(initial["means"], dtype=np.float32), np.asarray(initial["scales"], dtype=np.float32)
    from .train_sequence_policy import build_sequences

    replay_sequences = build_sequences(frame, splits["train"], means, scales)
    labels = frame[frame["match_id"].isin(splits["train"])]["label"]
    counts = labels.value_counts()
    class_weights = np.asarray(
        [(len(labels) / (4 * counts.get(label, 1))) ** 0.5 for label in LABELS], dtype=np.float32
    )
    minimum_share = {label: float(0.25 * counts.get(label, 0) / len(labels)) for label in LABELS}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-5)
    archive = [copy.deepcopy(initial_state)]
    history = []
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        schedule = iteration % 5
        if schedule == 1:
            opponent: Any | str = "balanced"
            opponent_name = "scripted_balanced"
        elif schedule == 2:
            opponent, opponent_name = "aggressive", "scripted_aggressive"
        elif schedule == 3:
            opponent, opponent_name = "farm", "scripted_farm"
        else:
            index = random.randrange(len(archive))
            opponent = copy.deepcopy(model).to(torch_device)
            opponent.load_state_dict(archive[index])
            opponent.recurrent.flatten_parameters()
            opponent.eval()
            opponent_name = f"archive_{index}"
        rollout = collect_team_rollout(
            model,
            opponent,
            calibration,
            means,
            scales,
            environments=environments,
            seed=seed + iteration * 7919,
            max_minutes=max_minutes,
            learner_side=iteration % 2,
            device=torch_device,
        )
        update = team_ppo_update(model, optimizer, rollout, device=torch_device)
        replay_loss = replay_regularization_update(
            model, optimizer, replay_sequences, class_weights, device=torch_device
        )
        action_counts = rollout["action_counts"]
        total = max(int(action_counts.sum()), 1)
        history.append(
            {
                "iteration": iteration,
                "opponent": opponent_name,
                **update,
                "replay_loss": replay_loss,
                "action_distribution": {
                    action: float(action_counts[index] / total)
                    for index, action in enumerate(ACTIONS)
                },
            }
        )
        if iteration % 5 == 0:
            archive.append(copy.deepcopy(model.state_dict()))

    original = copy.deepcopy(model).to(torch_device)
    original.load_state_dict(initial_state)
    original.recurrent.flatten_parameters()
    original.eval()
    symmetry_opponent = copy.deepcopy(model).to(torch_device)
    symmetry_opponent.recurrent.flatten_parameters()
    symmetry_opponent.eval()
    evaluation = {
        "original_imitation": evaluate_team_matchup(
            model, (original, "independent"), calibration, means, scales,
            games=evaluation_games, seed=seed + 1_000_003, max_minutes=max_minutes, device=torch_device
        ),
        "scripted_balanced": evaluate_team_matchup(
            model, "balanced", calibration, means, scales,
            games=evaluation_games, seed=seed + 2_000_003, max_minutes=max_minutes, device=torch_device
        ),
        "scripted_aggressive": evaluate_team_matchup(
            model, "aggressive", calibration, means, scales,
            games=evaluation_games, seed=seed + 3_000_003, max_minutes=max_minutes, device=torch_device
        ),
        "scripted_farm": evaluate_team_matchup(
            model, "farm", calibration, means, scales,
            games=evaluation_games, seed=seed + 4_000_003, max_minutes=max_minutes, device=torch_device
        ),
        "symmetry_control": evaluate_team_matchup(
            model, symmetry_opponent, calibration, means, scales,
            games=evaluation_games, seed=seed + 5_000_003, max_minutes=max_minutes, device=torch_device
        ),
    }
    replay_imitation = {
        "candidate": evaluate_replay_imitation(model, frame, splits["test"], means, scales, device=torch_device),
        "original": evaluate_replay_imitation(original, frame, splits["test"], means, scales, device=torch_device),
    }
    replay_imitation["macro_f1_delta"] = replay_imitation["candidate"]["macro_f1"] - replay_imitation["original"]["macro_f1"]
    performance = all(
        evaluation[name]["wilson_95"][0] > 0.5
        for name in ("original_imitation", "scripted_balanced", "scripted_aggressive", "scripted_farm")
    )
    symmetry = evaluation["symmetry_control"]["wilson_95"][0] <= 0.5 <= evaluation["symmetry_control"]["wilson_95"][1]
    diversity = all(
        all(evaluation[name]["action_distribution"][action] >= minimum_share[action] for action in ACTIONS)
        for name in ("original_imitation", "scripted_balanced", "scripted_aggressive", "scripted_farm")
    )
    retention = replay_imitation["macro_f1_delta"] >= -0.01
    passed = performance and symmetry and diversity and retention
    result = {
        "dataset": str(dataset),
        "initial_checkpoint": str(initial_checkpoint),
        "seed": seed,
        "device": str(torch_device),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "iterations": iterations,
        "environments": environments,
        "max_minutes": max_minutes,
        "train_seconds": time.perf_counter() - started,
        "composition_count": len(COMPOSITIONS),
        "composition_prior": prior.tolist(),
        "calibration": calibration.to_dict(),
        "calibration_audit": calibration_audit(frame, splits["test"], calibration),
        "match_ids": splits,
        "archive_size": len(archive),
        "minimum_action_share": minimum_share,
        "history": history,
        "evaluation": evaluation,
        "replay_imitation": replay_imitation,
        "promotion_gate": {
            "rule": "Wilson lower 95% > 0.5 vs four opponents; symmetry; historical diversity; replay delta >= -0.01",
            "performance_passed": performance,
            "symmetry_passed": symmetry,
            "noncollapse_passed": diversity,
            "replay_retention_passed": retention,
            "passed": passed,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "means": means,
            "scales": scales,
            "features": initial["features"],
            "labels": initial["labels"],
            "maximum_hero_id": initial["maximum_hero_id"],
            "compositions": COMPOSITIONS,
            "calibration": calibration.to_dict(),
        },
        output_dir / "team-selfplay-policy-v1.pt",
    )
    (output_dir / "team-selfplay-policy-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v3.csv"))
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("artifacts/sequence-models/sequence-policy-v1.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/team-selfplay-models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--environments", type=int, default=128)
    parser.add_argument("--max-minutes", type=int, default=45)
    parser.add_argument("--evaluation-games", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_team_selfplay(
        args.dataset, args.initial_checkpoint, args.output_dir,
        seed=args.seed, device=args.device, iterations=args.iterations,
        environments=args.environments, max_minutes=args.max_minutes,
        evaluation_games=args.evaluation_games,
    )
    print(json.dumps({
        "promotion": result["promotion_gate"]["passed"],
        "vs_imitation": result["evaluation"]["original_imitation"]["decisive_win_rate"],
        "vs_balanced": result["evaluation"]["scripted_balanced"]["decisive_win_rate"],
        "vs_aggressive": result["evaluation"]["scripted_aggressive"]["decisive_win_rate"],
        "replay_delta": result["replay_imitation"]["macro_f1_delta"],
        "seconds": result["train_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
