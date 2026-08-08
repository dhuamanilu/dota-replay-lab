"""Recurrent PPO self-play in the replay-calibrated symmetric 5v5 surrogate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from .selfplay_env import ACTIONS, DotaSelfPlayEnv, fit_replay_calibration
from .train_policy import split_match_ids


def make_selfplay_policy(maximum_hero_id: int) -> Any:
    """Match the imitation GRU exactly and add an independent value head."""

    import torch
    from torch import nn

    class SelfPlayPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hero_embedding = nn.Embedding(maximum_hero_id + 1, 16)
            self.team_embedding = nn.Embedding(2, 2)
            self.recurrent = nn.GRU(
                31, 96, num_layers=2, dropout=0.0, batch_first=True
            )
            self.head = nn.Sequential(
                nn.LayerNorm(96),
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(64, len(ACTIONS)),
            )
            self.value_head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1))

        def forward(
            self, numeric: Any, heroes: Any, teams: Any, hidden: Any | None = None
        ) -> tuple[Any, Any, Any]:
            steps = numeric.shape[1]
            hero = self.hero_embedding(heroes).unsqueeze(1).expand(-1, steps, -1)
            team = self.team_embedding(teams).unsqueeze(1).expand(-1, steps, -1)
            encoded, next_hidden = self.recurrent(
                torch.cat((numeric, hero, team), dim=-1), hidden
            )
            return self.head(encoded), self.value_head(encoded).squeeze(-1), next_hidden

    return SelfPlayPolicy()


def load_imitation_initialization(checkpoint: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    """Load only a trusted locally produced recurrent checkpoint."""

    import torch

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_selfplay_policy(int(bundle["maximum_hero_id"]))
    missing, unexpected = model.load_state_dict(bundle["state_dict"], strict=False)
    if unexpected or set(missing) != {"value_head.0.weight", "value_head.0.bias", "value_head.1.weight", "value_head.1.bias"}:
        raise ValueError(f"Checkpoint architecture mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device), bundle


def scripted_actions(observations: Any, style: str) -> Any:
    """Deterministic team actions used as interpretable opponents."""

    import numpy as np

    batch = observations.shape[0]
    minute = observations[:, 0, 0]
    advantage = observations[:, 0, 7]
    result = np.zeros((batch, 5), dtype=np.int64)
    if style == "farm":
        result[:] = np.asarray([0, 0, 0, 0, 3])
    elif style == "aggressive":
        result[:] = np.asarray([1, 1, 1, 2, 0])
    elif style == "balanced":
        for index in range(batch):
            if minute[index] < 10:
                result[index] = [0, 0, 0, 0, 3]
            elif advantage[index] < -2500:
                result[index] = [1, 1, 1, 0, 3]
            elif minute[index] >= 30:
                result[index] = [1, 1, 2, 2, 0]
            else:
                result[index] = [0, 0, 1, 1, 2]
    else:
        raise ValueError(f"Unknown scripted style: {style}")
    return result


def _normalize(values: Any, means: Any, scales: Any) -> Any:
    return (values - means) / scales


def _policy_actions(
    model: Any,
    numeric: Any,
    heroes: Any,
    teams: Any,
    hidden: Any,
    *,
    deterministic: bool,
) -> tuple[Any, Any, Any, Any]:
    import torch

    logits, values, next_hidden = model(
        numeric.unsqueeze(1), heroes, teams, hidden
    )
    distribution = torch.distributions.Categorical(logits=logits[:, 0])
    actions = logits[:, 0].argmax(dim=-1) if deterministic else distribution.sample()
    return actions, distribution.log_prob(actions), values[:, 0], next_hidden


def _snapshot_model(template: Any, state: dict[str, Any], device: Any) -> Any:
    model = copy.deepcopy(template).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def collect_rollout(
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
    deterministic_opponent: bool = False,
) -> dict[str, Any]:
    """Collect complete episodes and retain causal recurrent sequences."""

    import numpy as np
    import torch

    env = DotaSelfPlayEnv(
        calibration,
        environments=environments,
        seed=seed,
        max_minutes=max_minutes,
        stochastic=True,
    )
    observations, heroes, teams = env.observations()
    model.eval()
    hidden = torch.zeros(2, environments * 5, 96, device=device)
    opponent_hidden = torch.zeros_like(hidden)
    storage: dict[str, list[Any]] = {
        name: []
        for name in ("numeric", "heroes", "teams", "actions", "log_probs", "values", "rewards", "dones", "valid")
    }
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    for _ in range(max_minutes):
        active = ~env.done
        own_numeric_np = _normalize(observations[:, learner_side], means, scales)
        enemy_side = 1 - learner_side
        enemy_numeric_np = _normalize(observations[:, enemy_side], means, scales)
        own_numeric = torch.as_tensor(
            own_numeric_np.reshape(-1, own_numeric_np.shape[-1]), dtype=torch.float32, device=device
        )
        own_heroes = torch.as_tensor(heroes[:, learner_side].reshape(-1), device=device)
        own_teams = torch.as_tensor(teams[:, learner_side].reshape(-1), device=device)
        with torch.no_grad():
            own_actions, log_probs, values, hidden = _policy_actions(
                model, own_numeric, own_heroes, own_teams, hidden, deterministic=False
            )
            if isinstance(opponent, str):
                enemy_actions_np = scripted_actions(observations[:, enemy_side], opponent)
            else:
                enemy_numeric = torch.as_tensor(
                    enemy_numeric_np.reshape(-1, enemy_numeric_np.shape[-1]),
                    dtype=torch.float32,
                    device=device,
                )
                enemy_heroes = torch.as_tensor(heroes[:, enemy_side].reshape(-1), device=device)
                enemy_teams = torch.as_tensor(teams[:, enemy_side].reshape(-1), device=device)
                enemy_actions, _, _, opponent_hidden = _policy_actions(
                    opponent,
                    enemy_numeric,
                    enemy_heroes,
                    enemy_teams,
                    opponent_hidden,
                    deterministic=deterministic_opponent,
                )
                enemy_actions_np = enemy_actions.reshape(environments, 5).cpu().numpy()
        own_actions_np = own_actions.reshape(environments, 5).cpu().numpy()
        all_actions = np.empty((environments, 2, 5), dtype=np.int64)
        all_actions[:, learner_side] = own_actions_np
        all_actions[:, enemy_side] = enemy_actions_np
        (next_observations, next_heroes, next_teams), rewards, dones, _ = env.step(all_actions)
        newly_done = dones & active
        if newly_done.any():
            winners = env.winners()
            terminal = np.where(winners == learner_side, 3.0, np.where(winners < 0, 0.0, -3.0))
            rewards[:, learner_side] += newly_done * terminal
        storage["numeric"].append(torch.from_numpy(own_numeric_np.copy()))
        storage["heroes"].append(torch.from_numpy(heroes[:, learner_side].copy()))
        storage["teams"].append(torch.from_numpy(teams[:, learner_side].copy()))
        storage["actions"].append(torch.from_numpy(own_actions_np.copy()))
        storage["log_probs"].append(log_probs.reshape(environments, 5).cpu())
        storage["values"].append(values.reshape(environments, 5).cpu())
        storage["rewards"].append(
            torch.from_numpy(np.broadcast_to(rewards[:, learner_side, None], (environments, 5)).copy())
        )
        storage["dones"].append(
            torch.from_numpy(np.broadcast_to(dones[:, None], (environments, 5)).copy())
        )
        storage["valid"].append(
            torch.from_numpy(np.broadcast_to(active[:, None], (environments, 5)).copy())
        )
        for action in range(len(ACTIONS)):
            action_counts[action] += int(((own_actions_np == action) & active[:, None]).sum())
        observations = next_observations
        heroes = next_heroes
        teams = next_teams
        if dones.all():
            break
    result = {name: torch.stack(values) for name, values in storage.items()}
    result["action_counts"] = action_counts
    result["wins"] = env.winners()
    return result


def _gae(rollout: dict[str, Any], gamma: float, gae_lambda: float) -> tuple[Any, Any]:
    import torch

    rewards = rollout["rewards"].float()
    values = rollout["values"].float()
    dones = rollout["dones"].float()
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    return advantages, advantages + values


def ppo_update(
    model: Any,
    optimizer: Any,
    rollout: dict[str, Any],
    *,
    device: Any,
    epochs: int,
    sequence_batch: int,
    clip_ratio: float,
    entropy_weight: float,
    value_weight: float,
    gamma: float,
    gae_lambda: float,
) -> dict[str, float]:
    import numpy as np
    import torch

    advantages, returns = _gae(rollout, gamma, gae_lambda)
    # T,N,5,F -> N*5,T,F so the GRU is optimized on complete hero histories.
    numeric = rollout["numeric"].permute(1, 2, 0, 3).reshape(-1, len(rollout["numeric"]), 13)
    heroes = rollout["heroes"][0].reshape(-1)
    teams = rollout["teams"][0].reshape(-1)
    actions = rollout["actions"].permute(1, 2, 0).reshape(-1, len(numeric[0]))
    old_log_probs = rollout["log_probs"].permute(1, 2, 0).reshape_as(actions)
    valid = rollout["valid"].permute(1, 2, 0).reshape_as(actions).bool()
    advantages = advantages.permute(1, 2, 0).reshape_as(actions)
    returns = returns.permute(1, 2, 0).reshape_as(actions)
    valid_advantages = advantages[valid]
    advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
    indices = np.arange(len(numeric))
    losses, policy_losses, value_losses, entropies, kls = [], [], [], [], []
    # The self-play copy has dropout set to zero, so train/eval logits are equal
    # while cuDNN can still backpropagate through the recurrent layers.
    model.train()
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), sequence_batch):
            index = indices[start : start + sequence_batch]
            mask = valid[index].to(device)
            logits, predicted_values, _ = model(
                numeric[index].to(device), heroes[index].to(device), teams[index].to(device)
            )
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[index].to(device))
            old = old_log_probs[index].to(device)
            ratio = torch.exp(new_log_probs - old)
            advantage = advantages[index].to(device)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
            policy_loss = -torch.minimum(unclipped, clipped)[mask].mean()
            value_loss = torch.nn.functional.smooth_l1_loss(
                predicted_values[mask], returns[index].to(device)[mask]
            )
            entropy = distribution.entropy()[mask].mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = (old - new_log_probs)[mask].mean()
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            kls.append(float(approximate_kl.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approximate_kl": float(np.mean(kls)),
    }


def replay_regularization_update(
    model: Any,
    optimizer: Any,
    sequences: list[dict[str, Any]],
    class_weights: Any,
    *,
    device: Any,
    updates: int = 12,
    batch_size: int = 48,
) -> float:
    """Interleave historical supervision to prevent simulator forgetting."""

    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence

    losses = []
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        ignore_index=-100,
    )
    for _ in range(updates):
        indices = np.random.choice(len(sequences), size=min(batch_size, len(sequences)), replace=False)
        rows = [sequences[int(index)] for index in indices]
        numeric = pad_sequence(
            [torch.from_numpy(row["numeric"]) for row in rows], batch_first=True
        ).to(device)
        labels = pad_sequence(
            [torch.from_numpy(row["labels"]) for row in rows],
            batch_first=True,
            padding_value=-100,
        ).to(device)
        heroes = torch.tensor([row["hero_id"] for row in rows], device=device)
        teams = torch.tensor([row["team_id"] for row in rows], device=device)
        logits, _, _ = model(numeric, heroes, teams)
        loss = loss_function(logits.reshape(-1, len(ACTIONS)), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _wilson(wins: int, games: int) -> list[float]:
    if games == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = wins / games
    denominator = 1 + z * z / games
    center = (proportion + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / games + z * z / (4 * games * games)) / denominator
    return [center - margin, center + margin]


def calibration_audit(frame: Any, match_ids: list[int], calibration: Any) -> dict[str, Any]:
    """Audit empirical anchors while exposing causal-cost assumptions."""

    selected = frame[frame["match_id"].isin(match_ids)]
    farm = selected[selected["label"] == "farm"]
    result: dict[str, Any] = {
        "resource_anchor": {},
        "opportunity_cost_multipliers": {
            "gold": [1.0, 0.58, 0.68, 0.40],
            "experience": [1.0, 0.65, 0.75, 0.50],
            "last_hits": [1.0, 0.45, 0.65, 0.20],
        },
    }
    for column, expected in (
        ("gold_change", calibration.gold_mean[0]),
        ("experience_change", calibration.experience_mean[0]),
        ("last_hit_change", calibration.last_hit_mean[0]),
    ):
        observed = float(farm[column].clip(lower=0).mean())
        result["resource_anchor"][column] = {
            "calibrated": expected,
            "held_out": observed,
            "relative_error": abs(expected - observed) / max(abs(observed), 1.0),
        }
    fight_rate = float(
        selected.loc[selected["label"] == "fight", "kills_last_minute"].clip(0, 2).mean()
    )
    result["fight_kill_rate"] = {
        "calibrated": calibration.kill_rate[1],
        "held_out": fight_rate,
        "relative_error": abs(calibration.kill_rate[1] - fight_rate) / max(fight_rate, 1.0),
    }
    push_rate = float((selected["label"] == "push").mean())
    result["push_event_rate"] = {
        "calibrated": calibration.push_rate,
        "held_out": push_rate,
        "relative_error": abs(calibration.push_rate - push_rate) / max(push_rate, 1e-3),
    }
    return result


def evaluate_matchup(
    model: Any,
    opponent: Any | str,
    calibration: Any,
    means: Any,
    scales: Any,
    *,
    games: int,
    seed: int,
    max_minutes: int,
    device: Any,
    candidate_deterministic: bool = False,
    opponent_deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate deterministic policies on both sides with independent seeds."""

    import numpy as np
    import torch

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    all_outcomes = []
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    half = games // 2
    for side, count in ((0, half), (1, games - half)):
        env = DotaSelfPlayEnv(
            calibration, environments=count, seed=seed + side * 100003, max_minutes=max_minutes
        )
        observations, heroes, teams = env.observations()
        own_hidden = torch.zeros(2, count * 5, 96, device=device)
        enemy_hidden = torch.zeros_like(own_hidden)
        for _ in range(max_minutes):
            enemy_side = 1 - side
            own_np = _normalize(observations[:, side], means, scales)
            own = torch.as_tensor(own_np.reshape(-1, 13), dtype=torch.float32, device=device)
            with torch.no_grad():
                own_actions, _, _, own_hidden = _policy_actions(
                    model,
                    own,
                    torch.as_tensor(heroes[:, side].reshape(-1), device=device),
                    torch.as_tensor(teams[:, side].reshape(-1), device=device),
                    own_hidden,
                    deterministic=candidate_deterministic,
                )
                if isinstance(opponent, str):
                    enemy_np = scripted_actions(observations[:, enemy_side], opponent)
                else:
                    opponent_numeric = _normalize(observations[:, enemy_side], means, scales)
                    enemy_actions, _, _, enemy_hidden = _policy_actions(
                        opponent,
                        torch.as_tensor(
                            opponent_numeric.reshape(-1, 13), dtype=torch.float32, device=device
                        ),
                        torch.as_tensor(heroes[:, enemy_side].reshape(-1), device=device),
                        torch.as_tensor(teams[:, enemy_side].reshape(-1), device=device),
                        enemy_hidden,
                        deterministic=opponent_deterministic,
                    )
                    enemy_np = enemy_actions.reshape(count, 5).cpu().numpy()
            own_np_actions = own_actions.reshape(count, 5).cpu().numpy()
            actions = np.empty((count, 2, 5), dtype=np.int64)
            actions[:, side] = own_np_actions
            actions[:, enemy_side] = enemy_np
            (observations, heroes, teams), _, done, _ = env.step(actions)
            for action in range(len(ACTIONS)):
                action_counts[action] += int((own_np_actions == action).sum())
            if done.all():
                break
        winners = env.winners()
        all_outcomes.extend(
            1 if winner == side else 0.5 if winner < 0 else 0 for winner in winners
        )
    decisive_wins = sum(value == 1 for value in all_outcomes)
    decisive_games = sum(value != 0.5 for value in all_outcomes)
    total_actions = max(int(action_counts.sum()), 1)
    return {
        "games": games,
        "wins": decisive_wins,
        "losses": decisive_games - decisive_wins,
        "draws": games - decisive_games,
        "score_rate": float(sum(all_outcomes) / games),
        "decisive_win_rate": float(decisive_wins / decisive_games) if decisive_games else 0.5,
        "wilson_95": _wilson(decisive_wins, decisive_games),
        "action_distribution": {
            action: float(action_counts[index] / total_actions)
            for index, action in enumerate(ACTIONS)
        },
    }


def evaluate_replay_imitation(
    model: Any,
    frame: Any,
    match_ids: list[int],
    means: Any,
    scales: Any,
    *,
    device: Any,
) -> dict[str, Any]:
    """Guard against simulator gains that destroy held-out replay behavior."""

    import numpy as np
    import torch

    from .train_policy import LABELS, _metrics
    from .train_sequence_policy import build_sequences

    sequences = build_sequences(frame, match_ids, means, scales)
    truth, predicted = [], []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            numeric = torch.as_tensor(
                sequence["numeric"][None], dtype=torch.float32, device=device
            )
            hero = torch.tensor([sequence["hero_id"]], device=device)
            team = torch.tensor([sequence["team_id"]], device=device)
            logits, _, _ = model(numeric, hero, team)
            truth.extend(sequence["labels"].tolist())
            predicted.extend(logits[0].argmax(dim=-1).cpu().tolist())
    return _metrics(
        np.asarray([LABELS[index] for index in truth]),
        np.asarray([LABELS[index] for index in predicted]),
    )


def train_selfplay(
    dataset: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    device: str = "auto",
    iterations: int = 30,
    environments: int = 64,
    max_minutes: int = 45,
    evaluation_games: int = 400,
) -> dict[str, Any]:
    """Train recurrent PPO against a league of frozen historical snapshots."""

    import numpy as np
    import pandas as pd
    import torch

    frame = pd.read_csv(dataset)
    splits = split_match_ids(frame["match_id"].astype(int).tolist(), seed)
    calibration = fit_replay_calibration(frame, splits["train"])
    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but unavailable")
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    model, initial = load_imitation_initialization(initial_checkpoint, torch_device)
    means = np.asarray(initial["means"], dtype=np.float32)
    scales = np.asarray(initial["scales"], dtype=np.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-5)
    initial_state = copy.deepcopy(model.state_dict())
    archive = [copy.deepcopy(initial_state)]
    from .train_policy import LABELS
    from .train_sequence_policy import build_sequences

    replay_sequences = build_sequences(frame, splits["train"], means, scales)
    train_labels = frame[frame["match_id"].isin(splits["train"])]["label"]
    label_counts = train_labels.value_counts()
    class_weights = np.asarray(
        [
            (len(train_labels) / (len(LABELS) * label_counts.get(label, 1))) ** 0.5
            for label in LABELS
        ],
        dtype=np.float32,
    )
    minimum_action_share = {
        label: float(0.25 * label_counts.get(label, 0) / len(train_labels))
        for label in LABELS
    }
    history = []
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        schedule = iteration % 5
        if schedule == 1:
            opponent: Any | str = "balanced"
            opponent_name = "scripted_balanced"
        elif schedule == 2:
            opponent = "aggressive"
            opponent_name = "scripted_aggressive"
        elif schedule == 3:
            opponent = "farm"
            opponent_name = "scripted_farm"
        else:
            archive_index = random.randrange(len(archive))
            opponent = _snapshot_model(model, archive[archive_index], torch_device)
            opponent_name = f"archive_{archive_index}"
        rollout = collect_rollout(
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
        update = ppo_update(
            model,
            optimizer,
            rollout,
            device=torch_device,
            epochs=4,
            sequence_batch=64,
            clip_ratio=0.2,
            entropy_weight=0.03,
            value_weight=0.5,
            gamma=0.99,
            gae_lambda=0.95,
        )
        replay_loss = replay_regularization_update(
            model,
            optimizer,
            replay_sequences,
            class_weights,
            device=torch_device,
        )
        counts = rollout["action_counts"]
        total = max(int(counts.sum()), 1)
        history.append(
            {
                "iteration": iteration,
                "opponent": opponent_name,
                **update,
                "replay_loss": replay_loss,
                "action_distribution": {
                    action: float(counts[index] / total)
                    for index, action in enumerate(ACTIONS)
                },
            }
        )
        if iteration % 5 == 0:
            archive.append(copy.deepcopy(model.state_dict()))

    original_model = _snapshot_model(model, initial_state, torch_device)
    replay_imitation = {
        "candidate": evaluate_replay_imitation(
            model, frame, splits["test"], means, scales, device=torch_device
        ),
        "original": evaluate_replay_imitation(
            original_model, frame, splits["test"], means, scales, device=torch_device
        ),
    }
    replay_imitation["macro_f1_delta"] = (
        replay_imitation["candidate"]["macro_f1"]
        - replay_imitation["original"]["macro_f1"]
    )
    evaluation = {
        "original_imitation": evaluate_matchup(
            model,
            original_model,
            calibration,
            means,
            scales,
            games=evaluation_games,
            seed=seed + 1_000_003,
            max_minutes=max_minutes,
            device=torch_device,
        ),
        "scripted_balanced": evaluate_matchup(
            model,
            "balanced",
            calibration,
            means,
            scales,
            games=evaluation_games,
            seed=seed + 2_000_003,
            max_minutes=max_minutes,
            device=torch_device,
        ),
        "scripted_aggressive": evaluate_matchup(
            model,
            "aggressive",
            calibration,
            means,
            scales,
            games=evaluation_games,
            seed=seed + 3_000_003,
            max_minutes=max_minutes,
            device=torch_device,
        ),
        "scripted_farm": evaluate_matchup(
            model,
            "farm",
            calibration,
            means,
            scales,
            games=evaluation_games,
            seed=seed + 4_000_003,
            max_minutes=max_minutes,
            device=torch_device,
        ),
        "symmetry_control": evaluate_matchup(
            original_model,
            original_model,
            calibration,
            means,
            scales,
            games=evaluation_games,
            seed=seed + 5_000_003,
            max_minutes=max_minutes,
            device=torch_device,
        ),
    }
    performance_passed = all(
        evaluation[name]["wilson_95"][0] > 0.5
        for name in (
            "original_imitation",
            "scripted_balanced",
            "scripted_aggressive",
            "scripted_farm",
        )
    )
    symmetry_interval = evaluation["symmetry_control"]["wilson_95"]
    symmetry_passed = symmetry_interval[0] <= 0.5 <= symmetry_interval[1]
    noncollapse_passed = all(
        all(
            evaluation[name]["action_distribution"][action]
            >= minimum_action_share[action]
            for action in ACTIONS
        )
        for name in (
            "original_imitation",
            "scripted_balanced",
            "scripted_aggressive",
            "scripted_farm",
        )
    )
    replay_retention_passed = replay_imitation["macro_f1_delta"] >= -0.01
    promotion_passed = (
        performance_passed
        and symmetry_passed
        and noncollapse_passed
        and replay_retention_passed
    )
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
        "calibration": calibration.to_dict(),
        "calibration_audit": calibration_audit(frame, splits["test"], calibration),
        "match_ids": splits,
        "archive_size": len(archive),
        "minimum_action_share": minimum_action_share,
        "history": history,
        "evaluation": evaluation,
        "replay_imitation": replay_imitation,
        "promotion_gate": {
            "rule": "Stochastic Wilson lower 95% > 0.5 vs imitation/balanced/aggressive/farm; symmetry includes 0.5; each action >= 25% of historical share; replay macro-F1 delta >= -0.01",
            "performance_passed": performance_passed,
            "symmetry_passed": symmetry_passed,
            "noncollapse_passed": noncollapse_passed,
            "replay_retention_passed": replay_retention_passed,
            "passed": promotion_passed,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "initial_state_dict": initial_state,
            "means": means,
            "scales": scales,
            "features": initial["features"],
            "labels": initial["labels"],
            "maximum_hero_id": initial["maximum_hero_id"],
            "calibration": calibration.to_dict(),
            "training": {
                "seed": seed,
                "iterations": iterations,
                "environments": environments,
                "max_minutes": max_minutes,
            },
        },
        output_dir / "selfplay-policy-v1.pt",
    )
    (output_dir / "selfplay-policy-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v3.csv")
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("artifacts/sequence-models/sequence-policy-v1.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/selfplay-models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--max-minutes", type=int, default=45)
    parser.add_argument("--evaluation-games", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_selfplay(
        args.dataset,
        args.initial_checkpoint,
        args.output_dir,
        seed=args.seed,
        device=args.device,
        iterations=args.iterations,
        environments=args.environments,
        max_minutes=args.max_minutes,
        evaluation_games=args.evaluation_games,
    )
    original = result["evaluation"]["original_imitation"]
    balanced = result["evaluation"]["scripted_balanced"]
    print(
        f"vs imitation {original['decisive_win_rate']:.3f} "
        f"CI={original['wilson_95']}; vs balanced {balanced['decisive_win_rate']:.3f} "
        f"CI={balanced['wilson_95']}"
    )
    print(
        f"Promotion gate: {result['promotion_gate']['passed']}; "
        f"device {result['device']}; seconds {result['train_seconds']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
