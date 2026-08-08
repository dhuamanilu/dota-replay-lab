"""Train a causal multi-label policy on second-level professional replay actions."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

from .replay_behavior import ACTION_LABELS, NUMERIC_FEATURES, build_behavior_frame, load_trajectory_corpus
from .train_policy import split_match_ids


def _action_metrics(labels: Any, probabilities: Any, thresholds: Any) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    predicted = probabilities >= thresholds
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, average=None, zero_division=0
    )
    result = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "average_precision": float(average_precision_score(labels[:, index], probabilities[:, index])),
            "support": int(support[index]),
            "threshold": float(thresholds[index]),
        }
        for index, label in enumerate(ACTION_LABELS)
    }
    result["macro_f1"] = float(np.mean(f1))
    result["micro_f1"] = float(
        precision_recall_fscore_support(
            labels, predicted, average="micro", zero_division=0
        )[2]
    )
    return result


def _best_thresholds(labels: Any, probabilities: Any) -> Any:
    import numpy as np
    from sklearn.metrics import f1_score

    candidates = np.arange(0.1, 0.91, 0.05)
    return np.asarray(
        [
            max(
                candidates,
                key=lambda value: f1_score(
                    labels[:, index], probabilities[:, index] >= value, zero_division=0
                ),
            )
            for index in range(len(ACTION_LABELS))
        ],
        dtype=np.float32,
    )


def _grouped_bootstrap(deltas: list[float], seed: int, samples: int = 5000) -> dict[str, Any]:
    """Bootstrap the mean policy improvement with whole matches as units."""

    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "matches": int(len(values)),
        "mean_delta": float(values.mean()),
        "confidence_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_positive": float((means > 0).mean()),
        "samples": samples,
    }


def train_replay_behavior(
    trajectory_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    device: str = "auto",
    epochs: int = 20,
    patience: int = 4,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Fit an MLP with match-level train/validation/test isolation."""

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    raw = load_trajectory_corpus(trajectory_dir.glob("*.seconds.csv"))
    frame = build_behavior_frame(raw)
    splits = split_match_ids(frame["match_id"].astype(int).tolist(), seed)
    if not all(splits.values()):
        raise ValueError("At least five replay matches are required for isolated splits")
    train_mask = frame["match_id"].isin(splits["train"])
    means = frame.loc[train_mask, NUMERIC_FEATURES].mean().to_numpy(dtype=np.float32)
    scales = frame.loc[train_mask, NUMERIC_FEATURES].std().to_numpy(dtype=np.float32)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    label_columns = [f"label_{label}" for label in ACTION_LABELS]

    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    def tensors(match_ids: list[int]) -> TensorDataset:
        selected = frame[frame["match_id"].isin(match_ids)]
        numeric = selected.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        numeric = np.nan_to_num((numeric - means) / scales)
        heroes = selected["hero_id"].to_numpy(dtype=np.int64).copy()
        teams = (selected["team"] == "Dire").to_numpy(dtype=np.int64).copy()
        labels = selected[label_columns].to_numpy(dtype=np.float32)
        return TensorDataset(
            torch.from_numpy(numeric),
            torch.from_numpy(heroes),
            torch.from_numpy(teams),
            torch.from_numpy(labels),
        )

    datasets = {name: tensors(ids) for name, ids in splits.items()}
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=use_cuda,
        )
        for name, dataset in datasets.items()
    }
    maximum_hero_id = int(frame["hero_id"].max())

    class BehaviorPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hero_embedding = nn.Embedding(maximum_hero_id + 1, 12)
            self.team_embedding = nn.Embedding(2, 2)
            self.network = nn.Sequential(
                nn.Linear(len(NUMERIC_FEATURES) + 14, 96),
                nn.LayerNorm(96),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, len(ACTION_LABELS)),
            )

        def forward(self, numeric: Any, heroes: Any, teams: Any) -> Any:
            inputs = torch.cat(
                (numeric, self.hero_embedding(heroes), self.team_embedding(teams)), dim=1
            )
            return self.network(inputs)

    model = BehaviorPolicy().to(torch_device)
    train_labels = datasets["train"].tensors[-1].numpy()
    positives = train_labels.sum(axis=0)
    negatives = len(train_labels) - positives
    positive_weight = np.sqrt(negatives / np.maximum(positives, 1))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=torch_device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    def evaluate(loader: Any) -> tuple[float, Any, Any]:
        model.eval()
        losses, truths, probabilities = [], [], []
        with torch.no_grad():
            for numeric, heroes, teams, labels in loader:
                numeric = numeric.to(torch_device, non_blocking=use_cuda)
                heroes = heroes.to(torch_device, non_blocking=use_cuda)
                teams = teams.to(torch_device, non_blocking=use_cuda)
                labels = labels.to(torch_device, non_blocking=use_cuda)
                logits = model(numeric, heroes, teams)
                losses.append(float(loss_function(logits, labels).cpu()))
                truths.append(labels.cpu().numpy())
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return float(np.mean(losses)), np.concatenate(truths), np.concatenate(probabilities)

    best_score = -1.0
    best_state = None
    best_thresholds = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for numeric, heroes, teams, labels in loaders["train"]:
            numeric = numeric.to(torch_device, non_blocking=use_cuda)
            heroes = heroes.to(torch_device, non_blocking=use_cuda)
            teams = teams.to(torch_device, non_blocking=use_cuda)
            labels = labels.to(torch_device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = loss_function(model(numeric, heroes, teams), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss, validation_labels, validation_probabilities = evaluate(
            loaders["validation"]
        )
        thresholds = _best_thresholds(validation_labels, validation_probabilities)
        validation_metrics = _action_metrics(
            validation_labels, validation_probabilities, thresholds
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if validation_metrics["macro_f1"] > best_score + 1e-5:
            best_score = validation_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = thresholds.copy()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None or best_thresholds is None:
        raise RuntimeError("Training produced no behavior policy")
    model.load_state_dict(best_state)
    test_loss, test_labels, test_probabilities = evaluate(loaders["test"])
    test_metrics = _action_metrics(test_labels, test_probabilities, best_thresholds)
    persistence = datasets["test"].tensors[0][
        :, [NUMERIC_FEATURES.index(f"previous_{label}") for label in ACTION_LABELS]
    ].numpy()
    # Previous-action features are standardized; undo that before scoring the baseline.
    previous_indices = [NUMERIC_FEATURES.index(f"previous_{label}") for label in ACTION_LABELS]
    persistence = persistence * scales[previous_indices] + means[previous_indices]
    baseline_metrics = _action_metrics(
        test_labels, persistence, np.full(len(ACTION_LABELS), 0.5, dtype=np.float32)
    )
    test_rows = frame[frame["match_id"].isin(splits["test"])].reset_index(drop=True)
    per_match = {}
    deltas = []
    for match_id, indices in test_rows.groupby("match_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        candidate_score = _action_metrics(
            test_labels[index], test_probabilities[index], best_thresholds
        )["macro_f1"]
        baseline_score = _action_metrics(
            test_labels[index],
            persistence[index],
            np.full(len(ACTION_LABELS), 0.5, dtype=np.float32),
        )["macro_f1"]
        delta = candidate_score - baseline_score
        per_match[str(int(match_id))] = {
            "candidate_macro_f1": candidate_score,
            "persistence_macro_f1": baseline_score,
            "delta": delta,
        }
        deltas.append(delta)
    result = {
        "trajectory_dir": str(trajectory_dir),
        "matches": int(frame["match_id"].nunique()),
        "rows": int(len(frame)),
        "seed": seed,
        "device": str(torch_device),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "features": list(NUMERIC_FEATURES),
        "labels": list(ACTION_LABELS),
        "match_ids": splits,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "thresholds": best_thresholds.tolist(),
        "train_seconds": time.perf_counter() - started,
        "history": history,
        "test_loss": test_loss,
        "test": test_metrics,
        "persistence_baseline": baseline_metrics,
        "per_match_test": per_match,
        "grouped_bootstrap": _grouped_bootstrap(deltas, seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "means": means,
            "scales": scales,
            "features": list(NUMERIC_FEATURES),
            "labels": list(ACTION_LABELS),
            "thresholds": best_thresholds,
            "maximum_hero_id": maximum_hero_id,
        },
        output_dir / "replay-behavior-v1.pt",
    )
    (output_dir / "replay-behavior-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-dir", type=Path, default=Path("artifacts/replay-trajectories")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/replay-behavior-models")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_replay_behavior(
        args.trajectory_dir,
        args.output_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    print(
        f"Validation macro-F1 {result['best_validation_macro_f1']:.4f}; "
        f"test {result['test']['macro_f1']:.4f}; "
        f"persistence {result['persistence_baseline']['macro_f1']:.4f}"
    )
    print(f"Device: {result['device']}; seconds: {result['train_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
