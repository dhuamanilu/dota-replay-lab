"""Train a compact recurrent policy on complete hero timelines."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

from .train_policy import LABELS, NUMERIC_FEATURES, _metrics, split_match_ids


def fit_normalizer(frame: Any, match_ids: Iterable[int]) -> tuple[Any, Any]:
    """Fit numeric normalization on training matches only."""

    import numpy as np

    fit = frame[frame["match_id"].isin(set(match_ids))][NUMERIC_FEATURES]
    means = fit.mean(axis=0).to_numpy(dtype=np.float32)
    scales = fit.std(axis=0).to_numpy(dtype=np.float32)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return means, scales


def build_sequences(frame: Any, match_ids: Iterable[int], means: Any, scales: Any) -> list[dict[str, Any]]:
    """Build causal per-hero sequences for a match-level split."""

    import numpy as np

    selected = frame[frame["match_id"].isin(set(match_ids))].copy()
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    sequences = []
    for (match_id, player_slot), rows in selected.groupby(["match_id", "player_slot"], sort=True):
        rows = rows.sort_values("state_minute")
        numeric = rows[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        numeric = (numeric - means) / scales
        labels = np.array([label_to_index[value] for value in rows["label"]], dtype=np.int64)
        sequences.append(
            {
                "match_id": int(match_id),
                "player_slot": int(player_slot),
                "hero_id": int(rows.iloc[0]["hero_id"]),
                "team_id": 0 if rows.iloc[0]["team"] == "Radiant" else 1,
                "numeric": numeric,
                "labels": labels,
            }
        )
    return sequences


def make_recurrent_policy(maximum_hero_id: int) -> Any:
    """Construct the compact GRU without importing PyTorch at module import time."""

    from torch import nn

    class RecurrentPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hero_embedding = nn.Embedding(maximum_hero_id + 1, 16)
            self.team_embedding = nn.Embedding(2, 2)
            self.recurrent = nn.GRU(
                len(NUMERIC_FEATURES) + 18,
                96,
                num_layers=2,
                dropout=0.2,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(96),
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, len(LABELS)),
            )

        def forward(self, numeric: Any, heroes: Any, teams: Any) -> Any:
            steps = numeric.shape[1]
            hero = self.hero_embedding(heroes).unsqueeze(1).expand(-1, steps, -1)
            team = self.team_embedding(teams).unsqueeze(1).expand(-1, steps, -1)
            hidden, _ = self.recurrent(torch.cat((numeric, hero, team), dim=-1))
            return self.head(hidden)

    import torch

    return RecurrentPolicy()


def train_sequence_policy(
    dataset: Path,
    output_dir: Path,
    seed: int = 42,
    device: str = "auto",
    epochs: int = 25,
    patience: int = 5,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Fit a GRU using train/validation/test matches without leakage."""

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    frame = pd.read_csv(dataset)
    required = set(NUMERIC_FEATURES + ["match_id", "player_slot", "hero_id", "team", "label"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    unexpected = sorted(set(frame["label"]) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels: {', '.join(unexpected)}")

    splits = split_match_ids(frame["match_id"].astype(int).tolist(), seed)
    means, scales = fit_normalizer(frame, splits["train"])
    split_sequences = {
        name: build_sequences(frame, match_ids, means, scales)
        for name, match_ids in splits.items()
    }
    if not all(split_sequences.values()):
        raise ValueError("Every match split must contain at least one hero sequence.")

    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but PyTorch cannot access it.")
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    class TimelineDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.rows[index]

    def collate(rows: list[dict[str, Any]]) -> tuple[Any, Any, Any, Any]:
        numeric = pad_sequence(
            [torch.from_numpy(row["numeric"]) for row in rows], batch_first=True
        )
        labels = pad_sequence(
            [torch.from_numpy(row["labels"]) for row in rows],
            batch_first=True,
            padding_value=-100,
        )
        heroes = torch.tensor([row["hero_id"] for row in rows], dtype=torch.long)
        teams = torch.tensor([row["team_id"] for row in rows], dtype=torch.long)
        return numeric, labels, heroes, teams

    loaders = {
        name: DataLoader(
            TimelineDataset(rows),
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collate,
            num_workers=0,
            pin_memory=use_cuda,
        )
        for name, rows in split_sequences.items()
    }

    maximum_hero_id = int(frame["hero_id"].max())

    model = make_recurrent_policy(maximum_hero_id).to(torch_device)
    train_labels = frame[frame["match_id"].isin(splits["train"])]["label"]
    counts = train_labels.value_counts()
    weights = np.array(
        [(len(train_labels) / (len(LABELS) * counts.get(label, 1))) ** 0.5 for label in LABELS],
        dtype=np.float32,
    )
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=torch_device), ignore_index=-100
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    def run_epoch(loader: Any, training: bool) -> tuple[float, dict[str, Any]]:
        model.train(training)
        losses = []
        true_values = []
        predictions = []
        for numeric, labels, heroes, teams in loader:
            numeric = numeric.to(torch_device, non_blocking=use_cuda)
            labels = labels.to(torch_device, non_blocking=use_cuda)
            heroes = heroes.to(torch_device, non_blocking=use_cuda)
            teams = teams.to(torch_device, non_blocking=use_cuda)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.amp.autocast(
                "cuda", enabled=use_cuda
            ):
                logits = model(numeric, heroes, teams)
                loss = loss_function(logits.reshape(-1, len(LABELS)), labels.reshape(-1))
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            valid = labels != -100
            true_values.extend(labels[valid].detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=-1)[valid].detach().cpu().tolist())
            losses.append(float(loss.detach().cpu()))
        true_labels = np.array([LABELS[index] for index in true_values])
        predicted_labels = np.array([LABELS[index] for index in predictions])
        return float(np.mean(losses)), _metrics(true_labels, predicted_labels)

    history = []
    best_epoch = 0
    best_score = -1.0
    best_state = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(loaders["train"], True)
        validation_loss, validation_metrics = run_epoch(loaders["validation"], False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if validation_metrics["macro_f1"] > best_score + 1e-5:
            best_score = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no model state.")
    model.load_state_dict(best_state)
    test_loss, test_metrics = run_epoch(loaders["test"], False)
    elapsed = time.perf_counter() - started
    result = {
        "dataset": str(dataset),
        "seed": seed,
        "device": str(torch_device),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "features": list(NUMERIC_FEATURES),
        "labels": list(LABELS),
        "match_ids": splits,
        "sequence_counts": {name: len(rows) for name, rows in split_sequences.items()},
        "parameters": {
            "hero_embedding": 16,
            "team_embedding": 2,
            "hidden_size": 96,
            "layers": 2,
            "dropout": 0.2,
            "batch_size": batch_size,
            "class_weight_power": 0.5,
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "train_seconds": elapsed,
        "history": history,
        "test_loss": test_loss,
        "test": test_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "means": means,
            "scales": scales,
            "features": list(NUMERIC_FEATURES),
            "labels": list(LABELS),
            "maximum_hero_id": maximum_hero_id,
            "parameters": result["parameters"],
        },
        output_dir / "sequence-policy-v1.pt",
    )
    (output_dir / "sequence-policy-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v3.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequence-models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_sequence_policy(
        args.dataset,
        args.output_dir,
        args.seed,
        args.device,
        args.epochs,
        args.patience,
        args.batch_size,
    )
    print(
        f"Best validation macro-F1: {result['best_validation_macro_f1']:.4f} "
        f"at epoch {result['best_epoch']}"
    )
    print(f"Held-out test macro-F1: {result['test']['macro_f1']:.4f}")
    print(f"Device: {result['device']}; seconds: {result['train_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
