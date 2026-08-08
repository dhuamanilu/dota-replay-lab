"""Train and audit next-second hero-combat classifiers from raw replays."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .replay_behavior import load_trajectory_corpus
from .replay_combat import (
    COMBAT_FEATURES,
    COMBAT_LABELS,
    RUNTIME_COMBAT_FEATURES,
    build_combat_frame,
)
from .train_policy import split_match_ids
from .train_replay_behavior import _grouped_bootstrap


def _metrics(labels: Any, probabilities: Any, thresholds: Any) -> dict[str, Any]:
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
        for index, label in enumerate(COMBAT_LABELS)
    }
    result["macro_f1"] = float(np.mean(f1))
    return result


def _thresholds(labels: Any, probabilities: Any) -> Any:
    import numpy as np
    from sklearn.metrics import f1_score

    values = np.arange(0.05, 0.81, 0.025)
    return np.asarray(
        [
            max(
                values,
                key=lambda value: f1_score(
                    labels[:, index], probabilities[:, index] >= value, zero_division=0
                ),
            )
            for index in range(len(COMBAT_LABELS))
        ]
    )


def train_replay_combat(
    trajectory_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    iterations: int = 150,
    horizon: int = 5,
    runtime_features: bool = False,
) -> dict[str, Any]:
    """Fit bounded tree ensembles with match-isolated evaluation."""

    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    raw = load_trajectory_corpus(trajectory_dir.glob("*.seconds.csv"))
    frame = build_combat_frame(raw, horizon=horizon)
    feature_names = RUNTIME_COMBAT_FEATURES if runtime_features else COMBAT_FEATURES
    splits = split_match_ids(frame["match_id"].astype(int).tolist(), seed)
    label_columns = [f"label_{label}" for label in COMBAT_LABELS]

    def selected(name: str) -> tuple[Any, Any, Any]:
        rows = frame[frame["match_id"].isin(splits[name])]
        features = np.nan_to_num(rows.loc[:, feature_names].to_numpy(dtype=np.float32))
        labels = rows.loc[:, label_columns].to_numpy(dtype=np.int8)
        return rows.reset_index(drop=True), features, labels

    train_rows, train_features, train_labels = selected("train")
    validation_rows, validation_features, validation_labels = selected("validation")
    test_rows, test_features, test_labels = selected("test")
    if min(len(train_rows), len(validation_rows), len(test_rows)) == 0:
        raise ValueError("Every match split must contain combat rows")

    started = time.perf_counter()
    models = []
    validation_probabilities = []
    test_probabilities = []
    for index in range(len(COMBAT_LABELS)):
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=iterations,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1e-3,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(train_features, train_labels[:, index])
        models.append(model)
        validation_probabilities.append(model.predict_proba(validation_features)[:, 1])
        test_probabilities.append(model.predict_proba(test_features)[:, 1])
    validation_probabilities = np.column_stack(validation_probabilities)
    test_probabilities = np.column_stack(test_probabilities)
    thresholds = _thresholds(validation_labels, validation_probabilities)
    validation_metrics = _metrics(validation_labels, validation_probabilities, thresholds)
    test_metrics = _metrics(test_labels, test_probabilities, thresholds)
    persistence = np.column_stack(
        (
            test_rows["recent_damage_dealt"].to_numpy() > 0,
            test_rows["recent_damage_received"].to_numpy() > 0,
        )
    ).astype(np.float32)
    baseline_metrics = _metrics(test_labels, persistence, np.full(2, 0.5))

    per_match = {}
    deltas = []
    for match_id, indices in test_rows.groupby("match_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        candidate = _metrics(
            test_labels[index], test_probabilities[index], thresholds
        )["macro_f1"]
        baseline = _metrics(
            test_labels[index], persistence[index], np.full(2, 0.5)
        )["macro_f1"]
        delta = candidate - baseline
        per_match[str(int(match_id))] = {
            "candidate_macro_f1": candidate,
            "persistence_macro_f1": baseline,
            "delta": delta,
        }
        deltas.append(delta)

    result = {
        "trajectory_dir": str(trajectory_dir),
        "matches": int(frame["match_id"].nunique()),
        "rows": int(len(frame)),
        "features": list(feature_names),
        "feature_profile": "runtime" if runtime_features else "full",
        "labels": list(COMBAT_LABELS),
        "match_ids": splits,
        "iterations": iterations,
        "horizon_seconds": horizon,
        "thresholds": thresholds.tolist(),
        "train_seconds": time.perf_counter() - started,
        "validation": validation_metrics,
        "test": test_metrics,
        "persistence_baseline": baseline_metrics,
        "per_match_test": per_match,
        "grouped_bootstrap": _grouped_bootstrap(deltas, seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "features": list(feature_names),
            "labels": list(COMBAT_LABELS),
            "thresholds": thresholds,
        },
        output_dir / "replay-combat-v1.joblib",
    )
    (output_dir / "replay-combat-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-dir", type=Path, default=Path("artifacts/replay-combat-trajectories")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/replay-combat-models")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--runtime-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_replay_combat(
        args.trajectory_dir,
        args.output_dir,
        seed=args.seed,
        iterations=args.iterations,
        horizon=args.horizon,
        runtime_features=args.runtime_features,
    )
    print(
        f"Validation macro-F1 {result['validation']['macro_f1']:.4f}; "
        f"test {result['test']['macro_f1']:.4f}; "
        f"persistence {result['persistence_baseline']['macro_f1']:.4f}"
    )
    print(f"Rows: {result['rows']}; seconds: {result['train_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
