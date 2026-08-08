"""Train and evaluate small supervised policies without match leakage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


LABELS = ("farm", "fight", "push", "unknown")
CATEGORICAL_FEATURES = ["hero_id", "team"]
NUMERIC_FEATURES = [
    "state_minute",
    "gold",
    "experience",
    "last_hits",
    "gold_change",
    "experience_change",
    "last_hit_change",
    "team_gold_advantage",
    "team_experience_advantage",
    "kills_last_minute",
    "previous_fight",
    "previous_push",
    "previous_farm",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def split_match_ids(match_ids: list[int], seed: int = 42) -> dict[str, list[int]]:
    """Create deterministic 60/20/20 match-level splits."""

    import numpy as np

    unique = np.array(sorted(set(match_ids)), dtype=int)
    if len(unique) < 5:
        raise ValueError("At least five matches are required for train/validation/test splits.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    validation_size = max(1, round(len(unique) * 0.2))
    test_size = max(1, round(len(unique) * 0.2))
    train_size = len(unique) - validation_size - test_size
    return {
        "train": unique[:train_size].tolist(),
        "validation": unique[train_size : train_size + validation_size].tolist(),
        "test": unique[train_size + validation_size :].tolist(),
    }


def _metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

    report = classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    }


def _preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def train_and_evaluate(dataset: Path, output_dir: Path, seed: int, device: str) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    frame = pd.read_csv(dataset)
    missing = sorted(set(FEATURES + ["match_id", "label"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    unexpected_labels = sorted(set(frame["label"]) - set(LABELS))
    if unexpected_labels:
        raise ValueError(f"Unexpected labels: {', '.join(unexpected_labels)}")

    splits = split_match_ids(frame["match_id"].astype(int).tolist(), seed)
    subsets = {name: frame[frame["match_id"].isin(ids)].copy() for name, ids in splits.items()}
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    preprocessor = _preprocessor()
    transformed = {}
    transformed["train"] = preprocessor.fit_transform(subsets["train"][FEATURES])
    transformed["validation"] = preprocessor.transform(subsets["validation"][FEATURES])
    transformed["test"] = preprocessor.transform(subsets["test"][FEATURES])
    y_text = {name: subset["label"].to_numpy() for name, subset in subsets.items()}
    y_index = {name: np.array([label_to_index[value] for value in labels]) for name, labels in y_text.items()}

    validation_results: dict[str, Any] = {}
    candidates: dict[str, Any] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    }
    trained: dict[str, Any] = {}
    for name, model in candidates.items():
        started = time.perf_counter()
        model.fit(transformed["train"], y_text["train"])
        elapsed = time.perf_counter() - started
        validation_results[name] = _metrics(y_text["validation"], model.predict(transformed["validation"]))
        validation_results[name]["train_seconds"] = elapsed
        validation_results[name]["device"] = "cpu"
        trained[name] = model

    xgb_device = "cuda" if device in {"auto", "cuda"} else "cpu"
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        n_estimators=700,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.5,
        tree_method="hist",
        device=xgb_device,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    started = time.perf_counter()
    try:
        xgb.fit(
            transformed["train"],
            y_index["train"],
            sample_weight=compute_sample_weight("balanced", y_text["train"]),
            eval_set=[(transformed["validation"], y_index["validation"])],
            verbose=False,
        )
    except Exception:
        if device != "auto" or xgb_device == "cpu":
            raise
        xgb.set_params(device="cpu")
        xgb_device = "cpu-fallback"
        xgb.fit(
            transformed["train"],
            y_index["train"],
            sample_weight=compute_sample_weight("balanced", y_text["train"]),
            eval_set=[(transformed["validation"], y_index["validation"])],
            verbose=False,
        )
    elapsed = time.perf_counter() - started
    xgb_predictions = np.array([LABELS[index] for index in xgb.predict(transformed["validation"]).astype(int)])
    validation_results["xgboost"] = _metrics(y_text["validation"], xgb_predictions)
    validation_results["xgboost"]["train_seconds"] = elapsed
    validation_results["xgboost"]["device"] = xgb_device
    trained["xgboost"] = xgb

    chosen = max(validation_results, key=lambda name: validation_results[name]["macro_f1"])
    chosen_model = trained[chosen]
    raw_test_predictions = chosen_model.predict(transformed["test"])
    if chosen == "xgboost":
        test_predictions = np.array([LABELS[index] for index in raw_test_predictions.astype(int)])
    else:
        test_predictions = raw_test_predictions
    test_metrics = _metrics(y_text["test"], test_predictions)

    result = {
        "dataset": str(dataset),
        "seed": seed,
        "features": FEATURES,
        "labels": list(LABELS),
        "rows": {name: len(subset) for name, subset in subsets.items()},
        "match_ids": splits,
        "validation": validation_results,
        "chosen_model": chosen,
        "test": test_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": chosen_model,
            "model_name": chosen,
            "features": FEATURES,
            "labels": LABELS,
        },
        output_dir / "decision-policy-v1.joblib",
    )
    (output_dir / "decision-policy-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decision baselines with match-level holdouts.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_and_evaluate(args.dataset, args.output_dir, args.seed, args.device)
    validation = result["validation"]
    print("Validation macro-F1:")
    for name in sorted(validation):
        print(f"  {name}: {validation[name]['macro_f1']:.4f} ({validation[name]['device']})")
    print(f"Chosen: {result['chosen_model']}")
    print(f"Held-out test macro-F1: {result['test']['macro_f1']:.4f}")
    print(f"Saved model and metrics: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
