"""Train and evaluate small supervised policies without match leakage."""

from __future__ import annotations

import argparse
import itertools
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
XGB_TRIALS = {
    "xgboost_d4_w075": {"max_depth": 4, "n_estimators": 900, "learning_rate": 0.04, "weight_power": 0.75},
    "xgboost_d6_w050": {"max_depth": 6, "n_estimators": 750, "learning_rate": 0.04, "weight_power": 0.50},
    "xgboost_d6_w075": {"max_depth": 6, "n_estimators": 750, "learning_rate": 0.04, "weight_power": 0.75},
    "xgboost_d6_w100": {"max_depth": 6, "n_estimators": 750, "learning_rate": 0.04, "weight_power": 1.00},
    "xgboost_d8_w075": {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.05, "weight_power": 0.75},
}


def apply_probability_biases(probabilities: Any, biases: list[float]) -> Any:
    """Return class indices after applying fixed multiplicative decision biases."""

    import numpy as np

    adjusted = np.asarray(probabilities) * np.asarray(biases)
    return adjusted.argmax(axis=1)


def optimize_probability_biases(probabilities: Any, true_indices: Any) -> tuple[list[float], float]:
    """Tune small class biases on out-of-fold predictions using macro-F1."""

    from sklearn.metrics import f1_score

    best_biases = [1.0] * len(LABELS)
    best_score = float(f1_score(true_indices, apply_probability_biases(probabilities, best_biases), average="macro"))
    grids = (
        [1.0],
        [0.5, 0.75, 1.0, 1.25, 1.5],
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        [0.5, 0.75, 1.0, 1.25, 1.5],
    )
    for candidate in itertools.product(*grids):
        score = float(f1_score(true_indices, apply_probability_biases(probabilities, list(candidate)), average="macro"))
        if score > best_score:
            best_score = score
            best_biases = list(candidate)
    return best_biases, best_score


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


def match_folds(match_ids: list[int], fold_count: int = 5, seed: int = 42) -> list[list[int]]:
    """Partition unique match IDs into deterministic, disjoint validation folds."""

    import numpy as np

    unique = np.array(sorted(set(match_ids)), dtype=int)
    if fold_count < 2 or len(unique) < fold_count:
        raise ValueError("fold_count must be between 2 and the number of unique matches.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return [part.tolist() for part in np.array_split(unique, fold_count)]


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


def train_and_evaluate(
    dataset: Path, output_dir: Path, seed: int, device: str, cv_folds: int = 5
) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
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
    development_ids = splits["train"] + splits["validation"]
    development = frame[frame["match_id"].isin(development_ids)].copy().reset_index(drop=True)
    test = frame[frame["match_id"].isin(splits["test"])].copy().reset_index(drop=True)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    candidates: dict[str, Any] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    }
    xgb_device = "cuda" if device in {"auto", "cuda"} else "cpu"

    def make_xgb(params: dict[str, float], selected_device: str) -> XGBClassifier:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS),
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.5,
            tree_method="hist",
            device=selected_device,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )

    fold_ids = match_folds(development_ids, cv_folds, seed)
    fold_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in [*candidates, *XGB_TRIALS]
    }
    oof_probabilities = {
        name: np.zeros((len(development), len(LABELS)), dtype=float)
        for name in [*candidates, *XGB_TRIALS]
    }

    def aligned_probabilities(model: Any, values: Any) -> Any:
        aligned = np.zeros((values.shape[0], len(LABELS)), dtype=float)
        for column, raw_class in enumerate(model.classes_):
            label_index = (
                int(raw_class)
                if isinstance(raw_class, (int, np.integer))
                else label_to_index[str(raw_class)]
            )
            aligned[:, label_index] = values[:, column]
        return aligned

    for fold_number, validation_ids in enumerate(fold_ids, start=1):
        validation_mask = development["match_id"].isin(validation_ids).to_numpy()
        fit_frame = development[~validation_mask]
        validation_frame = development[validation_mask]
        preprocessor = _preprocessor()
        fit_x = preprocessor.fit_transform(fit_frame[FEATURES])
        validation_x = preprocessor.transform(validation_frame[FEATURES])
        fit_y_text = fit_frame["label"].to_numpy()
        validation_y_text = validation_frame["label"].to_numpy()
        fit_y_index = np.array([label_to_index[value] for value in fit_y_text])
        validation_y_index = np.array([label_to_index[value] for value in validation_y_text])

        for name, template in candidates.items():
            model = clone(template)
            started = time.perf_counter()
            model.fit(fit_x, fit_y_text)
            metrics = _metrics(validation_y_text, model.predict(validation_x))
            oof_probabilities[name][validation_mask] = aligned_probabilities(
                model, model.predict_proba(validation_x)
            )
            metrics.update({"fold": fold_number, "train_seconds": time.perf_counter() - started, "device": "cpu"})
            fold_results[name].append(metrics)

        base_weights = compute_sample_weight("balanced", fit_y_text)
        for name, params in XGB_TRIALS.items():
            actual_device = "cpu" if xgb_device == "cpu-fallback" else xgb_device
            model = make_xgb(params, actual_device)
            started = time.perf_counter()
            try:
                model.fit(
                    fit_x,
                    fit_y_index,
                    sample_weight=np.power(base_weights, params["weight_power"]),
                    eval_set=[(validation_x, validation_y_index)],
                    verbose=False,
                )
            except Exception:
                if device != "auto" or actual_device == "cpu":
                    raise
                xgb_device = "cpu-fallback"
                model = make_xgb(params, "cpu")
                model.fit(
                    fit_x,
                    fit_y_index,
                    sample_weight=np.power(base_weights, params["weight_power"]),
                    eval_set=[(validation_x, validation_y_index)],
                    verbose=False,
                )
            predictions = np.array([LABELS[index] for index in model.predict(validation_x).astype(int)])
            oof_probabilities[name][validation_mask] = aligned_probabilities(
                model, model.predict_proba(validation_x)
            )
            metrics = _metrics(validation_y_text, predictions)
            metrics.update(
                {
                    "fold": fold_number,
                    "train_seconds": time.perf_counter() - started,
                    "device": xgb_device,
                    "parameters": params,
                }
            )
            fold_results[name].append(metrics)

    cross_validation = {}
    for name, results in fold_results.items():
        scores = np.array([result["macro_f1"] for result in results])
        cross_validation[name] = {
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std()),
            "train_seconds": float(sum(result["train_seconds"] for result in results)),
            "device": results[0]["device"],
            "folds": results,
        }
    chosen = max(cross_validation, key=lambda name: cross_validation[name]["macro_f1_mean"])
    development_y_index = np.array([label_to_index[value] for value in development["label"]])
    class_biases, calibrated_oof_macro_f1 = optimize_probability_biases(
        oof_probabilities[chosen], development_y_index
    )

    final_preprocessor = _preprocessor()
    development_x = final_preprocessor.fit_transform(development[FEATURES])
    test_x = final_preprocessor.transform(test[FEATURES])
    started = time.perf_counter()
    if chosen.startswith("xgboost"):
        chosen_params = XGB_TRIALS[chosen]
        actual_device = "cpu" if xgb_device == "cpu-fallback" else xgb_device
        chosen_model = make_xgb(chosen_params, actual_device)
        development_y = np.array([label_to_index[value] for value in development["label"]])
        development_weights = compute_sample_weight("balanced", development["label"])
        chosen_model.fit(
            development_x,
            development_y,
            sample_weight=np.power(development_weights, chosen_params["weight_power"]),
            verbose=False,
        )
    else:
        chosen_model = clone(candidates[chosen])
        chosen_model.fit(development_x, development["label"].to_numpy())
    final_train_seconds = time.perf_counter() - started
    test_probabilities = aligned_probabilities(chosen_model, chosen_model.predict_proba(test_x))
    test_predictions = np.array(
        [LABELS[index] for index in apply_probability_biases(test_probabilities, class_biases)]
    )
    test_metrics = _metrics(test["label"].to_numpy(), test_predictions)

    result = {
        "dataset": str(dataset),
        "seed": seed,
        "features": FEATURES,
        "labels": list(LABELS),
        "rows": {"development": len(development), "test": len(test)},
        "match_ids": splits,
        "cross_validation": cross_validation,
        "cv_folds": fold_ids,
        "chosen_model": chosen,
        "class_biases": {label: class_biases[index] for index, label in enumerate(LABELS)},
        "calibrated_oof_macro_f1": calibrated_oof_macro_f1,
        "final_train_seconds": final_train_seconds,
        "test": test_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": final_preprocessor,
            "model": chosen_model,
            "model_name": chosen,
            "features": FEATURES,
            "labels": LABELS,
            "class_biases": class_biases,
        },
        output_dir / "decision-policy-v1.joblib",
    )
    (output_dir / "decision-policy-v1.metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decision baselines with match-level holdouts.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_and_evaluate(args.dataset, args.output_dir, args.seed, args.device, args.cv_folds)
    cross_validation = result["cross_validation"]
    print("Grouped cross-validation macro-F1 (mean +/- std):")
    for name in sorted(cross_validation):
        summary = cross_validation[name]
        print(f"  {name}: {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f} ({summary['device']})")
    print(f"Chosen: {result['chosen_model']}")
    print(f"Held-out test macro-F1: {result['test']['macro_f1']:.4f}")
    print(f"Saved model and metrics: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
