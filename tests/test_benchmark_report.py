from collections import Counter

from dota_replay_lab.benchmark_report import render_benchmark


def test_benchmark_report_includes_protocol_and_portable_metrics() -> None:
    training = {
        "cross_validation": {"model": {"macro_f1_mean": 0.4, "macro_f1_std": 0.01, "device": "cuda"}},
        "match_ids": {"train": [1, 2, 3], "validation": [4], "test": [5]},
        "rows": {"development": 40, "test": 10},
        "chosen_model": "model",
        "test": {
            "macro_f1": 0.39,
            "balanced_accuracy": 0.41,
            "accuracy": 0.5,
            "per_class": {
                "farm": {"precision": 0.5, "recall": 0.6, "f1": 0.55, "support": 10}
            },
        },
    }
    portable = {
        "chosen_strategy": "labels_balanced",
        "chosen_depth": 8,
        "node_count": 100,
        "teacher_fidelity_test": 0.9,
        "student_test": {"macro_f1": 0.38},
    }
    report = render_benchmark(training, portable, Counter({"farm": 50}))
    assert "cinco folds" in report
    assert "0.9000" in report
    assert "Partidas profesionales: 5" in report
