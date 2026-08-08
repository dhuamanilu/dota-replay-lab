"""Render a versioned Markdown benchmark from reproducible metric artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def render_benchmark(
    training: Mapping[str, Any], portable: Mapping[str, Any], label_counts: Counter[str]
) -> str:
    cv = training["cross_validation"]
    cv_rows = [
        f"| `{name}` | {summary['macro_f1_mean']:.4f} | {summary['macro_f1_std']:.4f} | {summary['device']} |"
        for name, summary in sorted(cv.items(), key=lambda item: item[1]["macro_f1_mean"], reverse=True)
    ]
    test_rows = [
        f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |"
        for label, metrics in training["test"]["per_class"].items()
    ]
    label_rows = [f"| {label} | {count} |" for label, count in label_counts.most_common()]
    match_count = len({match_id for split in training["match_ids"].values() for match_id in split})
    total_rows = sum(training["rows"].values())
    return "\n".join(
        [
            "# Benchmark de política v2",
            "",
            "## Datos y protocolo",
            "",
            f"- Partidas profesionales: {match_count}.",
            f"- Filas héroe/minuto: {total_rows}.",
            f"- Desarrollo: {training['rows']['development']} filas; test congelado: {training['rows']['test']} filas.",
            "- Selección: macro-F1 medio en cinco folds disjuntos por `match_id`.",
            "- Evaluación final: 20 % de partidas nunca usadas para ajustar ni seleccionar.",
            "",
            "| Etiqueta | Filas |",
            "| --- | ---: |",
            *label_rows,
            "",
            "## Validación cruzada agrupada",
            "",
            "| Modelo | Macro-F1 medio | Desv. estándar | Dispositivo |",
            "| --- | ---: | ---: | --- |",
            *cv_rows,
            "",
            f"Modelo elegido: `{training['chosen_model']}`.",
            "",
            "## Test congelado del modelo GPU",
            "",
            f"- Macro-F1: {training['test']['macro_f1']:.4f}.",
            f"- Balanced accuracy: {training['test']['balanced_accuracy']:.4f}.",
            f"- Accuracy: {training['test']['accuracy']:.4f}.",
            "",
            "| Etiqueta | Precisión | Recall | F1 | Soporte |",
            "| --- | ---: | ---: | ---: | ---: |",
            *test_rows,
            "",
            "## Política Lua destilada",
            "",
            f"- Profundidad: {portable['chosen_depth']}; nodos: {portable['node_count']}.",
            f"- Fidelidad al XGBoost en test: {portable['teacher_fidelity_test']:.4f}.",
            f"- Macro-F1 en test: {portable['student_test']['macro_f1']:.4f}.",
            "",
            "## Interpretación",
            "",
            "El resultado mide imitación de etiquetas heurísticas, no win rate ni nivel de MMR. "
            "`push` sigue siendo la clase más débil y el estado de OpenDota no contiene vida, maná, "
            "cooldowns, visión o posición exacta. La política Lua requiere evaluación dentro de Dota antes "
            "de atribuirle desempeño jugable.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the current policy benchmark as Markdown.")
    parser.add_argument("--training", type=Path, default=Path("artifacts/models/decision-policy-v1.metrics.json"))
    parser.add_argument("--portable", type=Path, default=Path("bots/decision_policy.metrics.json"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/decision-labels-v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark-v2.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training = json.loads(args.training.read_text(encoding="utf-8"))
    portable = json.loads(args.portable.read_text(encoding="utf-8"))
    with args.dataset.open(encoding="utf-8", newline="") as handle:
        label_counts = Counter(row["label"] for row in csv.DictReader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_benchmark(training, portable, label_counts), encoding="utf-8")
    print(f"Saved benchmark report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
