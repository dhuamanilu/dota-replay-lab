"""Run the trained decision policy at one historical checkpoint."""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

from .decision_labels import decision_features
from .opendota import OpenDotaError, get_hero_names


def probability_map(classes: Iterable[Any], probabilities: Iterable[float], labels: tuple[str, ...]) -> dict[str, float]:
    result = {}
    for raw_class, probability in zip(classes, probabilities):
        label = labels[int(raw_class)] if isinstance(raw_class, Integral) else str(raw_class)
        result[label] = float(probability)
    return {label: result.get(label, 0.0) for label in labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the next high-level decision for one hero checkpoint.")
    parser.add_argument("match_id", type=int)
    parser.add_argument("--minute", type=int, required=True, help="State checkpoint; prediction is for the next minute")
    parser.add_argument("--player-slot", type=int, required=True)
    parser.add_argument("--matches-dir", type=Path, default=Path("artifacts/matches"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/corpora/pro-matches-v1.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/decision-policy-v1.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions"))
    return parser.parse_args()


def main() -> int:
    import joblib
    import pandas as pd

    args = parse_args()
    match_path = args.matches_dir / f"{args.match_id}.json"
    if not match_path.exists():
        raise SystemExit(f"Missing raw match data: {match_path}")
    if not args.model.exists():
        raise SystemExit(f"Missing trained policy: {args.model}. Run train_policy first.")
    match = json.loads(match_path.read_text(encoding="utf-8"))
    hero_names = None
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        hero_names = {int(hero_id): str(name) for hero_id, name in manifest.get("hero_names", {}).items()}
    if not hero_names:
        try:
            hero_names = get_hero_names()
        except OpenDotaError as error:
            raise SystemExit(f"Could not load hero names: {error}") from error

    indexed_players = list(enumerate(match.get("players", []) or []))
    selected = [(index, player) for index, player in indexed_players if int(player.get("player_slot", -1)) == args.player_slot]
    if not selected:
        raise SystemExit(f"Player slot {args.player_slot} was not found in match {args.match_id}.")
    player_index, player = selected[0]
    row = decision_features(match, player, player_index, max(args.minute, 0), hero_names)
    bundle = joblib.load(args.model)
    feature_frame = pd.DataFrame([row])[bundle["features"]]
    transformed = bundle["preprocessor"].transform(feature_frame)
    probabilities = bundle["model"].predict_proba(transformed)[0]
    classes = bundle["model"].classes_
    labels = tuple(bundle["labels"])
    mapped = probability_map(classes, probabilities, labels)
    predicted = max(mapped, key=mapped.get)
    payload = {
        "match_id": args.match_id,
        "player_slot": args.player_slot,
        "hero": row["hero"],
        "state_minute": row["state_minute"],
        "decision_minute": row["decision_minute"],
        "prediction": predicted,
        "confidence": mapped[predicted],
        "probabilities": mapped,
        "model_name": bundle["model_name"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.match_id}.minute-{args.minute}.slot-{args.player_slot}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved prediction: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
