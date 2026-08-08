"""Build a versioned CSV of provisional high-level Dota decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .decision_labels import LABEL_RULES_VERSION, iter_decision_rows
from .opendota import OpenDotaError, get_hero_names


FIELDNAMES = [
    "match_id", "player_slot", "hero_id", "state_minute", "decision_minute", "team", "hero", "gold", "experience",
    "last_hits", "gold_change", "experience_change", "last_hit_change", "team_gold_advantage",
    "team_experience_advantage", "kills_last_minute", "label", "signals", "rules_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CSV with one provisional decision per hero/minute.")
    parser.add_argument("match_ids", nargs="*", type=int, help="Previously downloaded OpenDota match IDs")
    parser.add_argument("--manifest", type=Path, help="Corpus manifest created by collect_matches")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/matches"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(f"artifacts/datasets/decision-labels-{LABEL_RULES_VERSION}.csv"),
    )
    return parser.parse_args()


def write_dataset(matches: list[dict[str, Any]], hero_names: dict[int, str], output: Path) -> int:
    rows = [row for match in matches for row in iter_decision_rows(match, hero_names)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    match_ids = list(args.match_ids)
    hero_names = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        match_ids.extend(int(match_id) for match_id in manifest.get("match_ids", []))
        hero_names = {int(hero_id): str(name) for hero_id, name in manifest.get("hero_names", {}).items()}
    match_ids = list(dict.fromkeys(match_ids))
    if not match_ids:
        raise SystemExit("Provide at least one match ID or --manifest.")
    matches = []
    for match_id in match_ids:
        path = args.input_dir / f"{match_id}.json"
        if not path.exists():
            raise SystemExit(f"Missing raw match data: {path}. Run fetch_match first.")
        matches.append(json.loads(path.read_text(encoding="utf-8")))
    if not hero_names:
        try:
            hero_names = get_hero_names()
        except OpenDotaError as error:
            raise SystemExit(f"Could not load hero names: {error}") from error

    row_count = write_dataset(matches, hero_names, args.output)
    print(f"Saved {row_count} hero-minute rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
