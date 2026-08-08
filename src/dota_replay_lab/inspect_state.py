"""Inspect the minute-level state of all heroes in a downloaded match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hero_state import render_states
from .opendota import OpenDotaError, get_hero_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a per-hero state table for one minute of a match.")
    parser.add_argument("match_id", type=int, help="Previously downloaded OpenDota match identifier")
    parser.add_argument("--minute", type=int, required=True, help="Whole-minute checkpoint to inspect")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/matches"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_dir / f"{args.match_id}.json"
    if not input_path.exists():
        raise SystemExit(f"Missing raw match data: {input_path}. Run fetch_match first.")
    try:
        hero_names = get_hero_names()
    except OpenDotaError as error:
        raise SystemExit(f"Could not load hero names: {error}") from error

    match = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = args.input_dir / f"{args.match_id}.minute-{args.minute}.md"
    output_path.write_text(render_states(match, args.minute, hero_names), encoding="utf-8")
    print(f"Saved state table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
