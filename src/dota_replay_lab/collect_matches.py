"""Collect a reproducible corpus of parsed professional OpenDota matches."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .opendota import OpenDotaError, get_hero_names, get_json, get_match


CORPUS_VERSION = "v1"


def has_minute_series(match: Mapping[str, Any]) -> bool:
    """Return whether a match has enough parsed data for hero-minute rows."""

    players = match.get("players", []) or []
    if len(players) != 10:
        return False
    return all(
        len(player.get("gold_t", []) or []) > 1
        and len(player.get("xp_t", []) or []) > 1
        and len(player.get("lh_t", []) or []) > 1
        for player in players
    )


def collect_corpus(
    candidate_ids: Iterable[int],
    output_dir: Path,
    target_count: int,
    fetch_match: Callable[[int], dict[str, Any]],
    *,
    delay_seconds: float = 0.0,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Cache usable matches until target_count and return IDs plus rejection records."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[int] = []
    rejected: list[dict[str, Any]] = []
    for match_id in candidate_ids:
        if len(selected) >= target_count:
            break
        path = output_dir / f"{match_id}.json"
        try:
            if path.exists():
                match = json.loads(path.read_text(encoding="utf-8"))
                source = "cache"
            else:
                match = fetch_match(match_id)
                path.write_text(json.dumps(match, ensure_ascii=False, indent=2), encoding="utf-8")
                source = "api"
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        except (OpenDotaError, OSError, ValueError, json.JSONDecodeError) as error:
            rejected.append({"match_id": match_id, "reason": str(error)})
            continue
        if has_minute_series(match):
            selected.append(match_id)
        else:
            rejected.append({"match_id": match_id, "reason": f"missing minute series ({source})"})
    return selected, rejected


def write_manifest(
    output: Path,
    match_ids: list[int],
    rejected: list[dict[str, Any]],
    hero_names: Mapping[int, str],
    hero_internal_names: Mapping[int, str] | None = None,
) -> None:
    payload = {
        "corpus_version": CORPUS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "match_ids": match_ids,
        "hero_names": {str(hero_id): name for hero_id, name in sorted(hero_names.items())},
        "hero_internal_names": {
            str(hero_id): name for hero_id, name in sorted((hero_internal_names or {}).items())
        },
        "rejected": rejected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect parsed pro matches and freeze their IDs in a manifest.")
    parser.add_argument("--count", type=int, default=25, help="Number of usable matches to collect")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.1, help="Seconds between uncached API requests")
    parser.add_argument("--matches-dir", type=Path, default=Path("artifacts/matches"))
    parser.add_argument(
        "--manifest", type=Path, default=Path(f"artifacts/corpora/pro-matches-{CORPUS_VERSION}.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 2:
        raise SystemExit("--count must be at least 2 for a match-level train/test split.")
    feed = get_json("proMatches")
    candidate_ids = [int(item["match_id"]) for item in feed[: args.candidate_limit] if "match_id" in item]
    selected, rejected = collect_corpus(
        candidate_ids, args.matches_dir, args.count, get_match, delay_seconds=max(args.delay, 0.0)
    )
    hero_names = get_hero_names()
    hero_catalogue = get_json("constants/heroes")
    hero_internal_names = {
        int(hero.get("id", hero_id)): str(hero["name"])
        for hero_id, hero in hero_catalogue.items()
        if isinstance(hero, dict) and hero.get("name")
    }
    write_manifest(args.manifest, selected, rejected, hero_names, hero_internal_names)
    print(f"Saved corpus manifest with {len(selected)} matches: {args.manifest}")
    if len(selected) < args.count:
        print(f"Warning: requested {args.count}, but only {len(selected)} usable matches were available.")
    return 0 if len(selected) >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
