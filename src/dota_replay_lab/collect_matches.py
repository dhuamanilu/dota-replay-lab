"""Collect a reproducible corpus of parsed professional OpenDota matches."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .opendota import OpenDotaError, get_json, get_match


CORPUS_VERSION = "v1"


def paginated_pro_match_ids(
    candidate_limit: int, fetch_json: Callable[[str], Any] = get_json
) -> list[int]:
    """Walk the professional feed backwards without duplicating match IDs."""

    match_ids: list[int] = []
    seen: set[int] = set()
    cutoff: int | None = None
    while len(match_ids) < candidate_limit:
        path = "proMatches" if cutoff is None else f"proMatches?less_than_match_id={cutoff}"
        payload = fetch_json(path)
        if not isinstance(payload, list) or not payload:
            break
        batch = [int(item["match_id"]) for item in payload if isinstance(item, dict) and "match_id" in item]
        new_ids = [match_id for match_id in batch if match_id not in seen]
        if not new_ids:
            break
        match_ids.extend(new_ids)
        seen.update(new_ids)
        next_cutoff = min(batch)
        if cutoff is not None and next_cutoff >= cutoff:
            break
        cutoff = next_cutoff
    return match_ids[:candidate_limit]


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


def excluded_match_ids(manifests: Iterable[Path]) -> set[int]:
    """Load match IDs that must remain outside a newly collected audit corpus."""

    excluded: set[int] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        excluded.update(int(match_id) for match_id in manifest.get("match_ids", []))
    return excluded


def cached_match_ids(matches_dir: Path) -> list[int]:
    """Return cached numeric match IDs newest-first without network access."""

    ids = []
    for path in matches_dir.glob("*.json"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(set(ids), reverse=True)


def hero_names_from_manifests(manifests: Iterable[Path]) -> tuple[dict[int, str], dict[int, str]]:
    """Reuse frozen hero catalogues while resuming a network-limited collection."""

    display: dict[int, str] = {}
    internal: dict[int, str] = {}
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        display.update({int(key): str(value) for key, value in payload.get("hero_names", {}).items()})
        internal.update(
            {int(key): str(value) for key, value in payload.get("hero_internal_names", {}).items()}
        )
    return display, internal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect parsed pro matches and freeze their IDs in a manifest.")
    parser.add_argument("--count", type=int, default=25, help="Number of usable matches to collect")
    parser.add_argument(
        "--candidate-limit", type=int, help="Maximum feed IDs to inspect; defaults to twice --count"
    )
    parser.add_argument("--delay", type=float, default=1.1, help="Seconds between uncached API requests")
    parser.add_argument("--matches-dir", type=Path, default=Path("artifacts/matches"))
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Manifest whose match IDs must not enter the new corpus; may be repeated",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Build the corpus exclusively from cached match JSON files without API calls",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(f"artifacts/corpora/pro-matches-{CORPUS_VERSION}.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 2:
        raise SystemExit("--count must be at least 2 for a match-level train/test split.")
    excluded = excluded_match_ids(args.exclude_manifest)
    candidate_limit = args.candidate_limit or max(args.count * 2, 100)
    candidate_ids = (
        cached_match_ids(args.matches_dir)[:candidate_limit]
        if args.cached_only
        else paginated_pro_match_ids(candidate_limit)
    )
    candidate_ids = [match_id for match_id in candidate_ids if match_id not in excluded]
    selected, rejected = collect_corpus(
        candidate_ids, args.matches_dir, args.count, get_match, delay_seconds=max(args.delay, 0.0)
    )
    hero_names, hero_internal_names = hero_names_from_manifests(args.exclude_manifest)
    if not hero_names:
        hero_catalogue = get_json("constants/heroes")
        hero_names = {
            int(hero.get("id", hero_id)): str(hero["localized_name"])
            for hero_id, hero in hero_catalogue.items()
            if isinstance(hero, dict) and hero.get("localized_name")
        }
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
