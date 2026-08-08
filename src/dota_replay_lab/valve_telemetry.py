"""Parse structured Dota bot telemetry from a Valve console log."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PREFIX = "DRL_TELEMETRY "
ERROR_EVENTS = {"policy_load_error", "decision_error", "query_error", "order_error"}


def parse_lines(lines: Iterable[str]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid = 0
    for line in lines:
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(PREFIX) :].strip()
        try:
            record = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            invalid += 1
            continue
        if not isinstance(record, dict):
            invalid += 1
            continue
        records.append(record)
    return records, invalid


def summarize(records: Iterable[dict[str, Any]], invalid_lines: int = 0) -> dict[str, Any]:
    rows = list(records)
    events = Counter(str(row.get("event", "missing")) for row in rows)
    decisions = Counter(
        str(row.get("action", "missing")) for row in rows if row.get("event") == "decision"
    )
    fallbacks = Counter(str(row["fallback"]) for row in rows if row.get("fallback"))
    orders = Counter(
        str(row.get("order", "missing")) for row in rows if row.get("event") == "order_issued"
    )
    tower_attack_orders = sum(
        1
        for row in rows
        if row.get("event") == "order_issued"
        and row.get("order") == "attack"
        and "tower" in str(row.get("target", "")).lower()
    )
    error_events = Counter(
        str(row.get("event")) for row in rows if row.get("event") in ERROR_EVENTS
    )
    game_times = [
        float(row["game_time"])
        for row in rows
        if isinstance(row.get("game_time"), (int, float))
    ]
    minutes = sorted(
        {int(row["minute"]) for row in rows if isinstance(row.get("minute"), (int, float))}
    )
    schemas = sorted(
        {int(row["schema"]) for row in rows if isinstance(row.get("schema"), (int, float))}
    )
    decisions_by_bot: dict[str, dict[str, Any]] = {}
    activity_by_bot: dict[str, dict[str, Any]] = {}
    counter_fields = (
        "gold",
        "experience",
        "level",
        "xp_to_next_level",
        "last_hits",
        "denies",
        "kills",
        "deaths",
    )
    for row in rows:
        if row.get("event") != "decision" or "player_id" not in row:
            continue
        identity = f'{row["player_id"]}:{row.get("hero_name", "unknown")}'
        snapshot = {
            "player_id": row["player_id"],
            "team_id": row.get("team_id"),
            "hero_name": row.get("hero_name", "unknown"),
            "last_minute": row.get("minute"),
            "last_action": row.get("action"),
        }
        for field in counter_fields:
            if isinstance(row.get(field), (int, float)):
                snapshot[field] = row[field]
        decisions_by_bot[identity] = snapshot

    for row in rows:
        if row.get("event") != "activity" or "player_id" not in row:
            continue
        identity = f'{row["player_id"]}:{row.get("hero_name", "unknown")}'
        idle_seconds = float(row.get("idle_seconds", 0))
        activity_seconds = float(row.get("activity_seconds", 0))
        activity_by_bot[identity] = {
            "player_id": row["player_id"],
            "team_id": row.get("team_id"),
            "hero_name": row.get("hero_name", "unknown"),
            "idle_seconds": idle_seconds,
            "observed_alive_seconds": activity_seconds,
            "idle_ratio": idle_seconds / activity_seconds if activity_seconds else 0,
            "last_action_type": row.get("action_type"),
        }

    total_idle = sum(bot["idle_seconds"] for bot in activity_by_bot.values())
    total_activity = sum(bot["observed_alive_seconds"] for bot in activity_by_bot.values())

    counter_maxima = {
        field: max(
            (float(row[field]) for row in rows if isinstance(row.get(field), (int, float))),
            default=None,
        )
        for field in counter_fields
    }

    return {
        "records": len(rows),
        "invalid_telemetry_lines": invalid_lines,
        "schema_versions": schemas,
        "policy_loaded": events["policy_loaded"],
        "policy_load_errors": events["policy_load_error"],
        "event_counts": dict(sorted(events.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "fallback_counts": dict(sorted(fallbacks.items())),
        "order_counts": dict(sorted(orders.items())),
        "tower_attack_orders": tower_attack_orders,
        "error_event_counts": dict(sorted(error_events.items())),
        "first_game_time": min(game_times) if game_times else None,
        "last_game_time": max(game_times) if game_times else None,
        "observed_game_seconds": max(game_times) - min(game_times) if game_times else 0,
        "minutes_seen": minutes,
        "bot_snapshots": dict(sorted(decisions_by_bot.items())),
        "activity_by_bot": dict(sorted(activity_by_bot.items())),
        "aggregate_idle_ratio": total_idle / total_activity if total_activity else 0,
        "counter_maxima": counter_maxima,
    }


def summarize_log(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        records, invalid = parse_lines(stream)
    result = summarize(records, invalid)
    result["source"] = str(path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("console_log", type=Path, help="Dota console.log produced with -condebug")
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()

    result = summarize_log(args.console_log)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
