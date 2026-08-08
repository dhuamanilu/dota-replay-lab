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
        "error_event_counts": dict(sorted(error_events.items())),
        "first_game_time": min(game_times) if game_times else None,
        "last_game_time": max(game_times) if game_times else None,
        "observed_game_seconds": max(game_times) - min(game_times) if game_times else 0,
        "minutes_seen": minutes,
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
