"""Extract second-level causal hero trajectories from OpenDota parser JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


ACTION_GROUPS = {
    "move": {1, 2},
    "attack": {3, 4},
    "cast": {5, 6, 7, 8, 9, 20},
    "hold": {10, 21},
}
INTERVAL_FIELDS = (
    "time",
    "slot",
    "hero_id",
    "x",
    "y",
    "life_state",
    "level",
    "gold",
    "xp",
    "networth",
    "lh",
    "denies",
    "kills",
    "deaths",
    "assists",
    "towers_killed",
    "roshans_killed",
)
OUTPUT_FIELDS = (
    *INTERVAL_FIELDS,
    "team",
    "alive",
    "movement_distance",
    "orders",
    "move_orders",
    "attack_orders",
    "cast_orders",
    "hold_orders",
    "hero_damage_dealt",
    "hero_damage_received",
)


def _hero_name(value: Any) -> str | None:
    """Normalize parser unit names to the canonical hero suffix."""

    text = str(value or "").lower()
    for prefix in ("npc_dota_hero_", "cdota_unit_hero_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


def action_group(key: Any) -> str | None:
    try:
        value = int(key)
    except (TypeError, ValueError):
        return None
    return next((name for name, values in ACTION_GROUPS.items() if value in values), None)


def extract_second_rows(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join one-second hero snapshots with orders issued during that second."""

    intervals: list[dict[str, Any]] = []
    actions: Counter[tuple[int, int, str]] = Counter()
    totals: Counter[tuple[int, int]] = Counter()
    damage_dealt: Counter[tuple[str, int]] = Counter()
    damage_received: Counter[tuple[str, int]] = Counter()
    hero_slots: dict[str, int] = {}
    for event in events:
        event_type = event.get("type")
        if event_type == "actions" and int(event.get("time", -1)) >= 0:
            slot = int(event.get("slot", -1))
            second = int(event["time"])
            totals[(slot, second)] += 1
            group = action_group(event.get("key"))
            if group:
                actions[(slot, second, group)] += 1
        elif event_type == "interval" and int(event.get("time", -1)) >= 0:
            if "hero_id" in event and "x" in event and "y" in event:
                intervals.append(event)
                name = _hero_name(event.get("unit"))
                if name is not None:
                    hero_slots[name] = int(event["slot"])
        elif event_type == "DOTA_COMBATLOG_DAMAGE" and int(event.get("time", -1)) >= 0:
            second = int(event["time"])
            value = max(float(event.get("value", 0)), 0.0)
            target = _hero_name(event.get("targetname"))
            attacker = _hero_name(event.get("attackername"))
            if event.get("targethero") and target is not None:
                damage_received[(target, second)] += value
                if event.get("attackerhero") and attacker is not None:
                    damage_dealt[(attacker, second)] += value

    dealt_by_slot: Counter[tuple[int, int]] = Counter()
    received_by_slot: Counter[tuple[int, int]] = Counter()
    for (name, second), value in damage_dealt.items():
        if name in hero_slots:
            dealt_by_slot[(hero_slots[name], second)] += value
    for (name, second), value in damage_received.items():
        if name in hero_slots:
            received_by_slot[(hero_slots[name], second)] += value

    previous: dict[int, tuple[float, float, int]] = {}
    rows = []
    for event in sorted(intervals, key=lambda item: (int(item["time"]), int(item["slot"]))):
        slot = int(event["slot"])
        second = int(event["time"])
        x = float(event["x"])
        y = float(event["y"])
        last = previous.get(slot)
        movement = 0.0
        if last is not None and second == last[2] + 1:
            movement = math.hypot(x - last[0], y - last[1])
        previous[slot] = (x, y, second)
        row = {field: event.get(field, 0) for field in INTERVAL_FIELDS}
        row.update(
            {
                "team": "Radiant" if slot < 5 else "Dire",
                "alive": int(int(event.get("life_state", 0)) == 0),
                "movement_distance": movement,
                "orders": totals[(slot, second)],
                **{
                    f"{group}_orders": actions[(slot, second, group)]
                    for group in ACTION_GROUPS
                },
                "hero_damage_dealt": dealt_by_slot[(slot, second)],
                "hero_damage_received": received_by_slot[(slot, second)],
            }
        )
        rows.append(row)
    return rows


def write_second_rows(events_path: Path, output: Path) -> int:
    rows = extract_second_rows(iter_events(events_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.events.with_suffix(".seconds.csv")
    count = write_second_rows(args.events, output)
    print(f"Saved {count} hero-second rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
