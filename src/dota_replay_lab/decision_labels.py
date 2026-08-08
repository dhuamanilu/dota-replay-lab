"""Transparent heuristic labels for minute-level Dota decisions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

from .hero_state import state_at_minute


LABEL_RULES_VERSION = "v2"
LABEL_PRIORITY = ("fight", "push", "farm")


def _in_minute(event_time: Any, minute: int) -> bool:
    """Return whether an event belongs to the interval ((m-1)*60, m*60]."""

    moment = int(event_time or 0)
    return (minute - 1) * 60 < moment <= minute * 60


def _active_in_teamfight(match: Mapping[str, Any], player_index: int, minute: int) -> bool:
    for teamfight in match.get("teamfights", []) or []:
        if int(teamfight.get("end", 0)) <= (minute - 1) * 60:
            continue
        if int(teamfight.get("start", 0)) > minute * 60:
            continue
        players = teamfight.get("players", []) or []
        if player_index >= len(players):
            continue
        activity = players[player_index] or {}
        if (
            int(activity.get("damage", 0) or 0) > 0
            or int(activity.get("healing", 0) or 0) > 0
            or int(activity.get("deaths", 0) or 0) > 0
            or bool(activity.get("killed"))
        ):
            return True
    return False


def _objective_by_player(match: Mapping[str, Any], player_slot: int, minute: int) -> bool:
    objective_types = {"building_kill", "CHAT_MESSAGE_ROSHAN_KILL", "CHAT_MESSAGE_MINIBOSS_KILL"}
    if player_slot < 0:
        return False
    return any(
        event.get("type") in objective_types
        and int(event.get("player_slot", -1)) == player_slot
        and _in_minute(event.get("time"), minute)
        for event in match.get("objectives", []) or []
    )


def decision_signals(
    match: Mapping[str, Any], player: Mapping[str, Any], player_index: int, minute: int
) -> tuple[str, ...]:
    """Return every v1 signal, ordered by label precedence.

    The result keeps conflicts observable instead of hiding them. ``retreat`` is
    intentionally absent because minute aggregates have no defensible movement
    or safety signal.
    """

    signals: list[str] = []
    own_kill = any(_in_minute(event.get("time"), minute) for event in player.get("kills_log", []) or [])
    if own_kill or _active_in_teamfight(match, player_index, minute):
        signals.append("fight")

    player_slot = int(player.get("player_slot", -1))
    if _objective_by_player(match, player_slot, minute):
        signals.append("push")

    last_hits = player.get("lh_t", []) or []
    current = int(last_hits[min(minute, len(last_hits) - 1)] or 0) if last_hits else 0
    previous_index = min(max(minute - 1, 0), len(last_hits) - 1) if last_hits else 0
    previous = int(last_hits[previous_index] or 0) if last_hits else 0
    if current - previous > 0:
        signals.append("farm")
    return tuple(signals)


def label_decision(
    match: Mapping[str, Any], player: Mapping[str, Any], player_index: int, minute: int
) -> tuple[str, tuple[str, ...]]:
    """Choose one provisional label and return all supporting signals."""

    signals = decision_signals(match, player, player_index, minute)
    label = next((candidate for candidate in LABEL_PRIORITY if candidate in signals), "unknown")
    return label, signals


def iter_decision_rows(
    match: Mapping[str, Any], hero_names: Mapping[int, str]
) -> Iterable[dict[str, Any]]:
    """Yield one learning row for every hero and available whole minute."""

    match_id = int(match.get("match_id", 0))
    for player_index, player in enumerate(match.get("players", []) or []):
        series_lengths = [
            len(player.get(key, []) or [])
            for key in ("gold_t", "xp_t", "lh_t")
            if player.get(key)
        ]
        if not series_lengths:
            continue
        for decision_minute in range(1, max(series_lengths)):
            state = state_at_minute(match, player, decision_minute - 1, hero_names)
            label, signals = label_decision(match, player, player_index, decision_minute)
            previous_signals = decision_signals(match, player, player_index, decision_minute - 1)
            row = asdict(state)
            row["state_minute"] = row.pop("minute")
            row.update(
                {
                    "match_id": match_id,
                    "player_slot": int(player.get("player_slot", -1)),
                    "hero_id": int(player.get("hero_id", 0)),
                    "decision_minute": decision_minute,
                    "previous_fight": int("fight" in previous_signals),
                    "previous_push": int("push" in previous_signals),
                    "previous_farm": int("farm" in previous_signals),
                    "label": label,
                    "signals": "+".join(signals) if signals else "none",
                    "rules_version": LABEL_RULES_VERSION,
                }
            )
            yield row
