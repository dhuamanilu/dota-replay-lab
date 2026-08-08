"""Create a compact per-hero state from OpenDota's minute-level replay series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HeroMinuteState:
    minute: int
    team: str
    hero: str
    gold: int
    experience: int
    last_hits: int
    gold_change: int
    experience_change: int
    last_hit_change: int
    team_gold_advantage: int
    team_experience_advantage: int
    kills_last_minute: int


def _at(values: list[Any], minute: int) -> int:
    if not values:
        return 0
    return int(values[min(minute, len(values) - 1)] or 0)


def _change(values: list[Any], minute: int) -> int:
    return _at(values, minute) - _at(values, max(minute - 1, 0))


def _kills_in_window(player: Mapping[str, Any], start_seconds: int, end_seconds: int) -> int:
    return sum(
        1
        for event in player.get("kills_log", []) or []
        if start_seconds < int(event.get("time", -1)) <= end_seconds
    )


def state_at_minute(
    match: Mapping[str, Any], player: Mapping[str, Any], minute: int, hero_names: Mapping[int, str]
) -> HeroMinuteState:
    """Return one player's observable proxy state at a whole-minute checkpoint.

    OpenDota exposes these series at one-minute resolution. This is therefore a
    learning representation, not a claim that it matches the exact in-game
    observation available to a player at every frame.
    """

    minute = max(minute, 0)
    team = "Radiant" if player.get("isRadiant") else "Dire"
    sign = 1 if team == "Radiant" else -1
    return HeroMinuteState(
        minute=minute,
        team=team,
        hero=hero_names.get(int(player.get("hero_id", 0)), f"hero {player.get('hero_id', '?')}"),
        gold=_at(player.get("gold_t", []), minute),
        experience=_at(player.get("xp_t", []), minute),
        last_hits=_at(player.get("lh_t", []), minute),
        gold_change=_change(player.get("gold_t", []), minute),
        experience_change=_change(player.get("xp_t", []), minute),
        last_hit_change=_change(player.get("lh_t", []), minute),
        team_gold_advantage=sign * _at(match.get("radiant_gold_adv", []), minute),
        team_experience_advantage=sign * _at(match.get("radiant_xp_adv", []), minute),
        kills_last_minute=_kills_in_window(player, (minute - 1) * 60, minute * 60),
    )


def render_states(match: Mapping[str, Any], minute: int, hero_names: Mapping[int, str]) -> str:
    """Render the ten hero states at one minute as an inspectable Markdown table."""

    states = [state_at_minute(match, player, minute, hero_names) for player in match.get("players", [])]
    rows = [
        "| {team} | {hero} | {gold} | {xp} | {lh} | {gold_delta:+} | {xp_delta:+} | {lh_delta:+} | {gadv:+} | {xadv:+} | {kills} |".format(
            team=state.team,
            hero=state.hero,
            gold=state.gold,
            xp=state.experience,
            lh=state.last_hits,
            gold_delta=state.gold_change,
            xp_delta=state.experience_change,
            lh_delta=state.last_hit_change,
            gadv=state.team_gold_advantage,
            xadv=state.team_experience_advantage,
            kills=state.kills_last_minute,
        )
        for state in states
    ]
    return "\n".join(
        [
            f"# Estados de héroes — minuto {minute}",
            "",
            "Las ventajas están expresadas desde la perspectiva de cada equipo: positivo significa ventaja propia. "
            "Los cambios comparan este minuto con el anterior.",
            "",
            "| Equipo | Héroe | Oro | XP | LH | Δ oro | Δ XP | Δ LH | Ventaja oro | Ventaja XP | Kills último min. |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Límites conscientes",
            "",
            "Este estado no incluye vida, maná, cooldowns, visión ni posición exacta. OpenDota entrega estas series "
            "a nivel de minuto; para una política que controle Dota necesitaremos el replay `.dem` o la API de Valve.",
            "",
        ]
    )
