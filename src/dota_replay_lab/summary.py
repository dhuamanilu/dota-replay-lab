"""Convert a raw OpenDota match payload into a readable first-lab report."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _duration(seconds: Any) -> str:
    total = int(seconds or 0)
    return f"{total // 60}:{total % 60:02d}"


def _started_at(timestamp: Any) -> str:
    if not timestamp:
        return "desconocida"
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _team_name(match: dict[str, Any], side: str) -> str:
    key = "radiant_name" if side == "Radiant" else "dire_name"
    return str(match.get(key) or side)


def render_match_summary(match: dict[str, Any]) -> str:
    """Render stable Markdown from the useful high-level match fields."""

    radiant = _team_name(match, "Radiant")
    dire = _team_name(match, "Dire")
    winner = radiant if match.get("radiant_win") else dire
    rows: list[str] = []
    for player in match.get("players", []):
        hero = player.get("hero_name", "unknown hero").removeprefix("npc_dota_hero_").replace("_", " ")
        name = player.get("personaname") or player.get("name") or "anonymous"
        side = "Radiant" if player.get("isRadiant") else "Dire"
        rows.append(
            "| {side} | {hero} | {name} | {k}/{d}/{a} | {gpm} | {xpm} |".format(
                side=side,
                hero=hero,
                name=name,
                k=player.get("kills", 0),
                d=player.get("deaths", 0),
                a=player.get("assists", 0),
                gpm=player.get("gold_per_min", 0),
                xpm=player.get("xp_per_min", 0),
            )
        )

    return "\n".join(
        [
            f"# Match {match['match_id']}",
            "",
            f"**Resultado:** {winner} ganó · **Duración:** {_duration(match.get('duration'))} · "
            f"**Inicio:** {_started_at(match.get('start_time'))}",
            "",
            f"Radiant: **{radiant}**  |  Dire: **{dire}**",
            "",
            "## Jugadores",
            "",
            "| Equipo | Héroe | Jugador | K/D/A | GPM | XPM |",
            "| --- | --- | --- | ---: | ---: | ---: |",
            *rows,
            "",
            "## Qué aprendemos de este primer artefacto",
            "",
            "Este resumen describe el resultado de una partida; todavía no reconstruye el estado "
            "de cada segundo. La siguiente etapa convertirá eventos y posiciones en decisiones "
            "que un modelo pueda aprender.",
            "",
        ]
    )
