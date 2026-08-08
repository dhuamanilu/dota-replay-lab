"""Build a human-readable minute-by-minute timeline from parsed OpenDota data."""

from __future__ import annotations

from html import escape
from typing import Any, Mapping


def _signed(value: int) -> str:
    return f"{value:+,}".replace(",", " ")


def _hero_name(player: Mapping[str, Any], hero_names: Mapping[int, str]) -> str:
    return hero_names.get(int(player.get("hero_id", 0)), f"hero {player.get('hero_id', '?')}")


def _player_labels(match: Mapping[str, Any], hero_names: Mapping[int, str]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for player in match.get("players", []):
        if "player_slot" in player:
            labels[int(player["player_slot"])] = _hero_name(player, hero_names)
    return labels


def _event_text(event: Mapping[str, Any], player_labels: Mapping[int, str]) -> str | None:
    moment = int(event.get("time", 0))
    minute = f"{moment // 60}:{moment % 60:02d}"
    event_type = event.get("type")
    if event_type == "CHAT_MESSAGE_FIRSTBLOOD":
        hero = player_labels.get(int(event.get("player_slot", -1)), "unknown hero")
        return f"{minute} — First blood por {hero}."
    if event_type == "building_kill":
        building = str(event.get("key", "edificio")).replace("npc_dota_", "").replace("_", " ")
        return f"{minute} — Cayó {building}."
    return None


def render_timeline(match: Mapping[str, Any], hero_names: Mapping[int, str]) -> str:
    """Render minute advantages and strategic events as Markdown."""

    gold_advantage = [int(value or 0) for value in match.get("radiant_gold_adv", [])]
    xp_advantage = [int(value or 0) for value in match.get("radiant_xp_adv", [])]
    minutes = max(len(gold_advantage), len(xp_advantage))
    rows = []
    for minute in range(minutes):
        gold = gold_advantage[minute] if minute < len(gold_advantage) else 0
        xp = xp_advantage[minute] if minute < len(xp_advantage) else 0
        rows.append(f"| {minute} | {_signed(gold)} | {_signed(xp)} |")

    player_labels = _player_labels(match, hero_names)
    events = [
        text
        for event in match.get("objectives", [])
        if (text := _event_text(event, player_labels)) is not None
    ]
    event_lines = [f"- {event}" for event in events] or ["- OpenDota no incluyó eventos estratégicos parseados."]

    return "\n".join(
        [
            "## Línea de tiempo",
            "",
            "Los valores positivos favorecen a Radiant; los negativos favorecen a Dire. "
            "Esto es una vista de equipo, no todavía la observación parcial de un héroe.",
            "",
            "| Minuto | Ventaja de oro Radiant | Ventaja de experiencia Radiant |",
            "| ---: | ---: | ---: |",
            *rows,
            "",
            "### Eventos estratégicos",
            "",
            *event_lines,
            "",
        ]
    )


def render_advantage_svg(match: Mapping[str, Any]) -> str:
    """Render a small dependency-free SVG chart for team advantages."""

    series = {
        "Oro": [int(value or 0) for value in match.get("radiant_gold_adv", [])],
        "Experiencia": [int(value or 0) for value in match.get("radiant_xp_adv", [])],
    }
    width, height, padding = 900, 360, 48
    values = [value for points in series.values() for value in points] or [0]
    maximum = max(max(abs(value) for value in values), 1)
    steps = max(max(len(points) for points in series.values()), 1)

    def point(index: int, value: int) -> str:
        x = padding + index * (width - 2 * padding) / max(steps - 1, 1)
        y = height / 2 - value * (height / 2 - padding) / maximum
        return f"{x:.1f},{y:.1f}"

    colors = {"Oro": "#f59e0b", "Experiencia": "#22c55e"}
    lines = []
    labels = []
    for index, (name, points) in enumerate(series.items()):
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{colors[name]}" stroke-width="3" points="'
                + " ".join(point(i, value) for i, value in enumerate(points))
                + '" />'
            )
        labels.append(f'<text x="{padding + index * 170}" y="30" fill="{colors[name]}" font-size="16">{escape(name)}</text>')

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#111827" rx="12"/>',
            f'<line x1="{padding}" y1="{height / 2}" x2="{width - padding}" y2="{height / 2}" stroke="#94a3b8" stroke-width="1"/>',
            *labels,
            *lines,
            f'<text x="{padding}" y="{height - 16}" fill="#cbd5e1" font-size="13">Minuto 0</text>',
            f'<text x="{width - padding - 70}" y="{height - 16}" fill="#cbd5e1" font-size="13">Minuto {steps - 1}</text>',
            '<text x="450" y="190" fill="#94a3b8" text-anchor="middle" font-size="13">0 = partida pareja</text>',
            '</svg>',
        ]
    )
