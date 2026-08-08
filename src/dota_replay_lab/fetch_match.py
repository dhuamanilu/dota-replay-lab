"""Command line entry point for the first Dota Replay Lab experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .opendota import OpenDotaError, get_hero_names, get_match, latest_pro_match_id
from .summary import render_match_summary
from .timeline import render_advantage_svg, render_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one public OpenDota match and summarize it.")
    parser.add_argument("match_id", type=int, nargs="?", help="OpenDota match identifier")
    parser.add_argument(
        "--latest-pro",
        action="store_true",
        help="Use the most recent professional match reported by OpenDota.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/matches"),
        help="Directory for the raw JSON and the Markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.match_id) == args.latest_pro:
        raise SystemExit("Provide exactly one: a match_id or --latest-pro.")

    try:
        match_id = latest_pro_match_id() if args.latest_pro else args.match_id
        match = get_match(match_id)
        hero_names = get_hero_names()
    except OpenDotaError as error:
        raise SystemExit(f"Download failed: {error}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{match_id}.json"
    report_path = args.output_dir / f"{match_id}.md"
    chart_path = args.output_dir / f"{match_id}.advantages.svg"
    raw_path.write_text(json.dumps(match, indent=2, ensure_ascii=False), encoding="utf-8")
    chart_path.write_text(render_advantage_svg(match), encoding="utf-8")
    report = render_match_summary(match, hero_names)
    report += "\n" + render_timeline(match, hero_names)
    report += f"![Ventajas por minuto]({chart_path.name})\n"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved raw data: {raw_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved chart: {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
