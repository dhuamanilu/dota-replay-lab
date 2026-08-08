"""Parse downloaded replays through a local OpenDota parser into trajectories."""

from __future__ import annotations

import argparse
import http.client
import json
import warnings
from pathlib import Path
from urllib.parse import urlsplit

from .replay_download import download_match_replay
from .replay_events import write_second_rows


def _trajectory_rows(path: Path) -> int:
    """Return the number of data rows in a trajectory CSV."""

    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def parse_replay_file(
    replay_path: Path,
    events_path: Path,
    seconds_path: Path,
    *,
    parser_url: str = "http://localhost:5600",
    keep_events: bool = False,
) -> int:
    """POST one local demo to the parser's event-stream endpoint."""

    parsed = urlsplit(parser_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("parser_url must be a local HTTP URL")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=300
    )
    endpoint = parsed.path.rstrip("/") + "/"
    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(replay_path.stat().st_size))
        connection.endheaders()
        with replay_path.open("rb") as replay:
            while chunk := replay.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Parser HTTP {response.status}: {detail}")
        with events_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    finally:
        connection.close()

    count = write_second_rows(events_path, seconds_path)
    if count == 0:
        raise ValueError(f"Parser produced no hero intervals for {replay_path.name}")
    if not keep_events:
        events_path.unlink()
    return count


def parse_manifest_replays(
    manifest_path: Path,
    matches_dir: Path,
    output_dir: Path,
    *,
    count: int,
    parser_url: str = "http://localhost:5600",
    keep_events: bool = False,
) -> dict[str, int]:
    """Parse ``count`` successful manifest matches, resuming valid CSV files."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: dict[str, int] = {}
    work_dir = output_dir / "_work"
    for match_id in [int(value) for value in manifest.get("match_ids", [])]:
        if len(results) >= count:
            break
        seconds = output_dir / f"{match_id}.seconds.csv"
        existing_rows = _trajectory_rows(seconds)
        if existing_rows:
            results[str(match_id)] = existing_rows
            continue

        match_path = matches_dir / f"{match_id}.json"
        events = output_dir / f"{match_id}.events.jsonl"
        replay = work_dir / f"{match_id}.dem"
        compressed = work_dir / f"{match_id}.dem.compressed"
        try:
            replay, _ = download_match_replay(match_path, work_dir)
            results[str(match_id)] = parse_replay_file(
                replay,
                events,
                seconds,
                parser_url=parser_url,
                keep_events=keep_events,
            )
        except Exception as error:  # Keep collecting when one replay is unavailable.
            warnings.warn(f"Skipping replay {match_id}: {error}", stacklevel=2)
        finally:
            replay.unlink(missing_ok=True)
            compressed.unlink(missing_ok=True)
            if not keep_events:
                events.unlink(missing_ok=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--matches-dir", type=Path, default=Path("artifacts/matches"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/replay-trajectories"))
    parser.add_argument("--parser-url", default="http://localhost:5600")
    parser.add_argument("--keep-events", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    results = parse_manifest_replays(
        args.manifest,
        args.matches_dir,
        args.output_dir,
        count=args.count,
        parser_url=args.parser_url,
        keep_events=args.keep_events,
    )
    total = sum(results.values())
    print(f"Parsed {len(results)} replays and {total} hero-second rows")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
