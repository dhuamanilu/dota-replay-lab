"""Download and decompress a Valve Dota 2 replay without launching the client."""

from __future__ import annotations

import argparse
import bz2
import json
import shutil
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen


BZ2_MAGIC = b"BZh"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
DEMO_MAGIC = b"PBDEMS2\x00"


def compression_kind(header: bytes) -> str:
    """Identify Valve's actual compression by magic bytes, not URL suffix."""

    if header.startswith(BZ2_MAGIC):
        return "bz2"
    if header.startswith(ZSTD_MAGIC):
        return "zstd"
    raise ValueError(f"Unsupported replay compression magic: {header[:4].hex()}")


def decompress_replay(source: Path, destination: Path) -> str:
    """Stream a BZip2 or Zstandard replay into a validated Source 2 demo."""

    with source.open("rb") as handle:
        header = handle.read(4)
    kind = compression_kind(header)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "bz2":
        with bz2.open(source, "rb") as compressed, destination.open("wb") as output:
            shutil.copyfileobj(compressed, output)
    else:
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError('Install the replay extra: pip install -e ".[replay]"') from error
        with source.open("rb") as compressed, destination.open("wb") as output:
            zstandard.ZstdDecompressor().copy_stream(compressed, output)
    with destination.open("rb") as handle:
        demo_header = handle.read(len(DEMO_MAGIC))
    if demo_header != DEMO_MAGIC:
        raise ValueError("Decompressed file is not a Source 2 Dota replay")
    return kind


def _copy_response(response: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size == 0:
            raise ValueError("Replay download was empty")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_match_replay(match_path: Path, output_dir: Path) -> tuple[Path, str]:
    """Download the replay URL from a cached OpenDota match and decompress it."""

    match = json.loads(match_path.read_text(encoding="utf-8"))
    match_id = int(match.get("match_id", 0))
    replay_url = match.get("replay_url")
    if not match_id or not replay_url:
        raise ValueError(f"Match JSON has no replay URL: {match_path}")
    compressed = output_dir / f"{match_id}.dem.compressed"
    replay = output_dir / f"{match_id}.dem"
    if not compressed.exists() or compressed.stat().st_size == 0:
        compressed.unlink(missing_ok=True)
        request = Request(
            str(replay_url),
            headers={"User-Agent": "dota-replay-lab/0.1", "Accept": "application/octet-stream"},
        )
        with urlopen(request, timeout=120) as response:
            _copy_response(response, compressed)
    kind = decompress_replay(compressed, replay)
    return replay, kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_id", type=int)
    parser.add_argument("--matches-dir", type=Path, default=Path("artifacts/matches"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/replays"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay, kind = download_match_replay(
        args.matches_dir / f"{args.match_id}.json", args.output_dir
    )
    print(f"Saved {kind} replay: {replay} ({replay.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
