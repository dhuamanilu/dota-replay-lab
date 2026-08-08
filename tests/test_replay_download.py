import bz2

import pytest

from dota_replay_lab.replay_download import _copy_response, compression_kind, decompress_replay


def test_bzip2_replay_is_detected_and_validated(tmp_path) -> None:
    source = tmp_path / "replay.compressed"
    destination = tmp_path / "replay.dem"
    source.write_bytes(bz2.compress(b"PBDEMS2\x00payload"))
    assert compression_kind(source.read_bytes()) == "bz2"
    assert decompress_replay(source, destination) == "bz2"
    assert destination.read_bytes() == b"PBDEMS2\x00payload"


def test_unknown_replay_compression_is_rejected() -> None:
    with pytest.raises(ValueError, match="compression magic"):
        compression_kind(b"not-a-replay")


def test_response_copy_is_atomic(tmp_path) -> None:
    from io import BytesIO

    destination = tmp_path / "replay.compressed"
    _copy_response(BytesIO(b"payload"), destination)
    assert destination.read_bytes() == b"payload"
    assert not destination.with_name("replay.compressed.part").exists()
