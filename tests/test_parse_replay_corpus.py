import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dota_replay_lab.parse_replay_corpus import parse_manifest_replays, parse_replay_file


class _ParserHandler(BaseHTTPRequestHandler):
    received = b""

    def do_POST(self) -> None:
        type(self).received = self.rfile.read(int(self.headers["Content-Length"]))
        payload = (
            b'{"type":"interval","time":0,"slot":0,"hero_id":1,'
            b'"x":10,"y":20,"life_state":0}\n'
        )
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_replay_file_is_streamed_to_root_parser_endpoint(tmp_path) -> None:
    replay = tmp_path / "replay.dem"
    replay.write_bytes(b"PBDEMS2\x00payload")
    events = tmp_path / "events.jsonl"
    seconds = tmp_path / "seconds.csv"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ParserHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        rows = parse_replay_file(
            replay,
            events,
            seconds,
            parser_url=f"http://127.0.0.1:{server.server_port}",
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert rows == 1
    assert _ParserHandler.received == replay.read_bytes()
    assert not events.exists()
    assert seconds.read_text(encoding="utf-8").count("\n") == 2


def test_manifest_parser_resumes_existing_trajectory_without_network(tmp_path) -> None:
    matches = tmp_path / "matches"
    output = tmp_path / "output"
    matches.mkdir()
    output.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"match_ids": [1]}), encoding="utf-8")
    (matches / "1.json").write_text(
        json.dumps({"match_id": 1, "replay_url": "http://invalid"}), encoding="utf-8"
    )
    (output / "1.seconds.csv").write_text("time,slot\n0,0\n1,0\n", encoding="utf-8")
    assert parse_manifest_replays(manifest, matches, output, count=1) == {"1": 2}
