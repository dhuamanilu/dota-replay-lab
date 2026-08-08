import io
from urllib.error import HTTPError

from dota_replay_lab import opendota


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_get_json_retries_rate_limit_using_retry_after(monkeypatch) -> None:
    calls = []
    rate_limit = HTTPError(
        "https://api.opendota.com/api/test", 429, "rate limited", {"Retry-After": "0"}, None
    )
    responses = iter([rate_limit, _Response(b'{"ok": true}')])

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    sleeps = []
    monkeypatch.setattr(opendota, "urlopen", fake_urlopen)
    monkeypatch.setattr(opendota.time, "sleep", sleeps.append)

    assert opendota.get_json("test") == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [0.0]
