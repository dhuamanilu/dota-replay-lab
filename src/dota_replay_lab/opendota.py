"""Small, dependency-free client for the public OpenDota API."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://api.opendota.com/api"
USER_AGENT = "dota-replay-lab/0.1 (open-source learning project)"


class OpenDotaError(RuntimeError):
    """Raised when OpenDota cannot provide a usable response."""


def get_json(path: str, *, timeout_seconds: int = 30) -> Any:
    """Fetch a JSON document from a relative OpenDota API path."""

    request = Request(
        f"{API_BASE}/{path.lstrip('/')}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except HTTPError as error:
        raise OpenDotaError(f"OpenDota returned HTTP {error.code} for {path}.") from error
    except URLError as error:
        raise OpenDotaError(f"Could not reach OpenDota: {error.reason}.") from error
    except json.JSONDecodeError as error:
        raise OpenDotaError("OpenDota returned invalid JSON.") from error


def latest_pro_match_id() -> int:
    """Return the newest match identifier listed by OpenDota's pro feed."""

    matches = get_json("proMatches")
    if not isinstance(matches, list) or not matches or "match_id" not in matches[0]:
        raise OpenDotaError("OpenDota's professional-match feed had no usable match id.")
    return int(matches[0]["match_id"])


def get_match(match_id: int) -> dict[str, Any]:
    """Return details for one public Dota 2 match."""

    payload = get_json(f"matches/{match_id}")
    if not isinstance(payload, dict) or int(payload.get("match_id", 0)) != match_id:
        raise OpenDotaError(f"OpenDota returned no usable details for match {match_id}.")
    return payload


def get_hero_names() -> dict[int, str]:
    """Return OpenDota's numeric hero-id to display-name lookup table."""

    payload = get_json("constants/heroes")
    if not isinstance(payload, dict):
        raise OpenDotaError("OpenDota's hero catalogue had an unexpected format.")

    names: dict[int, str] = {}
    for key, hero in payload.items():
        if not isinstance(hero, dict):
            continue
        hero_id = hero.get("id", key)
        localized_name = hero.get("localized_name")
        if localized_name:
            names[int(hero_id)] = str(localized_name)
    if not names:
        raise OpenDotaError("OpenDota's hero catalogue was empty.")
    return names
