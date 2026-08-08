import json

from dota_replay_lab.collect_matches import (
    collect_corpus,
    has_minute_series,
    paginated_pro_match_ids,
    write_manifest,
)


def _match(match_id, *, parsed=True):
    series = [0, 1] if parsed else []
    return {
        "match_id": match_id,
        "players": [{"gold_t": series, "xp_t": series, "lh_t": series} for _ in range(10)],
    }


def test_has_minute_series_requires_ten_complete_players() -> None:
    assert has_minute_series(_match(1))
    assert not has_minute_series(_match(1, parsed=False))
    assert not has_minute_series({"players": _match(1)["players"][:9]})


def test_collect_corpus_reuses_cache_and_skips_unparsed(tmp_path) -> None:
    matches_dir = tmp_path / "matches"
    matches_dir.mkdir()
    (matches_dir / "1.json").write_text(json.dumps(_match(1)), encoding="utf-8")
    fetched = []

    def fetcher(match_id):
        fetched.append(match_id)
        return _match(match_id, parsed=match_id != 2)

    selected, rejected = collect_corpus([1, 2, 3], matches_dir, 2, fetcher)

    assert selected == [1, 3]
    assert fetched == [2, 3]
    assert rejected[0]["match_id"] == 2


def test_manifest_freezes_ids_and_hero_names(tmp_path) -> None:
    output = tmp_path / "manifest.json"
    write_manifest(output, [10, 20], [], {1: "Anti-Mage"}, {1: "npc_dota_hero_antimage"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["match_ids"] == [10, 20]
    assert payload["hero_names"] == {"1": "Anti-Mage"}
    assert payload["hero_internal_names"] == {"1": "npc_dota_hero_antimage"}
    assert payload["corpus_version"] == "v1"


def test_pro_feed_pagination_walks_backwards_and_deduplicates() -> None:
    responses = {
        "proMatches": [{"match_id": 5}, {"match_id": 4}, {"match_id": 3}],
        "proMatches?less_than_match_id=3": [{"match_id": 3}, {"match_id": 2}, {"match_id": 1}],
    }
    paths = []

    def fetch(path):
        paths.append(path)
        return responses[path]

    assert paginated_pro_match_ids(5, fetch) == [5, 4, 3, 2, 1]
    assert paths == ["proMatches", "proMatches?less_than_match_id=3"]
