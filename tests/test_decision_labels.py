import csv

from dota_replay_lab.build_dataset import write_dataset
from dota_replay_lab.decision_labels import iter_decision_rows, label_decision


def _player(**overrides):
    player = {
        "player_slot": 0, "isRadiant": True, "hero_id": 1,
        "gold_t": [100, 250], "xp_t": [0, 100], "lh_t": [0, 3], "kills_log": [],
    }
    player.update(overrides)
    return player


def test_fight_wins_conflict_but_all_signals_remain_visible() -> None:
    player = _player(kills_log=[{"time": 45}])
    match = {
        "players": [player],
        "objectives": [{"time": 50, "type": "building_kill", "player_slot": 0}],
    }
    label, signals = label_decision(match, player, 0, 1)
    assert label == "fight"
    assert signals == ("fight", "push", "farm")


def test_retreat_is_unknown_without_defensible_signal() -> None:
    player = _player(gold_t=[100, 100], xp_t=[0, 0], lh_t=[0, 0])
    label, signals = label_decision({"players": [player]}, player, 0, 1)
    assert label == "unknown"
    assert signals == ()


def test_teamfight_activity_labels_a_support_as_fight() -> None:
    player = _player(lh_t=[0, 0])
    match = {
        "players": [player],
        "teamfights": [{"start": 20, "end": 55, "players": [{"damage": 0, "healing": 120}]}],
    }
    label, _ = label_decision(match, player, 0, 1)
    assert label == "fight"


def test_missing_player_slot_cannot_claim_an_unattributed_objective() -> None:
    player = _player(player_slot=-1, lh_t=[0, 0])
    match = {"players": [player], "objectives": [{"time": 30, "type": "building_kill"}]}
    label, signals = label_decision(match, player, 0, 1)
    assert label == "unknown"
    assert signals == ()


def test_rows_and_csv_include_rules_and_evidence(tmp_path) -> None:
    match = {"match_id": 42, "players": [_player()]}
    rows = list(iter_decision_rows(match, {1: "Anti-Mage"}))
    output = tmp_path / "dataset.csv"
    count = write_dataset([match], {1: "Anti-Mage"}, output)
    assert count == 1
    assert rows[0]["label"] == "farm"
    with output.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["hero"] == "Anti-Mage"
    assert saved[0]["rules_version"] == "v1"
