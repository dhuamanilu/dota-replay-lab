from dota_replay_lab.train_policy import FEATURES, split_match_ids


def test_match_splits_are_disjoint_and_complete() -> None:
    splits = split_match_ids(list(range(10)), seed=7)
    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(range(10))


def test_policy_features_exclude_target_window_evidence() -> None:
    assert "label" not in FEATURES
    assert "signals" not in FEATURES
    assert "decision_minute" not in FEATURES
