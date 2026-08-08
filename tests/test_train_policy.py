from dota_replay_lab.train_policy import FEATURES, XGB_TRIALS, match_folds, split_match_ids


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


def test_xgboost_search_has_distinct_bounded_trials() -> None:
    assert len(XGB_TRIALS) >= 3
    assert len(XGB_TRIALS) == len(set(XGB_TRIALS))
    assert all(0 < trial["weight_power"] <= 1 for trial in XGB_TRIALS.values())


def test_match_folds_are_disjoint_and_cover_development_matches() -> None:
    folds = match_folds(list(range(20)), fold_count=5, seed=3)
    assert all(len(fold) == 4 for fold in folds)
    flattened = [match_id for fold in folds for match_id in fold]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(range(20))
