from dota_replay_lab.predict_policy import probability_map


def test_probability_map_handles_numeric_model_classes() -> None:
    mapped = probability_map([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4], ("farm", "fight", "push", "unknown"))
    assert mapped["push"] == 0.3
    assert mapped["unknown"] == 0.4
