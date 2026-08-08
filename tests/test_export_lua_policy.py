from dota_replay_lab.export_lua_policy import DEPTH_CANDIDATES, _raw_teacher_predictions


def test_distillation_depths_are_bounded() -> None:
    assert DEPTH_CANDIDATES == tuple(sorted(DEPTH_CANDIDATES))
    assert min(DEPTH_CANDIDATES) >= 4
    assert max(DEPTH_CANDIDATES) <= 12


def test_raw_teacher_predictions_decode_numeric_classes() -> None:
    class Model:
        def predict(self, transformed):
            return [0, 3]

    predictions = _raw_teacher_predictions({"model": Model()}, None)
    assert list(predictions) == ["farm", "unknown"]
