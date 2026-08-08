import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from dota_replay_lab.export_combat_lua import combat_bundle_to_lua


def test_combat_ensemble_exports_valid_lua() -> None:
    from lupa import LuaRuntime

    features = np.arange(60, dtype=np.float32).reshape(30, 2)
    labels = (features[:, 0] > 25).astype(int)
    models = []
    for offset in (0, 1):
        model = HistGradientBoostingClassifier(max_iter=3, min_samples_leaf=2, random_state=4)
        model.fit(features, np.roll(labels, offset))
        models.append(model)
    source = combat_bundle_to_lua(
        {
            "models": models,
            "features": ["a", "b"],
            "labels": ["engage", "threat"],
            "thresholds": np.array([0.5, 0.5]),
        }
    )
    lua = LuaRuntime(unpack_returned_tuples=True)
    module = lua.execute(source)
    prediction = module["predict"](lua.table_from({"a": 50, "b": 51}))
    assert 0 <= prediction["engage_probability"] <= 1
