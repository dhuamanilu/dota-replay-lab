from dota_replay_lab.export_lua_policy import DEPTH_CANDIDATES


def test_distillation_depths_are_bounded() -> None:
    assert DEPTH_CANDIDATES == tuple(sorted(DEPTH_CANDIDATES))
    assert min(DEPTH_CANDIDATES) >= 4
    assert max(DEPTH_CANDIDATES) <= 12
