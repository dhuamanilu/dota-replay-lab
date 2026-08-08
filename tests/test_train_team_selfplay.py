import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dota_replay_lab.train_team_selfplay import (
    COMPOSITIONS,
    assign_compositions,
    composition_prior,
)


def test_all_five_hero_compositions_are_enumerated() -> None:
    assert len(COMPOSITIONS) == 56
    assert len(set(COMPOSITIONS)) == 56
    assert all(sum(value) == 5 for value in COMPOSITIONS)


def test_assignment_respects_selected_composition() -> None:
    logits = torch.randn(3, 5, 4)
    selected = torch.tensor(
        [COMPOSITIONS.index((3, 2, 0, 0)), COMPOSITIONS.index((1, 1, 2, 1)), COMPOSITIONS.index((0, 0, 5, 0))]
    )
    actions = assign_compositions(logits, selected)
    for row, composition_index in zip(actions, selected):
        actual = tuple(int((row == action).sum()) for action in range(4))
        assert actual == COMPOSITIONS[int(composition_index)]


def test_composition_prior_is_smoothed_and_normalized() -> None:
    import pandas as pd

    rows = []
    for slot, label in enumerate(("farm", "farm", "fight", "push", "unknown")):
        rows.append({"match_id": 1, "team": "Radiant", "state_minute": 0, "player_slot": slot, "label": label})
    prior = composition_prior(pd.DataFrame(rows), [1])
    assert np.isclose(prior.sum(), 1)
    assert (prior > 0).all()
