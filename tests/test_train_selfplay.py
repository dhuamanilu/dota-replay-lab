from pathlib import Path

import numpy as np
import torch

from dota_replay_lab.train_policy import LABELS, NUMERIC_FEATURES
from dota_replay_lab.train_sequence_policy import make_recurrent_policy
from dota_replay_lab.train_selfplay import _wilson, load_imitation_initialization


def test_selfplay_actor_reuses_imitation_weights_exactly(tmp_path: Path) -> None:
    imitation = make_recurrent_policy(maximum_hero_id=20)
    checkpoint = tmp_path / "sequence-policy.pt"
    bundle = {
        "state_dict": imitation.state_dict(),
        "maximum_hero_id": 20,
        "means": np.zeros(len(NUMERIC_FEATURES), dtype=np.float32),
        "scales": np.ones(len(NUMERIC_FEATURES), dtype=np.float32),
        "features": list(NUMERIC_FEATURES),
        "labels": list(LABELS),
    }
    torch.save(bundle, checkpoint)
    model, bundle = load_imitation_initialization(
        checkpoint,
        torch.device("cpu"),
    )
    state = model.state_dict()
    for name, value in bundle["state_dict"].items():
        assert torch.equal(state[name], value)


def test_wilson_interval_tightens_around_clear_win_rate() -> None:
    lower, upper = _wilson(80, 100)
    assert 0.70 < lower < 0.80 < upper < 0.90
