import numpy as np
import pandas as pd

from dota_replay_lab.train_policy import NUMERIC_FEATURES
from dota_replay_lab.train_sequence_policy import build_sequences, fit_normalizer


def _frame() -> pd.DataFrame:
    rows = []
    for match_id in (1, 2):
        for minute in (0, 1):
            row = {feature: float(match_id + minute) for feature in NUMERIC_FEATURES}
            row.update(
                {
                    "match_id": match_id,
                    "player_slot": 0,
                    "hero_id": 1,
                    "team": "Radiant",
                    "state_minute": minute,
                    "label": "farm" if minute == 0 else "fight",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_normalization_is_fit_only_on_selected_matches() -> None:
    frame = _frame()
    means, scales = fit_normalizer(frame, [1])
    state_index = NUMERIC_FEATURES.index("state_minute")
    assert means[state_index] == 0.5
    assert scales[state_index] > 0


def test_sequences_are_causal_and_grouped_by_match_and_player() -> None:
    frame = _frame()
    means = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)
    scales = np.ones(len(NUMERIC_FEATURES), dtype=np.float32)
    sequences = build_sequences(frame, [2], means, scales)
    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence["match_id"] == 2
    assert sequence["numeric"].shape == (2, len(NUMERIC_FEATURES))
    assert sequence["labels"].tolist() == [0, 1]
