import pandas as pd
import pytest

from dota_replay_lab.offline_rl import (
    add_advantage_weights,
    add_discounted_returns,
    add_replay_rewards,
)


def _row(match: int, slot: int, team: str, minute: int, gold: float) -> dict:
    return {
        "match_id": match,
        "player_slot": slot,
        "team": team,
        "state_minute": minute,
        "decision_minute": minute + 1,
        "gold_change": gold,
        "experience_change": 0,
        "last_hit_change": 0,
        "deny_change": 0,
        "kills_last_minute": 0,
    }


def test_reward_uses_next_contiguous_state_and_is_zero_sum() -> None:
    frame = pd.DataFrame(
        [
            _row(1, 0, "Radiant", 0, 0),
            _row(1, 0, "Radiant", 1, 100),
            _row(1, 128, "Dire", 0, 0),
            _row(1, 128, "Dire", 1, 50),
        ]
    )
    rewarded = add_replay_rewards(frame, team_spirit=1.0)
    assert rewarded.loc[0, "individual_reward"] == pytest.approx(0.6)
    assert rewarded.loc[2, "individual_reward"] == pytest.approx(0.3)
    assert rewarded.loc[0, "replay_reward"] == pytest.approx(0.3)
    assert rewarded.loc[2, "replay_reward"] == pytest.approx(-0.3)
    assert not rewarded.loc[1, "reward_valid"]
    assert rewarded.loc[1, "replay_reward"] == 0


def test_reward_never_crosses_gap_or_timeline_boundary() -> None:
    frame = pd.DataFrame(
        [
            _row(1, 0, "Radiant", 0, 0),
            _row(1, 0, "Radiant", 2, 999),
            _row(2, 0, "Radiant", 0, 999),
        ]
    )
    rewarded = add_replay_rewards(frame)
    assert rewarded["reward_valid"].tolist() == [False, False, False]
    assert rewarded["individual_reward"].sum() == 0


def test_discounted_returns_reset_at_each_hero_boundary() -> None:
    frame = pd.DataFrame(
        [
            _row(1, 0, "Radiant", 0, 0),
            _row(1, 0, "Radiant", 1, 100),
            _row(1, 128, "Dire", 0, 0),
            _row(1, 128, "Dire", 1, 0),
        ]
    )
    returned = add_discounted_returns(add_replay_rewards(frame), gamma=0.5)
    assert returned.loc[0, "discounted_return"] == returned.loc[0, "replay_reward"]
    assert returned.loc[1, "discounted_return"] == 0


def test_reward_configuration_is_validated() -> None:
    frame = pd.DataFrame([_row(1, 0, "Radiant", 0, 0)])
    with pytest.raises(ValueError, match="team_spirit"):
        add_replay_rewards(frame, team_spirit=1.1)


def test_reward_enrichment_preserves_input_row_identity_and_order() -> None:
    frame = pd.DataFrame(
        [
            _row(2, 0, "Radiant", 0, 20),
            _row(1, 128, "Dire", 0, 10),
            _row(1, 128, "Dire", 1, 30),
            _row(2, 0, "Radiant", 1, 40),
        ],
        index=[40, 10, 30, 20],
    )
    frame["marker"] = ["a", "b", "c", "d"]
    rewarded = add_replay_rewards(frame)
    assert rewarded.index.tolist() == [40, 10, 30, 20]
    assert rewarded["marker"].tolist() == ["a", "b", "c", "d"]


def test_advantage_statistics_are_fit_on_training_matches_only() -> None:
    rows = [
        _row(1, 0, "Radiant", 0, 0),
        _row(1, 0, "Radiant", 1, 100),
        _row(1, 0, "Radiant", 2, 200),
        _row(1, 128, "Dire", 0, 0),
        _row(1, 128, "Dire", 1, 0),
        _row(1, 128, "Dire", 2, 0),
        _row(2, 0, "Radiant", 0, 0),
        _row(2, 0, "Radiant", 1, 10000),
        _row(2, 128, "Dire", 0, 0),
        _row(2, 128, "Dire", 1, 0),
    ]
    frame = add_replay_rewards(pd.DataFrame(rows))
    frame["label"] = "farm"
    weighted, stats = add_advantage_weights(frame, [1])
    assert stats["farm"]["median"] < 10
    assert weighted.loc[6, "sample_weight"] == 4.0
    assert weighted.loc[7, "sample_weight"] == 0.0
