"""Causal next-second combat examples from detailed replay combat logs."""

from __future__ import annotations

from .replay_behavior import NUMERIC_FEATURES as BEHAVIOR_FEATURES
from .replay_behavior import build_behavior_frame


COMBAT_LABELS = ("engage", "threat")
COMBAT_FEATURES = (
    *BEHAVIOR_FEATURES,
    "team_id",
    "hero_id",
    "recent_damage_dealt",
    "recent_damage_received",
)
RUNTIME_COMBAT_FEATURES = tuple(
    feature
    for feature in COMBAT_FEATURES
    if feature
    not in {
        "xp",
        "networth",
        "recent_damage_dealt",
        "recent_damage_received",
    }
)


def build_combat_frame(frame: object, horizon: int = 5) -> object:
    """Predict any hero combat in ``t+1..t+horizon`` from state through t."""

    if horizon < 1:
        raise ValueError("Combat horizon must be positive")

    required = {"hero_damage_dealt", "hero_damage_received"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Combat trajectory is missing columns: {', '.join(missing)}")
    result = build_behavior_frame(frame)
    result["team_id"] = (result["team"] == "Dire").astype("int8")
    result["recent_damage_dealt"] = result["hero_damage_dealt"].clip(lower=0)
    result["recent_damage_received"] = result["hero_damage_received"].clip(lower=0)
    groups = result.groupby(["match_id", "slot"], sort=False)
    result["next_time"] = groups["time"].shift(-horizon)
    future_dealt = [groups["hero_damage_dealt"].shift(-offset).gt(0) for offset in range(1, horizon + 1)]
    future_received = [
        groups["hero_damage_received"].shift(-offset).gt(0)
        for offset in range(1, horizon + 1)
    ]
    result["label_engage"] = future_dealt[0]
    result["label_threat"] = future_received[0]
    for values in future_dealt[1:]:
        result["label_engage"] |= values
    for values in future_received[1:]:
        result["label_threat"] |= values
    result["label_engage"] = result["label_engage"].astype("int8")
    result["label_threat"] = result["label_threat"].astype("int8")
    result = result[result["next_time"] == result["time"] + horizon].copy()
    return result.reset_index(drop=True)
