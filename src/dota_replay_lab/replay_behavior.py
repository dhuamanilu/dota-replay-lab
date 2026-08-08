"""Build causal second-level behavior examples from parsed replay trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


ACTION_LABELS = ("move", "attack", "cast")
PREVIOUS_ORDER_COLUMNS = tuple(f"previous_{label}" for label in ACTION_LABELS)
NUMERIC_FEATURES = (
    "time_minutes",
    "x",
    "y",
    "alive",
    "level",
    "gold",
    "xp",
    "networth",
    "lh",
    "denies",
    "kills",
    "deaths",
    "assists",
    "movement_distance",
    *PREVIOUS_ORDER_COLUMNS,
    "nearest_ally_distance",
    "nearest_enemy_distance",
    "allies_nearby",
    "enemies_nearby",
)


def load_trajectory_corpus(paths: Iterable[Path]) -> object:
    """Load trajectory CSVs and attach match identity from each filename."""

    import pandas as pd

    frames = []
    for path in sorted(paths):
        match_id = int(path.name.split(".", 1)[0])
        frames.append(pd.read_csv(path).assign(match_id=match_id))
    if not frames:
        raise ValueError("No replay trajectory CSV files were found")
    return pd.concat(frames, ignore_index=True)


def add_spatial_context(frame: object, nearby_distance: float = 20.0) -> object:
    """Add team-relative distances using only simultaneous visible positions."""

    import numpy as np

    result = frame.reset_index(drop=True).copy()
    positions_all = result.loc[:, ["x", "y"]].to_numpy(dtype=np.float32)
    teams_all = result["team"].to_numpy()
    nearest_ally = np.zeros(len(result), dtype=np.float32)
    nearest_enemy = np.zeros(len(result), dtype=np.float32)
    allies_nearby = np.zeros(len(result), dtype=np.int8)
    enemies_nearby = np.zeros(len(result), dtype=np.int8)
    for indices in result.groupby(["match_id", "time"], sort=False).indices.values():
        positions = positions_all[indices]
        teams = teams_all[indices]
        distances = np.sqrt(((positions[:, None] - positions[None, :]) ** 2).sum(axis=2))
        same_team = teams[:, None] == teams[None, :]
        np.fill_diagonal(same_team, False)
        enemy = ~same_team
        np.fill_diagonal(enemy, False)
        ally_distances = np.where(same_team, distances, np.inf)
        enemy_distances = np.where(enemy, distances, np.inf)
        nearest_ally[indices] = ally_distances.min(axis=1)
        nearest_enemy[indices] = enemy_distances.min(axis=1)
        allies_nearby[indices] = (same_team & (distances <= nearby_distance)).sum(axis=1)
        enemies_nearby[indices] = (enemy & (distances <= nearby_distance)).sum(axis=1)
    result["nearest_ally_distance"] = nearest_ally
    result["nearest_enemy_distance"] = nearest_enemy
    result["allies_nearby"] = allies_nearby
    result["enemies_nearby"] = enemies_nearby
    return result


def build_behavior_frame(frame: object, nearby_distance: float = 20.0) -> object:
    """Create causal features and multi-label actions for behavior cloning."""

    required = {
        "match_id",
        "time",
        "slot",
        "hero_id",
        "team",
        "x",
        "y",
        "alive",
        "level",
        "gold",
        "xp",
        "networth",
        "lh",
        "denies",
        "kills",
        "deaths",
        "assists",
        "movement_distance",
        *(f"{label}_orders" for label in ACTION_LABELS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trajectory frame is missing columns: {', '.join(missing)}")
    result = frame.sort_values(["match_id", "slot", "time"]).copy()
    result["time_minutes"] = result["time"] / 60.0
    groups = result.groupby(["match_id", "slot"], sort=False)
    for label in ACTION_LABELS:
        result[f"label_{label}"] = (result[f"{label}_orders"] > 0).astype("int8")
        result[f"previous_{label}"] = (
            groups[f"label_{label}"].shift(1).fillna(0).astype("int8")
        )
    result = add_spatial_context(result, nearby_distance=nearby_distance)
    return result.sort_values(["match_id", "time", "slot"]).reset_index(drop=True)
