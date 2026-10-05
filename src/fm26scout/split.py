"""One fixed player-level train/test split, shared by every target and every model.

Stratified by age band, position group and PA quintile, so each test slice is represented.
Deterministic: the same save and seed always give the same split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import POSITIONS

AGE_BANDS = [0, 18, 21, 25, 29, 99]
AGE_LABELS = ["<=18", "19-21", "22-25", "26-29", "30+"]
POSITION_GROUPS = {"gk": "GK", "sw": "DEF", "dl": "DEF", "dc": "DEF", "dr": "DEF",
                   "wbl": "DEF", "wbr": "DEF", "dm": "MID", "ml": "MID", "mc": "MID", "mr": "MID",
                   "aml": "ATT", "amc": "ATT", "amr": "ATT", "stc": "ATT"}


def age_band(players: pd.DataFrame) -> pd.Series:
    return pd.cut(players["age"], AGE_BANDS, labels=AGE_LABELS).astype(str)


def position_group(players: pd.DataFrame) -> pd.Series:
    """The group of the position the player is most familiar with."""
    ranks = players[[f"pos_{p}" for p in POSITIONS]].apply(lambda c: c.cat.codes)
    best = ranks.to_numpy().argmax(axis=1)
    return pd.Series([POSITION_GROUPS[POSITIONS[i]] for i in best], index=players.index)


def assign(players: pd.DataFrame, test_size: float = 0.2, seed: int = 0) -> pd.DataFrame:
    players = players.sort_values("player_id").reset_index(drop=True)
    pa_bin = pd.qcut(players["potential_ability"].rank(method="first"), 5, labels=False)
    strata = age_band(players) + "|" + position_group(players) + "|" + pa_bin.astype(str)
    _, test = train_test_split(
        np.arange(len(players)), test_size=test_size, random_state=seed, stratify=strata
    )
    players["split"] = "train"
    players.loc[test, "split"] = "test"
    return players
