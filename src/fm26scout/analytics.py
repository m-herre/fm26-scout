"""Scouting analytics on top of the predicted DB: similar players, peer comparison, Shapley
explanations.

Similarity space (validated on 10,000 players, 10 neighbours each):
  TabPFN embedding alone     same position group 52%, current-ability gap 4.3
  attribute profile alone    same position group 88%, current-ability gap 12.6
  50/50 mix (used here)      same position group 87%, current-ability gap 4.9
The embedding knows level, the attributes know role; the mix keeps both.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .models import TabPFN, numeric_labels
from .schema import (
    ATTRIBUTES,
    CATEGORICAL,
    ORDINAL,
    POSITIONS,
    SEASON_PER_90,
    TARGETS,
    feature_columns,
)
from .split import age_band, position_group

PROFILE = [*ATTRIBUTES, *(f"pos_{p}" for p in POSITIONS), "age"]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x**2).sum(axis=1)).mean()


def embed(players: pd.DataFrame, context: int = 3000, dims: int = 64,
          seed: int = 0) -> np.ndarray:
    """One vector per player: TabPFN embedding (fitted on current ability, local TabPFN) mixed
    50/50 with the standardised visible attribute profile."""
    features = feature_columns("current_ability")
    ctx = players.sample(min(context, len(players)), random_state=seed)
    tab = TabPFN(ctx[features], ctx["current_ability"], seed, local=True).embeddings(
        players[features])
    tab = PCA(dims, random_state=seed).fit_transform(tab)
    x = numeric_labels(players[PROFILE]).astype(float)
    profile = ((x - x.mean()) / x.std()).fillna(0).to_numpy()
    return np.hstack([_unit(tab), _unit(profile)]).astype(np.float16)  # small enough to ship


def similar(vectors: np.ndarray, index: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers of the k players closest to row `index`, and for every returned player a
    match score: the share of all players (in %) who are further away than he is."""
    distance = np.linalg.norm(vectors.astype(np.float32) - vectors[index].astype(np.float32),
                              axis=1)
    distance[index] = np.inf
    order = np.argsort(distance)[:k]
    return order, 100 * (1 - np.arange(1, len(order) + 1) / len(vectors))


def peer_z(players: pd.DataFrame, values: pd.Series) -> pd.Series:
    """How exceptional a value is among players of the same age band and position group."""
    peers = age_band(players) + "|" + position_group(players)
    grouped = values.groupby(peers)
    return (values - grouped.transform("median")) / grouped.transform("std")


GROUPS = {
    **{a: [a] for a in ATTRIBUTES},
    "age": ["age"],
    "reputation": ["reputation"],
    "club": ["club_id", "club_stars", "club_league_position", "club_nation_id"],
    "market value": ["market_value"],
    "wage": ["wage"],
    "contract": ["release_clause", "contract_years_left", "squad_status", "contract_kind"],
    "positions": [f"pos_{p}" for p in POSITIONS] + ["natural_positions"],
    "this season": ["season_apps", "season_minutes", "season_rating", "season_potm",
                    "season_pass_pct"] + [f"season_{s}_p90" for s in SEASON_PER_90],
    "injury history": ["injuries", "injuries_major", "injuries_in_match"],
    "traits": [c for c in feature_columns("potential_ability") if c.startswith("trait_")],
    "height": ["height_cm"],
    "feet": ["left_foot", "right_foot"],
    "nationality": ["nationalities", "nation_id"],
    "years at club": ["years_at_club"],
    "on loan": ["on_loan"],
    "home grown": ["home_grown_club", "home_grown_nation"],
}
# coalitions per player: the budget of the reference in scripts/bench_shap.py (at 2048 the
# why line already named the reference's top reason for 18 of 20 players)
SHAP_BUDGET = 8192


def _typical(peers: pd.DataFrame, column: str):
    values = peers[column].dropna()
    if values.empty:
        return np.nan
    if column in CATEGORICAL or column in ORDINAL:
        return values.mode().iloc[0]
    median = values.median()
    return round(median) if pd.api.types.is_integer_dtype(peers[column]) else median


class _Batch:
    """The model as the explainers see it. A first pass only records the rows they ask for,
    then one TabPFN call answers all of them, and a second pass reads the answers. Coalition
    sampling is seeded, so both passes ask for the same rows (a miss costs one more call)."""

    def __init__(self, predict):
        self.predict, self.answers, self.asked, self.recording = predict, {}, [], True

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=float)
        if self.recording:
            self.asked.extend(rows)
            return np.zeros(len(rows))
        self.ask(rows)
        return np.array([self.answers[r.tobytes()] for r in rows])

    def ask(self, rows) -> None:
        new = {r.tobytes(): r for r in rows if r.tobytes() not in self.answers}
        if new:
            values = self.predict(np.array(list(new.values())))
            self.answers.update(zip(new, map(float, values), strict=True))

    def run(self, work):
        work()
        self.ask(self.asked)
        self.recording = False
        return work()


def explain(players: pd.DataFrame, rows: list[int], target: str, context: int = 10_000,
            seed: int = 0, budget: int = SHAP_BUDGET,
            tier: str = "default") -> dict[int, list[tuple[str, float]]]:
    """Why each player's estimate differs from his peers' (same age band and position group):
    Shapley values from Prior Labs' TabPFN explainer (tabpfn-extensions, imputation with the
    peer-typical player as baseline), summed per feature group, so his effects add up to his
    estimate minus the peer estimate. One TabPFN call for all players' coalitions (the API
    charges per call). Only for shortlists, never the whole DB. `tier`: the save's TabPFN model;
    Thinking is explained by the default model, as its ranges are (a thinking fit for every
    explanation would cost minutes and ~10x the tokens)."""
    from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer

    features = feature_columns(target)
    others = players.drop(index=rows)
    known = others[others[target].notna()]
    ctx = known.sample(min(context, len(known)), random_state=seed)
    y = np.log1p(ctx[target]) if TARGETS[target].log else ctx[target]
    tier = "default" if tier == "thinking" else tier
    model = TabPFN(ctx[features], y, seed, tier=tier, cache=True)
    batch = _Batch(model.predict_encoded)
    dtypes = players[features].dtypes.to_dict()
    peers = age_band(players) + "|" + position_group(players)
    index = {c: i for i, c in enumerate(features)}
    group_of = {index[c]: g for g, columns in GROUPS.items() for c in columns if c in index}

    def shapley() -> dict[int, np.ndarray]:
        out = {}
        for row in rows:
            player = players.loc[[row], features]
            group = players[peers == peers[row]]
            typical = player.copy()
            for column in features:
                typical[column] = _typical(group, column)
            x, base = (model.encode(f.astype(dtypes))[0] for f in (player, typical))
            explainer = get_tabpfn_imputation_explainer(batch, base[None, :], index="SV",
                                                        max_order=1, imputer="baseline",
                                                        random_state=seed)
            out[row] = explainer.explain(x, budget=budget).get_n_order_values(1)
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = batch.run(shapley)
    out = {}
    for row, phi in values.items():
        effects = dict.fromkeys(group_of.values(), 0.0)
        for i, v in enumerate(phi):
            effects[group_of[i]] += float(v)
        out[row] = sorted(effects.items(), key=lambda e: -abs(e[1]))
    return out
