"""Every model behind one call (`fit_predict`), plus a reusable fitted `TabPFN`.

TabPFN runs at Prior Labs (`tabpfn-client`, TABPFN_TOKEN from the environment or .env) or
locally with FM26_TABPFN=local (FM26_TABPFN_DEVICE cpu, cuda or mps, else the fastest found;
weights cached in data/tabpfn-weights).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from . import quota
from .schema import CATEGORICAL, ORDINAL
from .settings import WEIGHTS, device

MODELS = ("median", "ridge", "random_forest", "catboost", "tabpfn")
CHUNK = 1_000  # rows per local TabPFN prediction; more swaps a 16 GB Mac

# The model behind a save's estimates (the picker next to Send). Quoted cost per fit of
# 10,000 training + 10,000 predicted players, Prior Labs quota_v3 (2026-10-05):
#   fast      TabPFN-3.5-Fast (v3.5-fast): ~1.7x less server compute, slightly wider error
#             (2,000-player check: MAE 9.7 vs 9.35, coverage 80%)        10.0k tokens  ~1x
#   default   TabPFN-3.5 (v3.5_default): in the cloud this is TabPFN-3.5-Plus (Prior Labs'
#             text processing; no text columns here), locally the open base model  10.6k  1x
#   thinking  TabPFN-3.5-Thinking (medium effort, THINKING_S budget per fit). Regression in
#             thinking mode returns the mean only, so the 80% range comes from a default fit
#             centred on the thinking estimate                         ~100k tokens  ~9.5x
TIERS = ("fast", "default", "thinking")
# quantile levels behind each CDF, dense in the tails
DENSE = [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, *np.round(np.arange(0.03, 0.971, 0.02), 3),
         0.98, 0.985, 0.99, 0.995, 0.997, 0.999]
THINKING_S = 120  # seconds of thinking per fit: an upload is 10 fits, so ~20 minutes at most


def _codes(train: pd.DataFrame, test: pd.DataFrame, cats: list[str]):
    """Categories as integer codes learned on train; unseen or missing become NaN."""
    train, test = train.copy(), test.copy()
    for column in cats:
        levels = pd.Index(train[column].dropna().unique())
        for frame in (train, test):
            codes = levels.get_indexer(frame[column]).astype(float)
            frame[column] = np.where(codes < 0, np.nan, codes)
    return train.astype(float), test.astype(float)


def _ridge(train, y, test, cats, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    nums = [c for c in train.columns if c not in cats]
    model = make_pipeline(
        ColumnTransformer(
            [
                ("num", make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()), nums),
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30),
                 cats),
            ]
        ),
        RidgeCV(alphas=np.logspace(-2, 3, 12)),
    )
    as_text = {c: lambda f, c=c: f[c].astype(str) for c in cats}
    return model.fit(train.assign(**as_text), y).predict(test.assign(**as_text))


def _random_forest(train, y, test, cats, seed):
    from sklearn.ensemble import RandomForestRegressor

    train, test = _codes(train, test, cats)
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.33, n_jobs=-1, random_state=seed
    )
    return model.fit(train, y).predict(test)


def _catboost(train, y, test, cats, seed):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise RuntimeError("CatBoost is a benchmark extra: pip install -e '.[bench]'") from None

    as_text = {c: lambda f, c=c: f[c].astype(str) for c in cats}
    model = CatBoostRegressor(cat_features=cats, random_seed=seed, verbose=0, thread_count=-1,
                              allow_writing_files=False)
    return model.fit(train.assign(**as_text), y).predict(test.assign(**as_text))


def hosted() -> bool:
    """Prior Labs API unless FM26_TABPFN=local: running on this computer is opt-in."""
    return os.environ.get("FM26_TABPFN", "hosted") != "local"


def _regressor(categorical: list[int], seed: int, local: bool, tier: str = "default"):
    """An unfitted TabPFN-3.5 regressor: hosted, or on this computer's GPU."""
    if tier not in TIERS:
        raise ValueError(f"Unknown TabPFN model {tier!r}")
    if not local:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor

        if not os.environ.get("TABPFN_TOKEN"):
            raise RuntimeError("No TabPFN API token: add one in Settings (or TABPFN_TOKEN in "
                               ".env), or turn on Run TabPFN locally")
        tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
        options = {"model_path": "v3.5-fast_default" if tier == "fast" else "v3.5_default"}
        if tier == "thinking":
            options |= {"thinking_effort": "medium", "thinking_timeout_s": THINKING_S,
                        "thinking_metric": "mae"}
        return TabPFNRegressor(**options, categorical_features_indices=categorical,
                               random_state=seed)
    if tier == "thinking":
        raise RuntimeError("TabPFN-3.5 Thinking runs only in the Prior Labs cloud: turn off "
                           "Run TabPFN locally, or pick another model")
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(WEIGHTS))
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor.create_default_for_version(
        "v3.5-fast" if tier == "fast" else "v3.5", device=device(),
        categorical_features_indices=categorical, random_state=seed,
        ignore_pretraining_limits=True,
    )


class TabPFN:
    """TabPFN-3.5 fitted on visible features: point predictions, full predictive
    distributions, embeddings."""

    def __init__(self, train: pd.DataFrame, y, seed: int = 0, local: bool | None = None,
                 tier: str = "default", cache: bool = False):
        self.local = not hosted() if local is None else local
        self.columns = list(train.columns)
        cats = [c for c in self.columns if c in CATEGORICAL]
        self.levels = {c: pd.Index(train[c].dropna().unique()) for c in cats}
        index = [self.columns.index(c) for c in cats]
        # thinking: the estimate from the thinking model, ranges from a default fit
        self.model = _regressor(index, seed, self.local, "default" if tier == "thinking" else tier)
        self.thinker = _regressor(index, seed, self.local, tier) if tier == "thinking" else None
        if cache and self.local:  # many predictions against one fit (explanations)
            self.model.set_params(fit_mode="fit_with_cache")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for model in filter(None, (self.model, self.thinker)):
                if model is self.thinker:
                    quota.pace()  # Prior Labs: 10 thinking fits a minute, 30 an hour
                self._call(model.fit, self._numeric(train), np.asarray(y, dtype=float))

    def _call(self, fn, *args, **kwargs):
        """Hosted: through Prior Labs' rate limits and token budget (quota.call)."""
        return fn(*args, **kwargs) if self.local else quota.call(fn, *args, **kwargs)

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = numeric_labels(frame[self.columns])
        for column, levels in self.levels.items():
            codes = levels.get_indexer(frame[column]).astype(float)
            frame[column] = np.where(codes < 0, np.nan, codes)
        return frame.astype(float)

    def _chunks(self, frame: pd.DataFrame, call) -> list:
        """Hosted: one request for everything. Local: small chunks, memory is the limit."""
        size = CHUNK if self.local else max(len(frame), 1)
        rows = self._numeric(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [call(rows.iloc[i : i + size]) for i in range(0, len(rows), size)]

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows as the model sees them (labels -> ranks, categories -> codes)."""
        return self._numeric(frame).to_numpy()

    def predict_encoded(self, rows: np.ndarray) -> np.ndarray:
        """Point predictions for rows from `encode` (default model, chunked when local)."""
        size = CHUNK if self.local else max(len(rows), 1)
        frame = pd.DataFrame(rows, columns=self.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.concatenate([self._call(self.model.predict, frame.iloc[i : i + size])
                                   for i in range(0, len(frame), size)])

    def mean(self, frame: pd.DataFrame) -> np.ndarray:
        model = self.thinker or self.model
        return np.concatenate(self._chunks(frame, lambda rows: self._call(model.predict, rows)))

    def distribution(self, frame: pd.DataFrame, levels: list[float],
                     grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TabPFN's predictive distribution from one quantiles call at DENSE levels: quantiles
        at `levels` (levels x rows) and the CDF at `grid` (rows x grid points). Thinking: the
        default model's distribution, moved so its median sits on the thinking estimate (the
        trust report checks these ranges still hold)."""
        dense = sorted({*DENSE, *levels, 0.5})

        def call(rows):
            q = self._call(self.model.predict, rows, output_type="quantiles", quantiles=dense)
            q = np.sort(np.asarray(q), axis=0)
            if self.thinker is not None:  # its own call: a retry never repeats the first
                q = q + (self._call(self.thinker.predict, rows) - q[dense.index(0.5)])
            cdf = np.array([np.interp(grid, row, dense, left=0, right=1) for row in q.T])
            return q[[dense.index(v) for v in levels]], cdf.astype(np.float32)

        parts = self._chunks(frame, call)
        return np.concatenate([q for q, _ in parts], axis=1), np.concatenate([c for _, c in parts])

    def embeddings(self, frame: pd.DataFrame) -> np.ndarray:
        """rows x dims, averaged over the ensemble (local TabPFN only)."""
        parts = self._chunks(frame, lambda rows: self.model.get_embeddings(rows).mean(axis=0))
        return np.concatenate(parts)


def numeric_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """In-game labels (ordered categories) -> their rank; everything else unchanged."""
    ranks = {c: lambda f, c=c: f[c].cat.codes.replace(-1, np.nan).astype(float)
             for c in frame.columns if c in ORDINAL}
    return frame.assign(**ranks)


def fit_predict(model: str, train: pd.DataFrame, y: pd.Series, test: pd.DataFrame,
                seed: int = 0) -> np.ndarray:
    if model == "median":
        return np.full(len(test), float(np.median(y)))
    if model == "tabpfn":
        return TabPFN(train, y, seed).mean(test)
    cats = [c for c in train.columns if c in CATEGORICAL]
    train, test = numeric_labels(train), numeric_labels(test)
    fit = {"ridge": _ridge, "random_forest": _random_forest, "catboost": _catboost}[model]
    return np.asarray(fit(train, y.to_numpy(float), test, cats, seed), dtype=float)
