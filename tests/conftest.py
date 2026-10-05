import numpy as np
import pandas as pd
import pytest

from fm26scout.predict import trust
from fm26scout.schema import TARGETS
from fm26scout.settings import SAMPLE


@pytest.fixture(scope="session")
def folder(tmp_path_factory):
    """2,000 sample players with fake predictions: truth + noise, so trust is computable."""
    out = tmp_path_factory.mktemp("save")
    players = pd.read_parquet(SAMPLE).sample(2000, random_state=0).reset_index(drop=True)
    players.to_parquet(out / "players.parquet")
    rng = np.random.default_rng(0)
    db = players[["player_id"]].copy()
    for name, target in TARGETS.items():
        truth = players[name].fillna(players[name].median()).to_numpy(float)
        noise = 0.05 if name != "injury_proneness" else 5.0  # one target the model can't do
        pred = truth * np.exp(rng.normal(0, noise, len(truth))) if target.log else \
            truth + rng.normal(0, noise * (target.high - target.low), len(truth))
        spread = np.abs(pred - truth) + 0.1 * np.abs(pred)
        db[f"pred_{name}"], db[f"lo_{name}"], db[f"hi_{name}"] = pred, pred - spread, pred + spread
    db.to_parquet(out / "predictions.parquet")
    trust(db, players).to_csv(out / "trust.csv", index=False)
    return out


@pytest.fixture(autouse=True)
def no_prior_labs(monkeypatch):
    """No network in tests: usage and cost quotes are unknown unless a test fakes them."""
    from fm26scout import quota

    def offline(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(quota, "_fetch_usage", offline)
    monkeypatch.setattr(quota, "_quote", offline)
    monkeypatch.setattr(quota, "_usage", (-quota.USAGE_TTL, None))
    monkeypatch.setattr(quota, "_thinking", [])
