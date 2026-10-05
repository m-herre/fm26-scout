"""Save -> ready folder, the same steps for `fm26 extract/predict/embed` and the web upload:

extract   save -> players.parquet + meta.json (fixed split)
predict   cross-fitted TabPFN estimates for every player and target -> predictions.parquet,
          trust.csv, reports/<save>/trust.md (one file per target in db/: a stopped run resumes;
          beside it <target>_cdf.npy, every player's full predictive CDF at <target>_grid.npy;
          db/players.sha1 names the players they were fitted for)
embed     similarity vectors -> embedding.npy (always local TabPFN)

Each TabPFN model (models.TIERS) keeps its fitted targets in its own folder, db/ for default
and db/<model>/ for the others; meta.json "model" says which one predictions.parquet and
trust.csv are built from. Switching a save back to a model it had before costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .settings import ROOT, SAMPLE, local_weights

READY = ("players.parquet", "predictions.parquet", "trust.csv")
FIRST = ("potential_ability", "market_value")  # fitted at upload; the rest when first needed
GOOD = ("reliable", "rough")
_writing = threading.Lock()  # one rebuild of predictions.parquet / trust.csv at a time


def db_dir(folder: Path, tier: str = "default") -> Path:
    return folder / "db" if tier == "default" else folder / "db" / tier


def distribution(done: Path, target: str):
    """(grid in the target's units, players x points CDF, memory-mapped) from db/, None for
    targets fitted before the full distribution was kept."""
    cdf, grid = done / f"{target}_cdf.npy", done / f"{target}_grid.npy"
    if not (cdf.exists() and grid.exists()):
        return None
    return np.load(grid), np.load(cdf, mmap_mode="r")


def model_of(folder: Path) -> str:
    meta = folder / "meta.json"
    return (json.loads(meta.read_text()) if meta.exists() else {}).get("model", "default")


def ready(folder: Path) -> bool:
    return all((folder / name).exists() for name in READY)


def extract(save: Path, out: Path | None = None, allow_warnings: bool = False,
            sample: bool = False, log=print) -> Path:
    from .extract import anonymise, canonical, read_save
    from .split import assign

    raw, meta = read_save(save, allow_warnings)
    players = assign(canonical(raw, meta["save_date"]))
    out = out or ROOT / "data" / save.stem
    out.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out / "players.parquet", index=False)
    meta |= {"players": len(players), "test_players": int((players["split"] == "test").sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"{len(players):,} players -> {out / 'players.parquet'}")
    if sample:
        SAMPLE.parent.mkdir(exist_ok=True)
        anonymise(players).to_parquet(SAMPLE, index=False)
        (SAMPLE.parent / "meta.json").write_text(json.dumps(meta | {"names": "made up"}, indent=2))
        log(f"anonymised sample -> {SAMPLE}")
    return out


def predict(folder: Path, n_train: int = 10_000, log=print, progress=None,
            targets=None, tier: str = "default") -> str:
    """Fit `targets` (default: all) with model `tier`, then rebuild predictions.parquet and
    trust.csv from every target that model has fitted so far. `progress(done, total)` counts
    folds over the targets fitted now."""
    from . import quota
    from .predict import FOLDS, grid, predict_target
    from .schema import TARGETS

    targets = list(targets or TARGETS)
    players = pd.read_parquet(folder / "players.parquet")
    _forget_other_players(folder, players)
    done = db_dir(folder, tier)
    done.mkdir(parents=True, exist_ok=True)
    total = len(targets) * FOLDS
    # refused up front if the token budget or the thinking hour can't hold it (quota.plan)
    quota.plan(players, [t for t in targets if not (done / f"{t}.parquet").exists()], tier,
               n_train)
    for i, target in enumerate(targets):
        if progress:
            progress(i * FOLDS, total)
        if not (done / f"{target}.parquet").exists():
            log(f"predicting {target} for {len(players):,} players")
            fold = (lambda k, i=i: progress(i * FOLDS + k + 1, total)) if progress else None
            frame, cdf = predict_target(players, target, n_train, on_fold=fold, tier=tier)
            x = grid(target)  # the CDF goes first: the target file marks the target finished
            np.save(done / f"{target}_grid.npy", np.expm1(x) if TARGETS[target].log else x)
            np.save(done / f"{target}_cdf.npy", cdf.astype(np.float16))
            frame.insert(0, "player_id", players["player_id"].to_numpy())
            frame.to_parquet(done / f"{target}.parquet", index=False)
    with _writing:
        return _assemble(folder, players, tier)


def _forget_other_players(folder: Path, players: pd.DataFrame) -> None:
    """Delete every estimate fitted for other players: a different save uploaded into the same
    folder (same name, same in-game date) must not reuse them."""
    stamp = folder / "db" / "players.sha1"
    sha = hashlib.sha1(players["player_id"].to_numpy().tobytes()).hexdigest()
    if stamp.exists():
        stale = stamp.read_text() != sha
    else:  # fitted before the stamp was kept: still good if they cover as many players
        stale = any(pq.read_metadata(f).num_rows != len(players)
                    for f in (folder / "db").rglob("*.parquet"))
    if stale:
        shutil.rmtree(folder / "db")
        shutil.rmtree(folder / "why", ignore_errors=True)  # kept why lines (Scout.explain)
        for name in ("predictions.parquet", "trust.csv", "embedding.npy"):
            (folder / name).unlink(missing_ok=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(sha)


def _fitted(path: Path, players: pd.DataFrame) -> pd.DataFrame:
    """One target's estimates in the row order of `players`."""
    part = pd.read_parquet(path)
    if "player_id" not in part:  # fitted before player ids were kept: same rows, same order
        return part.reset_index(drop=True)
    return part.set_index("player_id").reindex(players["player_id"]).reset_index(drop=True)


def _assemble(folder: Path, players: pd.DataFrame, tier: str = "default") -> str:
    """predictions.parquet + trust.csv from the targets in db/. A target not fitted here takes
    its row from the demo save's trust report (checked_on "demo save"): either the game's data
    can't predict it, or it is fitted the first time a search needs it (see Scout.ensure)."""
    from .schema import TARGETS

    done = db_dir(folder, tier)
    have = [t for t in TARGETS if (done / f"{t}.parquet").exists()]
    db = pd.concat([players[["player_id"]].reset_index(drop=True),
                    *(_fitted(done / f"{t}.parquet", players) for t in have)], axis=1)
    floats = db.columns.drop("player_id")
    db[floats] = db[floats].astype("float32")
    db.to_parquet(folder / "predictions.parquet", index=False)
    report = _trust_files(folder, players, db, have, done)
    if tier != model_of(folder):
        meta_file = folder / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        meta_file.write_text(json.dumps(meta | {"model": tier}, indent=2))
    return report


def retrust(folder: Path, tier: str = "default") -> str:
    """trust.csv and reports/<save>/trust.md again from predictions.parquet and the kept
    CDFs: no fit."""
    from .schema import TARGETS

    players = pd.read_parquet(folder / "players.parquet")
    db = pd.read_parquet(folder / "predictions.parquet")
    have = [t for t in TARGETS if f"pred_{t}" in db]
    return _trust_files(folder, players, db, have, db_dir(folder, tier))


def _trust_files(folder: Path, players: pd.DataFrame, db: pd.DataFrame, have: list[str],
                 done: Path) -> str:
    from .predict import chance_reliability, trust, trust_markdown
    from .schema import TARGETS

    own = table = trust(db, players)
    if len(have) < len(TARGETS):
        demo = reference_trust()
        borrowed = demo[~demo["target"].isin(have)].assign(checked_on="demo save")
        table = pd.concat([own.assign(checked_on="this save"), borrowed])
        table = table.set_index("target").loc[list(TARGETS)].reset_index()
    table.to_csv(folder / "trust.csv", index=False)
    out = ROOT / "reports" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    pa = distribution(done, "potential_ability")
    reliability = ({f"PA {t}+": chance_reliability(db, players, "potential_ability", t, pa)
                    for t in (150, 160)} if "potential_ability" in have else {})
    report = trust_markdown(own, reliability)  # this save's own checks only
    (out / "trust.md").write_text(report)
    return report


def remodel(folder: Path, tier: str, step, n_train: int = 10_000) -> None:
    """Rebuild a save's estimates with another model: the first-stage targets now (skipped if
    that model fitted them before); the rest again on first use."""
    first = [t for t in FIRST if t in worth_fitting()]
    predict(folder, n_train, log=lambda _: None, targets=first, tier=tier,
            progress=lambda done, total: step(f"TabPFN estimates: {done} of {total} fits",
                                              0.05 + 0.9 * done / total))
    step("Ready", 1.0)


def reference_trust() -> pd.DataFrame:
    """The demo save's trust report: which targets the visible data can predict at all."""
    return pd.read_csv(SAMPLE.parent / "trust.csv")


def worth_fitting() -> list[str]:
    """Targets a new save gets estimates for: reliable or rough on the demo save."""
    demo = reference_trust()
    return demo.loc[demo["verdict"].isin(GOOD), "target"].tolist()


def embed(folder: Path, log=print) -> None:
    from .analytics import embed as embed_players

    vectors = embed_players(pd.read_parquet(folder / "players.parquet"))
    np.save(folder / "embedding.npy", vectors)
    log(f"{vectors.shape[0]:,} players x {vectors.shape[1]} dims -> {folder / 'embedding.npy'}")


def from_upload(save: Path, step, n_train: int = 10_000, on_folder=None,
                tier: str = "default", log=print) -> Path:
    """The web upload: extract into data/<save>-<in-game date> (the same save at the same date
    resumes its finished targets), predict potential and fair value only (the other targets the
    demo save rates predictable are fitted on first use, the rest never), embed when the local
    TabPFN weights are here, then delete the uploaded file. `step(text, fraction)` reports
    progress. A failed upload leaves no folder behind unless it holds finished fits."""
    step("Reading the save", 0.0)
    tmp, folder = ROOT / "data" / ".extracting", None
    shutil.rmtree(tmp, ignore_errors=True)  # left over from a stopped upload
    try:
        extract(save, tmp, log=lambda _: None)
        meta = json.loads((tmp / "meta.json").read_text())
        folder = ROOT / "data" / f"{slug(save.stem)}-{meta['save_date']}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("players.parquet", "meta.json"):
            (tmp / name).replace(folder / name)
        if on_folder:
            on_folder(folder)
        save.unlink(missing_ok=True)
        predict(folder, n_train, log=lambda _: None, tier=tier,
                targets=[t for t in FIRST if t in worth_fitting()],
                progress=lambda done, total: step(f"TabPFN estimates: {done} of {total} fits",
                                                  0.05 + 0.85 * done / total))
    except BaseException:
        if folder and not ready(folder) and not any(folder.glob("db/**/*.parquet")):
            shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    meta |= {"source": save.name, "prepared": datetime.now().isoformat(timespec="seconds"),
             "model": tier}
    if local_weights():
        step("Similar players (TabPFN embeddings on this computer)", 0.92)
        try:
            embed(folder, log=lambda _: None)
        except Exception as error:  # similarity is optional: the save works without it
            log(f"similar players skipped: {error}")
            meta["similarity_error"] = str(error) or type(error).__name__
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    step("Ready", 1.0)
    return folder


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:60] or "save"
