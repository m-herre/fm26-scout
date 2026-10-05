"""The scouting engine: hard filters, a weighted ranking, saved reports. Deterministic, no LLM.

It loads visible columns and predictions only. The save's hidden values are dropped on load,
and only targets the trust report rates "reliable" or "rough" can be filtered or ranked on.

A search spec (plain JSON, so a person, a UI or later an LLM can write one):
  {"filters": {"age": [null, 21], "position": ["STC"], "value": [null, 8000000],
               "pred_potential_ability": [150, null], "reputation": ["National", "Continental"]},
   "rank": {"pred_potential_ability": 0.6, "pace": 0.2, "acceleration": 0.2},
   "chance": {"target": "potential_ability", "at_least": 160, "min": 0.4},
   "risk": 0.3, "like": 12345, "limit": 10, "explain": "potential_ability", "by": "expected"}
Filters: [min, max] for numbers (null = open), a list of allowed values otherwise; "position"
means Accomplished or Natural there. Rank: weights on columns, standardised among the
candidates as percentiles, so one extreme value can't dominate (missing = average).
Risk: penalty on the width of the ranked targets' 80% ranges.
Chance: P(target >= at_least) for every candidate (a "chance" column to rank on), and with
"min" a filter; read off TabPFN's full predictive distribution (db/<target>_cdf.npy) when
kept. By: "expected" (the estimates), "ceiling" (the top of each 80% range) or "safe" (the
bottom; already the cautious end, so no risk penalty on top of either), set by the page's Ranking
switch. Explain: why lines from football features only (not price, wage, contract).

A save prepared in the browser starts with potential and fair value only; any other usable
target is fitted for all players the first time a search or player page asks for it (ensure).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import prepare
from .analytics import explain, peer_z, similar
from .predict import FOLDS, chance
from .prepare import GOOD
from .schema import ATTRIBUTES, HIDDEN, ORDINAL, TARGETS
from .settings import ROOT
from .split import position_group

MONEY = {"market value", "wage", "contract"}  # true signal, but not a scouting reason
NOT_REASONS = MONEY | {"club", "reputation"}  # kept in the full explanation, not the why line
_WHY_LOCK = threading.Lock()  # one writer of the why files at a time


class Scout:
    def __init__(self, folder: Path):
        players = pd.read_parquet(folder / "players.parquet").drop(columns=HIDDEN)
        db = pd.read_parquet(folder / "predictions.parquet")
        self.trust = pd.read_csv(folder / "trust.csv").set_index("target")
        self.targets = [t for t in TARGETS if self.trust.loc[t, "verdict"] in GOOD]
        self.pending = [t for t in self.targets if f"pred_{t}" not in db]  # fitted on first use
        keep = [f"{p}_{t}" for t in self.targets if t not in self.pending
                for p in ("pred", "lo", "hi")]
        self.frame = _derived(players.merge(db[["player_id", *keep]], on="player_id"))
        self.fitting: dict | None = None  # {"target", "done", "total"} while ensure() runs
        self.frozen: str | None = None  # why no target may be fitted now (estimates rebuilt)
        self._lock = threading.Lock()
        self.model = prepare.model_of(folder)  # the TabPFN model behind these estimates
        self.vectors = (np.load(folder / "embedding.npy")
                        if (folder / "embedding.npy").exists() else None)
        self.folder = folder
        self._dists: dict = {}  # target -> (grid, CDF rows) or None, loaded on first use

    def freeze(self, reason: str) -> None:
        """No target may be fitted until `frozen` is cleared; refused while one is."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TabPFN is estimating for this save right now: switch models "
                               "when it is done")
        self.frozen = reason
        self._lock.release()

    def ensure(self, targets) -> None:
        """Fit every pending target in `targets` for all players (5 TabPFN fits each), once."""
        for target in [t for t in targets if t in self.pending]:
            with self._lock:
                if self.frozen:
                    raise RuntimeError(self.frozen)
                if target not in self.pending:
                    continue  # another request fitted it meanwhile
                self.fitting = {"target": target, "done": 0, "total": FOLDS}
                try:
                    prepare.predict(self.folder, log=lambda _: None, targets=[target],
                                    tier=self.model,
                                    progress=lambda done, total: self.fitting.update(
                                        done=done, total=total))
                finally:
                    self.fitting = None
                cols = [f"{p}_{target}" for p in ("pred", "lo", "hi")]
                db = pd.read_parquet(self.folder / "predictions.parquet",
                                     columns=["player_id", *cols]).set_index("player_id")
                new = db.loc[self.frame["player_id"], cols]
                frame = self.frame.assign(**{c: new[c].to_numpy() for c in cols})
                self._dists.pop(target, None)
                self.trust = pd.read_csv(self.folder / "trust.csv").set_index("target")
                # this save's own check may rate it differently from the demo save
                self.targets = [t for t in TARGETS if self.trust.loc[t, "verdict"] in GOOD]
                self.pending = [t for t in self.pending if t != target and t in self.targets]
                self.frame = _derived(frame)  # complete before readers can see it

    def distribution(self, target: str, rows=None):
        """(grid, CDF of `rows`) of a target's full TabPFN distribution; None for estimates
        fitted before it was kept (chance then reads the 10/50/90% points)."""
        if target not in self._dists:
            dist = prepare.distribution(prepare.db_dir(self.folder, self.model), target)
            self._dists[target] = dist if dist and len(dist[1]) == len(self.frame) else None
        dist = self._dists[target]
        return None if dist is None else (dist[0], dist[1][np.asarray(
            self.frame.index if rows is None else rows)])

    def needs(self, spec: dict) -> list[str]:
        """Targets a search spec reads."""
        columns = [*(spec.get("filters") or {}), *(spec.get("rank") or {})]
        names = [c.split("_", 1)[1] for c in columns if c.startswith(("pred_", "lo_", "hi_"))]
        if spec.get("chance"):
            names.append(spec["chance"]["target"])
        return [n for n in dict.fromkeys(names) if n in self.targets]

    def row(self, query: str | int) -> int:
        """Row number of a player id or (part of) a name."""
        f = self.frame
        hit = f.index[f["player_id"].astype(str) == str(query)]
        if not len(hit):
            hit = f.index[f["name"].str.contains(str(query), case=False, na=False, regex=False)]
        if not len(hit):
            raise KeyError(f"No player matches {query!r}")
        return int(hit[0])

    def _check(self, column: str, frame: pd.DataFrame | None = None) -> None:
        name = column.split("_", 1)[1] if column.startswith(("pred_", "lo_", "hi_")) else None
        if column in HIDDEN or (name in TARGETS and name not in self.targets):
            raise ValueError(f"{column} is not available: the model cannot predict it reliably")
        if column not in (self.frame if frame is None else frame):
            raise ValueError(f"Unknown column {column!r}")

    def search(self, spec: dict) -> dict:
        self.ensure(self.needs(spec))
        f = self.frame.copy()
        if spec.get("like") is not None:
            if self.vectors is None:
                raise ValueError("Similar players are not available for this save (needs local "
                                 "TabPFN: fm26 embed)")
            index = self.row(spec["like"])
            rows, match = similar(self.vectors, index, k=len(f) - 1)
            f["similarity"] = pd.Series(match, index=rows)
            f["closest"] = pd.Series(np.arange(1, len(rows) + 1), index=rows)  # 1 = most similar
            f = f.drop(index=index)
        funnel = [("all players", len(f))]
        for column, rule in (spec.get("filters") or {}).items():
            f = f[self._keep(f, column, rule)]
            funnel.append((_label(column, rule), len(f)))
        if spec.get("chance"):
            c = spec["chance"]
            self._check(f"pred_{c['target']}")
            f["chance"] = chance(*(f[f"{p}_{c['target']}"] for p in ("lo", "pred", "hi")),
                                 c["at_least"], self.distribution(c["target"], f.index))
            if c.get("min"):
                f = f[f["chance"] >= c["min"]]
                funnel.append((f"{c['min']:.0%}+ chance of {c['target'].replace('_', ' ')} "
                               f"{c['at_least']:g}+", len(f)))
        score = pd.Series(0.0, index=f.index)
        for column, weight in _ranked(spec).items():
            self._check(column, f)
            score += weight * _z(f[column])
        widths = [f"hi_{c[5:]}" for c in spec.get("rank") or {} if c.startswith("pred_")]
        if spec.get("risk") and widths and spec.get("by") in (None, "expected"):
            width = sum(_z(f[w] - f[f"lo_{w[3:]}"]) for w in widths) / len(widths)
            score -= spec["risk"] * width
        f = f.assign(score=score).sort_values("score", ascending=False)
        f = f.head(spec.get("limit", 10))
        report = {"spec": spec, "created": datetime.now().isoformat(timespec="seconds"),
                  "funnel": funnel, "players": [self.card(i, similar_players=0) for i in f.index]}
        for player, (_, row) in zip(report["players"], f.iterrows(), strict=True):
            player["score"] = round(float(row["score"]), 3)
            for extra in ("similarity", "closest", "chance"):
                if extra in f:
                    player[extra] = _num(row[extra])
        if spec.get("like") is not None:
            report["like"] = self.card(index, similar_players=0)
        if spec.get("explain") and len(f):
            effects = self.explain(list(f.index), spec["explain"])
            for player, row in zip(report["players"], f.index, strict=True):
                player["why"] = [e for e in effects[row] if e[0] not in NOT_REASONS][:4]
        return report

    def _keep(self, f: pd.DataFrame, column: str, rule) -> pd.Series:
        if column == "position":
            ranks = [f[f"pos_{p.lower()}"].cat.codes for p in rule]
            return pd.concat(ranks, axis=1).max(axis=1) >= 4  # Accomplished or Natural
        self._check(column)
        if isinstance(rule, list) and len(rule) == 2 and column not in ORDINAL \
                and all(v is None or isinstance(v, int | float) for v in rule):
            low, high = (-np.inf if rule[0] is None else rule[0]), (
                np.inf if rule[1] is None else rule[1])
            return f[column].between(low, high)
        return f[column].astype(str).isin([str(v) for v in rule])

    def card(self, row: int, similar_players: int = 5) -> dict:
        """Everything the engine may say about one player."""
        p = self.frame.loc[row]
        card = {
            "player_id": int(p["player_id"]), "name": p["name"], "age": int(p["age"]),
            "club": p["club"], "positions": p["natural_positions"],
            "value": _num(p.get("value")), "value_estimated": bool(p["value_estimated"]),
            "predictions": {t: self._prediction(p, t) for t in self.targets
                            if t not in self.pending},
        }
        for extra in ("value_gap", "pa_vs_peers"):
            if extra in p:
                card[extra] = _num(p[extra])
        if similar_players and self.vectors is not None:
            rows, match = similar(self.vectors, row, k=similar_players)
            f = self.frame
            card["similar"] = [{"player_id": int(f.loc[r, "player_id"]), "name": f.loc[r, "name"],
                                "age": int(f.loc[r, "age"]), "club": f.loc[r, "club"],
                                "positions": f.loc[r, "natural_positions"]} for r in rows]
        return card

    def details(self, row: int) -> dict:
        """The player page: card, similar players, visible attributes and this season."""
        p = self.frame.loc[row]
        card = self.card(row)
        for target, out in card["predictions"].items():
            dist = self.distribution(target, [row]) if "range_80" in out else None
            if dist is not None and not np.isnan(dist[1].astype(float)).any():
                out["distribution"] = _curve(*dist)
        season = {c[7:]: _num(p[c]) for c in self.frame.columns
                  if c.startswith("season_") and pd.notna(p[c])}
        return card | {
            "attributes": {a: int(p[a]) for a in ATTRIBUTES if pd.notna(p[a])},
            "reputation": str(p["reputation"]), "club_stars": str(p["club_stars"]),
            "height_cm": _num(p["height_cm"]), "feet": [str(p["left_foot"]), str(p["right_foot"])],
            "contract_years_left": _num(p["contract_years_left"]), "wage": _num(p["wage"]),
            "season": season, "injuries": int(p["injuries"]),
            "not_estimated_yet": self.pending,
        }

    def _prediction(self, p: pd.Series, target: str) -> dict:
        t = self.trust.loc[target]
        out = {"estimate": _num(p[f"pred_{target}"]), "verdict": t["verdict"]}
        if t["range_shown"]:
            out["range_80"] = [_num(p[f"lo_{target}"]), _num(p[f"hi_{target}"])]
        return out

    def explain(self, rows: list[int], target: str) -> dict[int, list]:
        """Why lines, kept per save in why/<model>_<target>.json (player id -> effects): only
        players not kept yet go to TabPFN, all of them in one call. Thinking estimates are
        explained by the default model (analytics.explain), so they share its file."""
        if target not in self.targets:
            raise ValueError(f"{target} is not predictable enough to explain")
        tier = "default" if self.model == "thinking" else self.model
        file = self.folder / "why" / f"{tier}_{target}.json"
        ids = {r: str(self.frame.loc[r, "player_id"]) for r in rows}
        kept = _read_json(file)
        missing = [r for r in rows if ids[r] not in kept]
        if missing:
            full = pd.read_parquet(self.folder / "players.parquet")  # labels, never shown
            effects = explain(full, missing, target, tier=self.model)
            new = {ids[r]: [[name.replace("_", " "), round(v, 2)] for name, v in e]
                   for r, e in effects.items()}
            with _WHY_LOCK:  # another request may have added players meanwhile
                kept = _read_json(file) | new
                file.parent.mkdir(exist_ok=True)
                tmp = file.with_suffix(".tmp")  # written whole and swapped in: never half a file
                tmp.write_text(json.dumps(kept, separators=(",", ":")))
                tmp.replace(file)
        return {r: kept[ids[r]] for r in rows}

    def save(self, report: dict, name: str) -> Path:
        out = ROOT / "reports" / name / "searches"
        out.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(json.dumps(report["spec"], sort_keys=True).encode()).hexdigest()[:8]
        path = out / f"{report['created'].replace(':', '')}-{digest}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        return path


def _read_json(file: Path) -> dict:
    return json.loads(file.read_text()) if file.exists() else {}


def _derived(f: pd.DataFrame) -> pd.DataFrame:
    """The columns worked out from the visible data and the estimates, on a copy."""
    f = f.assign(position_group=position_group(f),
                 value_estimated=f["market_value"].isna(),  # the save stores no price for him
                 free_agent=f["club"].isna())
    if "pred_market_value" in f:
        f["value"] = f["market_value"].fillna(f["pred_market_value"])
        # log(fair value / price): > 0 means the profile is worth more than he costs
        f["value_gap"] = np.log(f["pred_market_value"] / f["market_value"])
    if "pred_potential_ability" in f:
        f["pa_vs_peers"] = peer_z(f, f["pred_potential_ability"])
    return f


def _ranked(spec: dict) -> dict[str, float]:
    """The rank weights, with "by": "ceiling" (or "safe") ranking each estimate, and a chance,
    on the top (or bottom) of its 80% range instead."""
    side = {"ceiling": "hi_", "safe": "lo_"}.get(spec.get("by"))
    target = (spec.get("chance") or {}).get("target")
    out: dict[str, float] = {}
    for column, weight in (spec.get("rank") or {}).items():
        if side and column.startswith("pred_"):
            column = side + column[5:]
        elif side and column == "chance" and target:
            column = side + target
        out[column] = out.get(column, 0) + weight
    return out


def _label(column: str, rule) -> str:
    """A filter in plain words: 'age up to 21', 'position MC or DM', 'value 1.5M to 20M'."""
    flags = {"value_estimated": ("price unknown", "price known"),
             "free_agent": ("free agent", "has a club")}
    if column in flags and isinstance(rule, list) and len(rule) == 1:
        return flags[column][0 if str(rule[0]).lower() == "true" else 1]
    if column == "value_gap" and isinstance(rule, list) and rule[0] is not None:
        return f"worth {np.exp(rule[0]):.1f}× his price"
    name = column.replace("pred_", "estimated ").replace("_", " ")
    if column == "position" or not (isinstance(rule, list) and len(rule) == 2 and all(
            v is None or isinstance(v, int | float) for v in rule)):
        return f"{name} {' or '.join(map(str, rule))}"
    short = _money if column in ("value", "market_value", "wage") else (lambda v: f"{v:g}")
    low, high = rule
    if low is None:
        return f"{name} up to {short(high)}"
    return f"{name} {short(low)}+" if high is None else f"{name} {short(low)} to {short(high)}"


def _money(v: float) -> str:
    """Money as the page writes it (money() in index.html): 500k, 1.5M, 20M."""
    if v >= 1e6:
        return f"{v / 1e6:.{0 if v >= 1e7 else 1}f}M"
    return f"{round(v / 1e3)}k" if v >= 1e3 else f"{round(v)}"


def _curve(x: np.ndarray, cdf: np.ndarray) -> dict:
    """One player's CDF for the page: the grid points where it rises, and one either side."""
    c = np.asarray(cdf[0], dtype=float)
    rising = np.flatnonzero((c > 0.002) & (c < 0.998))
    if not len(rising):
        rising = np.array([int(np.searchsorted(c, 0.5))])
    keep = slice(max(rising[0] - 1, 0), rising[-1] + 2)
    return {"x": [float(f"{v:.4g}") for v in x[keep]],
            "cdf": [round(float(v), 3) for v in c[keep]]}


def _z(values: pd.Series) -> pd.Series:
    """Percentile among the candidates, centred: -0.5 (worst) to +0.5 (best), missing = 0."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.codes.replace(-1, np.nan)  # in-game labels rank by their order
    values = pd.to_numeric(values, errors="coerce").astype(float)
    return (values.rank(pct=True) - 0.5).fillna(0)


def _num(value):
    return None if value is None or pd.isna(value) else round(float(value), 2)
