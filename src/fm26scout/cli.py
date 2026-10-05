"""fm26 extract | schema | player | benchmark | predict | trust | embed | search | card | web.

Everything is written inside the project folder; only the save to extract may come from
anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .schema import BY_NAME, COLUMNS
from .settings import ROOT, SAMPLE, read_env


def _inside(path: str | Path) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise SystemExit(f"{path} is outside the project folder {ROOT}")
    return resolved


def extract(args) -> None:
    from . import prepare

    prepare.extract(Path(args.save).expanduser().resolve(),
                    _inside(args.out) if args.out else None, args.allow_warnings, args.sample)


def schema(args) -> None:
    for visibility in ("visible", "hidden", "meta"):
        cols = [c for c in COLUMNS if c.visibility == visibility]
        print(f"\n{visibility.upper()} ({len(cols)})")
        for c in cols:
            print(f"  {c.name:<24} {c.kind:<5} {c.note}")


def player(args) -> None:
    players = pd.read_parquet(_inside(args.data))
    hit = players[players["player_id"].astype(str) == args.query]
    if hit.empty:
        hit = players[players["name"].str.contains(args.query, case=False, na=False)]
    if hit.empty:
        raise SystemExit(f"No player matches {args.query!r}")
    row = hit.iloc[0]
    print(f"{row['name']} · {row['age']:.0f} · {row['club']} · split={row['split']}"
          + (f"  ({len(hit) - 1} more matches)" if len(hit) > 1 else ""))
    for visibility, title in (("visible", "VISIBLE: model inputs"),
                              ("hidden", "HIDDEN: targets only, never inputs")):
        values = [f"{name}={_fmt(row[name])}" for name in players.columns
                  if BY_NAME[name].visibility == visibility
                  and not (name.startswith("trait_") and row[name] == 0)]
        print(f"\n{title} ({len(values)} shown)")
        for i in range(0, len(values), 6):
            print("  " + "  ".join(values[i : i + 6]))


def _fmt(value) -> str:
    if isinstance(value, float):
        return "–" if pd.isna(value) else f"{value:.4g}"
    return str(value)


def benchmark(args) -> None:
    from .benchmark import compare, markdown, metrics, more_rows, predict_all, slices
    from .models import MODELS
    from .schema import TARGETS

    data = _inside(args.data)
    players = pd.read_parquet(data)
    out = _inside(args.out or f"reports/{data.parent.name}")
    done = out / "predictions"  # one file per target: a stopped run resumes where it was
    done.mkdir(parents=True, exist_ok=True)
    targets = args.targets or list(TARGETS)
    for target in targets:
        if not (done / f"{target}.parquet").exists():
            part = predict_all(players, [target], args.models or MODELS, args.n_train,
                               log=lambda line: print(line, flush=True))
            part.to_parquet(done / f"{target}.parquet", index=False)
    predictions = pd.concat([pd.read_parquet(done / f"{t}.parquet") for t in targets])
    table, by_slice = metrics(predictions), slices(predictions, players)
    versus = compare(predictions)
    table.to_csv(out / "metrics.csv", index=False)
    by_slice.to_csv(out / "slices.csv", index=False)
    versus.to_csv(out / "versus.csv", index=False)
    meta = {"data": data.relative_to(ROOT), "train rows per target": f"{args.n_train:,}",
            "test players": f"{(players['split'] == 'test').sum():,}"}
    more = more_meta = None
    if args.more_rows:  # the rivals again on more rows; TabPFN keeps its own fit
        n, parts = {}, []
        for target in [t for t in args.more_targets if t in targets]:
            n[target] = min(args.more_rows, int(
                (players[target].notna() & (players["split"] == "train")).sum()))
            file = done / f"{target}_rows{args.more_rows}.parquet"
            if not file.exists():
                predict_all(players, [target], args.more_models, args.more_rows,
                            log=lambda line: print(line, flush=True)).to_parquet(file, index=False)
            parts.append(pd.read_parquet(file))
        more = more_rows(pd.concat(parts), predictions, n)
        more.to_csv(out / "more_rows.csv", index=False)
        more_meta = {"rows asked": f"{args.more_rows:,}", "models": ", ".join(args.more_models)}
    (out / "benchmark.md").write_text(markdown(table, by_slice, meta, versus, more, more_meta))
    print((out / "benchmark.md").read_text())


def predict(args) -> None:
    from . import prepare

    print(prepare.predict(_inside(args.data).parent, args.n_train,
                          log=lambda line: print(line, flush=True)))


def trust(args) -> None:
    from . import prepare

    print(prepare.retrust(_inside(args.data).parent))


def embed(args) -> None:
    from . import prepare

    prepare.embed(_inside(args.data).parent)


def search(args) -> None:
    from .scout import Scout

    data = _inside(args.data)
    scout = Scout(data.parent)
    report = scout.search(json.loads(Path(args.spec).read_text()))
    for step, count in report["funnel"]:
        print(f"  {count:>6,}  {step}")
    for i, p in enumerate(report["players"], 1):
        preds = "  ".join(f"{t}={v['estimate']:g}"
                          + (" [{:g}-{:g}]".format(*v["range_80"]) if "range_80" in v else "")
                          for t, v in p["predictions"].items()
                          if t in ("potential_ability", "current_ability"))
        value = "?" if p["value"] is None else \
            f"{p['value']:,.0f}{' est.' if p['value_estimated'] else ''}"
        print(f"{i:>2}. {p['name']} · {p['age']} · {p['club']} · {p['positions']} · {value}\n"
              f"    {preds}" + ("\n    why: " + ", ".join(f"{v:+.1f} {n}" for n, v in p["why"])
                                  if "why" in p else ""))
    print(f"report -> {scout.save(report, data.parent.name)}")


def card(args) -> None:
    from .scout import Scout

    scout = Scout(_inside(args.data).parent)
    print(json.dumps(scout.card(scout.row(args.query)), indent=2, default=str))


def web(args) -> None:
    from .web import serve

    serve(_inside(args.data).parent if args.data else None, port=args.port)


def main() -> None:
    for name, value in read_env().items():  # the project's .env; real environment wins
        os.environ.setdefault(name, value)
    parser = argparse.ArgumentParser(prog="fm26")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("extract", help="save -> data/<save>/players.parquet with the fixed split")
    p.add_argument("save")
    p.add_argument("--out")
    p.add_argument("--allow-warnings", action="store_true")
    p.add_argument("--sample", action="store_true", help="also write the anonymised sample")
    p.set_defaults(run=extract)
    sub.add_parser("schema", help="every column and its visibility").set_defaults(run=schema)
    p = sub.add_parser("player", help="what the models can and cannot see for one player")
    p.add_argument("query", help="player id or part of a name")
    p.add_argument("--data", default=SAMPLE)
    p.set_defaults(run=player)
    p = sub.add_parser("benchmark", help="every target x model on the held-out players")
    p.add_argument("--data", default=SAMPLE)
    p.add_argument("--targets", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--n-train", type=int, default=10_000)
    p.add_argument("--more-rows", type=int, help="also fit --more-models on this many rows")
    p.add_argument("--more-models", nargs="+", default=["catboost", "random_forest"])
    p.add_argument("--more-targets", nargs="+",
                   default=["potential_ability", "current_ability", "market_value"])
    p.add_argument("--out")
    p.set_defaults(run=benchmark)
    p = sub.add_parser("predict", help="cross-fitted predictions + ranges for every player")
    p.add_argument("--data", default=SAMPLE)
    p.add_argument("--n-train", type=int, default=10_000)
    p.set_defaults(run=predict)
    p = sub.add_parser("trust", help="the trust report again from the kept predictions, no fit")
    p.add_argument("--data", default=SAMPLE)
    p.set_defaults(run=trust)
    p = sub.add_parser("embed", help="similarity vectors (local TabPFN embedding + attributes)")
    p.add_argument("--data", default=SAMPLE)
    p.set_defaults(run=embed)
    p = sub.add_parser("search", help="run a search spec (JSON) and save the report")
    p.add_argument("spec")
    p.add_argument("--data", default=SAMPLE)
    p.set_defaults(run=search)
    p = sub.add_parser("card", help="everything the engine may say about one player")
    p.add_argument("query")
    p.add_argument("--data", default=SAMPLE)
    p.set_defaults(run=card)
    p = sub.add_parser("web", help="the scouting chat in your browser")
    p.add_argument("--data", help="a prepared players.parquet (default: the save last used)")
    p.add_argument("--port", type=int, default=8626)
    p.set_defaults(run=web)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, RuntimeError) as error:  # bad spec, no token, quota: message only
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
