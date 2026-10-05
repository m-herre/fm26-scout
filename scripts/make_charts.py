"""README charts from the committed reports: docs/img/benchmark.svg and docs/img/chances.svg.

    python scripts/make_charts.py

Plain SVG with light and dark colours (prefers-color-scheme), so GitHub shows them in both themes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
STYLE = """<style>
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .ink2{fill:#52514e} .muted{fill:#898781}
  .hero{fill:#2a78d6} .rest{fill:#b4b2a9} .grid{stroke:#e1e0d9} .axis{stroke:#c3c2b7}
  .ref{stroke:#c3c2b7} .line{stroke:#2a78d6} .ring{stroke:#fcfcfb}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .ink{fill:#ffffff} .ink2{fill:#c3c2b7} .hero{fill:#3987e5}
    .rest{fill:#4a4945} .grid{stroke:#2c2c2a} .axis{stroke:#383835} .ref{stroke:#5c5b56}
    .line{stroke:#3987e5} .ring{stroke:#1a1a19}
  }
  text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
</style>"""
NAMES = {"tabpfn": "TabPFN-3.5", "catboost": "CatBoost", "random_forest": "Random forest",
         "ridge": "Ridge"}
PANELS = [("potential_ability", "Potential ability", "{:.1f}"),
          ("current_ability", "Current ability", "{:.2f}"),
          ("market_value", "Market value (log)", "{:.2f}")]


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{title}">\n'
            f"<title>{title}</title>\n{STYLE}\n"
            f'<rect class="bg" width="{width}" height="{height}" rx="8"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def benchmark() -> str:
    """Small multiples: held-out error per model for three targets, TabPFN-3.5 highlighted."""
    m = pd.read_csv(ROOT / "reports" / "sample" / "metrics.csv")
    width, panel_w, top, bar_h, gap = 960, 300, 78, 22, 10
    body = ['<text class="ink" x="24" y="34" font-size="18" font-weight="600">'
            "Error on 10,041 players no model saw (lower is better)</text>",
            '<text class="ink2" x="24" y="56" font-size="13">Same 10,000 training players for '
            "every model · no tuning for TabPFN-3.5</text>"]
    for p, (target, label, fmt) in enumerate(PANELS):
        x0 = 24 + p * (panel_w + 12)
        rows = m[(m["target"] == target) & m["model"].isin(NAMES)].set_index("model")["mae"]
        rows = rows.reindex(list(NAMES))
        scale = (panel_w - 150) / rows.max()
        body.append(f'<text class="ink" x="{x0}" y="{top}" font-size="14" font-weight="600">'
                    f"{label}</text>")
        for i, (model, mae) in enumerate(rows.items()):
            y = top + 14 + i * (bar_h + gap)
            w = mae * scale
            cls = "hero" if model == "tabpfn" else "rest"
            weight = ' font-weight="600"' if model == "tabpfn" else ""
            body += [f'<text class="ink2" x="{x0}" y="{y + 15}" font-size="12"{weight}>'
                     f"{NAMES[model]}</text>",
                     f'<path class="{cls}" d="M{x0 + 96},{y} h{w - 4} a4,4 0 0 1 4,4 '
                     f'v{bar_h - 8} a4,4 0 0 1 -4,4 h{-(w - 4)} z"/>',
                     f'<text class="ink" x="{x0 + 100 + w}" y="{y + 15}" font-size="12"'
                     f'{weight}>{fmt.format(mae)}</text>']
        base = top + 14
        body.append(f'<line class="axis" x1="{x0 + 96}" y1="{base - 4}" x2="{x0 + 96}" '
                    f'y2="{base + 4 * (bar_h + gap) - gap + 4}" stroke-width="1"/>')
    return _svg(width, top + 14 + 4 * (bar_h + gap) + 24, body,
                "Held-out error per model: TabPFN-3.5 lowest on potential ability, current "
                "ability and market value")


def chances() -> str:
    """Quoted vs real chance of PA 160+, against the diagonal."""
    text = (ROOT / "reports" / "sample" / "trust.md").read_text()
    section = text.split("## Chance of PA 160+")[1].split("##")[0]
    auc = re.search(r"AUC ([0-9.]+)", section).group(1)
    pts = [(int(q), int(r), int(n.replace(",", ""))) for n, q, r in
           re.findall(r"\| ([0-9,]+) \| (\d+)% \| (\d+)% \|", section)]
    pts = [p for p in pts if p[0] > 0]  # the 0% bin is everyone with no chance
    size, pad_l, pad_t = 420, 64, 84
    plot = size - pad_l - 28

    def xy(q, r):
        return pad_l + q / 100 * plot, pad_t + plot - r / 100 * plot

    body = ['<text class="ink" x="24" y="34" font-size="18" font-weight="600">'
            "“X% chance of PA 160+” means X%</text>",
            f'<text class="ink2" x="24" y="56" font-size="13">All 50,202 players, cross-fitted · '
            f"ranking AUC {auc}</text>"]
    for v in (0, 25, 50, 75, 100):
        x, y = xy(v, v)
        body += [f'<line class="grid" x1="{pad_l}" y1="{y}" x2="{pad_l + plot}" y2="{y}"/>',
                 f'<text class="muted" x="{pad_l - 8}" y="{y + 4}" font-size="11" '
                 f'text-anchor="end">{v}%</text>',
                 f'<text class="muted" x="{x}" y="{pad_t + plot + 18}" font-size="11" '
                 f'text-anchor="middle">{v}%</text>']
    x0, y0 = xy(0, 0)
    x1, y1 = xy(100, 100)
    body += [f'<line class="ref" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke-width="1.5" '
             'stroke-dasharray="4 4"/>',
             '<text class="muted" x="{:.1f}" y="{:.1f}" font-size="11">perfect</text>'.format(
                 *xy(89, 79)),
             '<polyline class="line" fill="none" stroke-width="2" points="'
             + " ".join("{:.1f},{:.1f}".format(*xy(q, r)) for q, r, _ in pts) + '"/>']
    for q, r, n in pts:
        x, y = xy(q, r)
        left = q > 80  # the top point is labelled on its left, inside the plot
        body += [f'<circle class="hero ring" cx="{x:.1f}" cy="{y:.1f}" r="5" stroke-width="2"/>',
                 f'<text class="ink" x="{x - 9 if left else x + 9:.1f}" y="{y + 4:.1f}" '
                 f'font-size="11" text-anchor="{"end" if left else "start"}">'
                 f"{r}% <tspan class=\"muted\">({n:,})</tspan></text>"]
    body += [f'<text class="ink2" x="{pad_l + plot / 2}" y="{pad_t + plot + 40}" font-size="12" '
             'text-anchor="middle">quoted chance</text>',
             f'<text class="ink2" font-size="12" text-anchor="middle" '
             f'transform="translate(18,{pad_t + plot / 2}) rotate(-90)">really reached 160+</text>']
    return _svg(size, pad_t + plot + 56, body, f"Quoted vs real chance of PA 160+, AUC {auc}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark.svg").write_text(benchmark())
    (OUT / "chances.svg").write_text(chances())
    print("wrote", OUT / "benchmark.svg", OUT / "chances.svg")
