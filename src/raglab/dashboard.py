"""Metrics dashboard: a static, self-contained HTML monitoring page over
the eval store. Health verdicts against the CI thresholds first, history
second, detail last. No external assets — viewable from any browser."""

from pathlib import Path

import psycopg

from raglab import config
from raglab.eval_retrieval import THRESHOLDS

OUT_PATH = config.REPO_ROOT / "data" / "eval" / "dashboard.html"

_STYLE = """
:root { --bg:#FAFAF7; --surface:#FFF; --ink:#1F2A33; --muted:#5C6B76;
  --line:#DDE2E0; --teal:#17707E; --indigo:#4956A8; --amber:#A66B1F;
  --good:#3E7C4F; --bad:#B4453A; --good-soft:#E6F0E8; --bad-soft:#F7E4E1;
  --chip:#EFF2F0; }
@media (prefers-color-scheme: dark) { :root { --bg:#131A20; --surface:#1B242C;
  --ink:#E5E9EA; --muted:#93A1AB; --line:#2C3842; --teal:#4FB3C1;
  --indigo:#98A5E8; --amber:#D9A05B; --good:#7CBF8C; --bad:#E08A7E;
  --good-soft:#1E3326; --bad-soft:#3A211D; --chip:#242F38; } }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); margin: 0;
  font: 15px/1.45 -apple-system, "Segoe UI", system-ui, sans-serif;
  padding: 2.2rem 1.25rem 3.5rem; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0; }
.sub { color: var(--muted); font-size: .85rem; margin: .25rem 0 0; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .8rem; }
.mono { font-family: ui-monospace, "SF Mono", Consolas, monospace; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .6rem; }
.card { background: var(--surface); border: 1px solid var(--line);
  border-radius: 9px; padding: .7rem .85rem; }
.card .label { font-size: .72rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .06em; }
.card .value { font-size: 1.55rem; font-weight: 650; margin: .15rem 0;
  font-variant-numeric: tabular-nums; }
.card .rule { font-size: .72rem; color: var(--muted); }
.card.pass { border-left: 4px solid var(--good); }
.card.fail { border-left: 4px solid var(--bad); background: var(--bad-soft); }
.card.info { border-left: 4px solid var(--indigo); }
.chart-box { background: var(--surface); border: 1px solid var(--line);
  border-radius: 9px; padding: 1rem; overflow-x: auto; }
.chart-box .legend { font-size: .76rem; color: var(--muted); margin-bottom: .4rem; }
.legend .k { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin: 0 .3rem 0 1rem; vertical-align: -1px; }
table { border-collapse: collapse; width: 100%; background: var(--surface);
  border: 1px solid var(--line); border-radius: 9px; overflow: hidden;
  font-size: .84rem; }
th, td { padding: .45rem .7rem; text-align: right;
  font-variant-numeric: tabular-nums; border-top: 1px solid var(--line); }
th { background: var(--chip); color: var(--muted); font-size: .72rem;
  text-transform: uppercase; letter-spacing: .05em; border-top: 0; }
td:first-child, th:first-child { text-align: left; }
.bar { position: relative; background: var(--chip); border-radius: 4px;
  height: 10px; min-width: 90px; }
.bar i { position: absolute; inset: 0 auto 0 0; border-radius: 4px;
  background: var(--teal); }
.bar.low i { background: var(--bad); }
td.num { width: 4.2rem; }
details { margin-top: .8rem; }
summary { cursor: pointer; color: var(--muted); font-size: .82rem; }
.note { color: var(--muted); font-size: .78rem; margin-top: .4rem; }
"""


def _pct_bar(value: float, low: float = 0.5) -> str:
    cls = "bar low" if value < low else "bar"
    return (f'<div class="{cls}" title="{value:.3f}">'
            f'<i style="width:{max(2, value * 100):.0f}%"></i></div>')


def _card(label: str, value, rule: str, ok: bool | None) -> str:
    cls = "info" if ok is None else ("pass" if ok else "fail")
    shown = f"{value:.3f}" if isinstance(value, float) else value
    return (f'<div class="card {cls}"><div class="label">{label}</div>'
            f'<div class="value">{shown}</div><div class="rule">{rule}</div></div>')


def _trend_svg(points: list[tuple], threshold: float) -> str:
    """points: (run_id, config_label, hit5, coverage). Inline SVG, no libs."""
    if len(points) < 2:
        return "<p class='note'>Not enough retrieval runs for a trend yet.</p>"
    w, h, pad = 900, 220, 34
    lo, hi = 0.5, 1.0
    xs = [pad + i * (w - 2 * pad) / (len(points) - 1) for i in range(len(points))]

    def y(v):
        v = max(lo, min(hi, v))
        return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)

    def line(vals, color):
        pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals) if v is not None)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    def dots(vals, color):
        out = []
        for (rid, label, *_), x, v in zip(points, xs, vals):
            if v is None:
                continue
            out.append(
                f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="3.5" fill="{color}">'
                f'<title>run {rid} ({label}): {v:.3f}</title></circle>'
            )
        return "".join(out)

    grid = "".join(
        f'<line x1="{pad}" y1="{y(g):.1f}" x2="{w - pad}" y2="{y(g):.1f}" '
        f'stroke="var(--line)" stroke-width="1"/>'
        f'<text x="{pad - 6}" y="{y(g) + 4:.1f}" text-anchor="end" '
        f'font-size="11" fill="var(--muted)">{g:.1f}</text>'
        for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )
    thr = (f'<line x1="{pad}" y1="{y(threshold):.1f}" x2="{w - pad}" '
           f'y2="{y(threshold):.1f}" stroke="var(--amber)" stroke-width="1.5" '
           f'stroke-dasharray="6 4"><title>CI gate {threshold}</title></line>')
    xlabels = "".join(
        f'<text x="{x:.1f}" y="{h - pad + 16}" text-anchor="middle" font-size="10" '
        f'fill="var(--muted)">{rid}</text>'
        for (rid, *_), x in zip(points, xs)
    )
    hit = [p[2] for p in points]
    cov = [p[3] for p in points]
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'style="min-width:640px;width:100%">{grid}{thr}'
            f'{line(cov, "var(--indigo)")}{dots(cov, "var(--indigo)")}'
            f'{line(hit, "var(--teal)")}{dots(hit, "var(--teal)")}{xlabels}</svg>')


def render(conn: psycopg.Connection, out_path: Path = OUT_PATH) -> Path:
    latest_run = conn.execute(
        "SELECT id, config_label, git_sha, to_char(started_at, 'YYYY-MM-DD HH24:MI') "
        "FROM eval_runs WHERE kind = 'retrieval' AND config_label NOT LIKE '%%SABOTAGE%%' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()

    overall, by_cat = {}, {}
    if latest_run:
        for cat, metric, val in conn.execute(
            "SELECT category, metric, round(avg(value), 3) FROM eval_scores "
            "WHERE run_id = %s GROUP BY 1, 2", (latest_run[0],)
        ).fetchall():
            by_cat.setdefault(cat, {})[metric] = float(val)
        for metric, val in conn.execute(
            "SELECT metric, round(avg(value), 3) FROM eval_scores "
            "WHERE run_id = %s GROUP BY 1", (latest_run[0],)
        ).fetchall():
            overall[metric] = float(val)

    trend = conn.execute(
        "SELECT r.id, r.config_label, "
        "round(avg(s.value) FILTER (WHERE s.metric = 'hit@5'), 3), "
        "round(avg(s.value) FILTER (WHERE s.metric = 'source_coverage'), 3) "
        "FROM eval_runs r JOIN eval_scores s ON s.run_id = r.id "
        "WHERE r.kind = 'retrieval' AND r.config_label NOT LIKE '%%SABOTAGE%%' "
        "GROUP BY r.id, r.config_label ORDER BY r.id"
    ).fetchall()
    trend = [(r, l, float(h) if h is not None else None,
              float(c) if c is not None else None) for r, l, h, c in trend]

    deid = dict(conn.execute(
        "SELECT question_id, value FROM eval_scores WHERE run_id = "
        "(SELECT max(id) FROM eval_runs WHERE kind = 'deid')"
    ).fetchall())

    gen = conn.execute(
        "SELECT generator, judge, metric, round(avg(value), 3), count(*) "
        "FROM eval_scores WHERE generator IS NOT NULL GROUP BY 1, 2, 3 ORDER BY 1, 3"
    ).fetchall()

    runs = conn.execute(
        "SELECT r.id, to_char(r.started_at, 'YYYY-MM-DD HH24:MI'), r.kind, "
        "r.config_label, r.git_sha, count(s.id) "
        "FROM eval_runs r LEFT JOIN eval_scores s ON s.run_id = r.id "
        "GROUP BY r.id ORDER BY r.id DESC LIMIT 30"
    ).fetchall()

    # ---- health strip: current values vs the CI thresholds -------------
    cards = []
    if overall:
        hit = overall.get("hit@5")
        cards.append(_card("hit@5", hit, f"gate ≥ {THRESHOLDS['hit@5']}",
                           hit is not None and hit >= THRESHOLDS["hit@5"]))
        gate = overall.get("gate_correct")
        cards.append(_card("scope gate", gate, "gate = 1.0",
                           gate is not None and gate >= 1.0))
        wrong = overall.get("wrong_abstention", 0.0)
        cards.append(_card("wrong abstention", wrong,
                           f"gate ≤ {THRESHOLDS['wrong_abstention_rate']}",
                           wrong <= THRESHOLDS["wrong_abstention_rate"]))
        for key, label in (("deny_abstained", "persona: deny blocked"),
                           ("allow_answered", "persona: allow served")):
            val = overall.get(key)
            if val is not None:
                cards.append(_card(label, val, "gate = 1.0", val >= 1.0))
    if deid:
        cards.append(_card("de-id recall", float(deid.get("overall_recall", 0)),
                           "measured vs manifest", None))
        cards.append(_card("PHI leakage", float(deid.get("leakage_rate", 0)),
                           "verbatim survivors", None))

    # ---- per-category table -------------------------------------------
    metrics_order = ["hit@5", "precision@5", "source_coverage", "gate_correct",
                     "deny_abstained", "allow_answered", "allow_hit"]
    cat_rows = []
    for cat in sorted(by_cat):
        cells = [f"<td>{cat}</td>"]
        for m in metrics_order:
            v = by_cat[cat].get(m)
            cells.append(
                "<td>–</td>" if v is None else
                f'<td class="num">{v:.3f}</td><td>{_pct_bar(v)}</td>'
                if m == "hit@5" else f"<td>{v:.3f}</td>"
            )
        cat_rows.append("<tr>" + "".join(cells) + "</tr>")

    gen_rows = "".join(
        f"<tr><td>{g}</td><td>{j}</td><td>{m}</td><td>{v:.3f}</td><td>{n}</td></tr>"
        for g, j, m, v, n in gen
    )
    run_rows = "".join(
        f'<tr><td>{i}</td><td>{ts}</td><td>{k}</td><td>{c}</td>'
        f'<td class="mono">{sha}</td><td>{n}</td></tr>'
        for i, ts, k, c, sha, n in runs
    )
    deid_types = {k.removeprefix("recall_"): float(v) for k, v in deid.items()
                  if k.startswith("recall_")}
    deid_rows = "".join(
        f"<tr><td>{t}</td><td class='num'>{v:.3f}</td><td>{_pct_bar(v)}</td></tr>"
        for t, v in sorted(deid_types.items(), key=lambda kv: -kv[1])
    )

    stamp = (f"latest retrieval run {latest_run[0]} · {latest_run[3]} · "
             f"<span class='mono'>{latest_run[2]}</span>" if latest_run else "no runs yet")
    html = f"""<meta charset="utf-8"><title>raglab — evaluation dashboard</title>
<style>{_STYLE}</style><div class="wrap">
<h1>raglab — evaluation dashboard</h1>
<p class="sub">Retrieval quality, refusal behavior, entitlement enforcement, and
PHI de-identification — measured continuously against a human-verified golden
set. {stamp}</p>

<h2>Health — current values vs CI gates</h2>
<div class="cards">{''.join(cards)}</div>

<h2>Retrieval quality over time</h2>
<div class="chart-box">
  <div class="legend"><span class="k" style="background:var(--teal)"></span>hit@5
  <span class="k" style="background:var(--indigo)"></span>source_coverage
  <span class="k" style="background:var(--amber)"></span>CI gate ({THRESHOLDS['hit@5']})
  &nbsp;·&nbsp; hover a point for the run's config</div>
  {_trend_svg(trend, THRESHOLDS['hit@5'])}
</div>

<h2>Latest run — by question category</h2>
<table><tr><th>category</th><th colspan="2">hit@5</th><th>precision@5</th>
<th>coverage</th><th>gate</th><th>deny</th><th>allow</th><th>allow_hit</th></tr>
{''.join(cat_rows)}</table>
<p class="note">unanswerable is scored on the gate; persona_negative on
deny/allow — dashes are metrics that don't apply to a category.</p>

<h2>PHI de-identification (latest measured run)</h2>
<table><tr><th>entity type</th><th colspan="2">detection recall</th></tr>
{deid_rows or '<tr><td colspan=3>no deid runs yet</td></tr>'}</table>

<h2>Generation — cross-family judged</h2>
<table><tr><th>generator</th><th>judge</th><th>metric</th><th>mean</th><th>n</th></tr>
{gen_rows or '<tr><td colspan=5>no generation runs yet</td></tr>'}</table>

<details><summary>All runs (ledger)</summary>
<table><tr><th>run</th><th>started</th><th>kind</th><th>config</th><th>sha</th>
<th>scores</th></tr>{run_rows}</table></details>
</div>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
