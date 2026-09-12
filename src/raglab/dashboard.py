"""Metrics dashboard: a static, self-contained HTML monitoring page over
the eval store. Health verdicts against the CI thresholds first, history
second, detail last. No external assets — viewable from any browser."""

from pathlib import Path

import psycopg

from raglab import config
from raglab import stats
from raglab.timing import BUDGET_P95_MS, percentile
from raglab.eval_retrieval import THRESHOLDS
from raglab import taxonomy

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
td.num { width: 5.4rem; }
.ci { font-size: .72rem; color: var(--muted, #777); white-space: nowrap; }
details { margin-top: .8rem; }
summary { cursor: pointer; color: var(--muted); font-size: .82rem; }
.note { color: var(--muted); font-size: .78rem; margin-top: .4rem; }
td.left, th.left { text-align: left; }
td.cat { text-align: left; font-weight: 600; white-space: nowrap; }
td.cat small { display: block; font-weight: 400; color: var(--muted); white-space: normal; }
td.ex { text-align: left; color: var(--muted); font-style: italic; }
tr.band td { background: var(--chip); color: var(--muted); font-size: .72rem;
  text-transform: uppercase; letter-spacing: .05em; text-align: left; }
.defects { list-style: none; padding: 0; margin: 0; }
.defects li { padding: .3rem 0; border-top: 1px solid var(--line); }
.defects li b { display: inline-block; min-width: 1.6rem; color: var(--amber); }
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
    # Annotate gate breaches — regressions the eval caught are the story.
    breaches = "".join(
        f'<line x1="{x:.1f}" y1="{y(v) - 8:.1f}" x2="{x:.1f}" y2="{y(v) - 26:.1f}" '
        f'stroke="var(--bad)" stroke-width="1.5"/>'
        f'<text x="{x:.1f}" y="{y(v) - 31:.1f}" text-anchor="middle" font-size="11" '
        f'fill="var(--bad)">run {rid}: below gate'
        f'<title>run {rid} ({label}): hit@5 {v:.3f} &lt; {threshold} — '
        f'caught by the eval, fixed in the following run</title></text>'
        for (rid, label, *_), x, v in zip(points, xs, hit)
        if v is not None and v < threshold
    )
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'style="min-width:640px;width:100%">{grid}{thr}'
            f'{line(cov, "var(--indigo)")}{dots(cov, "var(--indigo)")}'
            f'{line(hit, "var(--teal)")}{dots(hit, "var(--teal)")}'
            f'{breaches}{xlabels}</svg>')


def _item_verdicts(conn: psycopg.Connection) -> tuple[dict, dict, tuple | None]:
    """Latest verdict per golden item across FULL retrieval runs (never a
    partial or sabotage run). An item skipped in the latest run keeps the
    verdict of the last run that verified it, so a CI run without warehouse
    credentials neither hides nor fakes the warehouse-backed items.
    Returns (by_work, by_group, (oldest_run, newest_run))."""
    rows = conn.execute(
        "SELECT DISTINCT ON (s.question_id) s.question_id, s.category, s.value, "
        "s.detail->>'work_category', r.id "
        "FROM eval_scores s JOIN eval_runs r ON r.id = s.run_id "
        "WHERE s.metric = 'item_pass' AND r.kind = 'retrieval' "
        "AND r.config_label NOT LIKE '%%SABOTAGE%%' AND r.config_label NOT LIKE '%%-partial:%%' "
        "ORDER BY s.question_id, r.id DESC"
    ).fetchall()
    by_work: dict[int, list[int]] = {}
    by_group: dict[str, list[int]] = {}
    run_ids = []
    for _qid, group, value, work, run_id in rows:
        by_work.setdefault(int(work), []).append(int(float(value) >= 1.0))
        by_group.setdefault(group, []).append(int(float(value) >= 1.0))
        run_ids.append(run_id)
    span = (min(run_ids), max(run_ids)) if run_ids else None
    return by_work, by_group, span


def _pass_cells(verdicts: list[int] | None) -> str:
    if not verdicts:
        return '<td class="num">0</td><td class="num">–</td><td></td>'
    n, passed = len(verdicts), sum(verdicts)
    rate = passed / n
    return (f'<td class="num">{n}</td><td class="num">{rate * 100:.0f}%</td>'
            f'<td>{_pct_bar(rate, low=1.0)}</td>')


def _capability_html(conn: psycopg.Connection) -> str:
    by_work, by_group, span = _item_verdicts(conn)
    open_defects: dict[int, int] = {}
    for number, _text in taxonomy.DEFECTS:
        open_defects[number] = open_defects.get(number, 0) + 1

    work_rows = []
    for band, title in taxonomy.BANDS:
        work_rows.append(f'<tr class="band"><td colspan="7">{title}</td></tr>')
        for c in taxonomy.WORK_CATEGORIES.values():
            if c.band != band:
                continue
            label = "G" if c.number == taxonomy.GUARDRAILS else str(c.number)
            if band == "designed_out":
                cells = '<td class="num">–</td><td class="num">–</td><td></td>'
            else:
                cells = _pass_cells(by_work.get(c.number))
            work_rows.append(
                f'<tr><td class="num">{label}</td>'
                f'<td class="cat">{c.name}<small>{c.description}</small></td>'
                f'<td class="ex">“{c.example}”</td>{cells}'
                f'<td class="num">{open_defects.get(c.number, 0) or "–"}</td></tr>'
            )

    group_rows = []
    for band, title in taxonomy.GROUP_BANDS:
        group_rows.append(f'<tr class="band"><td colspan="5">{title}</td></tr>')
        for g in taxonomy.GROUPS.values():
            if g.band != band:
                continue
            group_rows.append(
                f'<tr><td class="cat">{g.name}<small>{g.description}</small></td>'
                f'<td class="left">{g.metric}</td>{_pass_cells(by_group.get(g.name))}</tr>'
            )

    defect_items = "".join(
        f"<li><b>{'G' if n == taxonomy.GUARDRAILS else n}</b>{text}</li>" for n, text in taxonomy.DEFECTS
    ) or "<li>none open</li>"
    blind = conn.execute(
        "SELECT r.id, to_char(r.started_at, 'YYYY-MM-DD'), r.config_label, "
        "sum(s.value) FILTER (WHERE s.metric = 'right_context'), "
        "sum(s.value) FILTER (WHERE s.metric = 'not_in_corpus'), "
        "sum(s.value) FILTER (WHERE s.metric = 'defect'), count(DISTINCT s.question_id) "
        "FROM eval_runs r JOIN eval_scores s ON s.run_id = r.id "
        "WHERE r.kind = 'blind' GROUP BY r.id ORDER BY r.id DESC LIMIT 1"
    ).fetchone()
    blind_html = (
        f"run {blind[0]} · {blind[1]} · {blind[2]}: {int(blind[6])} questions — "
        f"{int(blind[3] or 0)} right context · {int(blind[4] or 0)} not in corpus, said so · "
        f"{int(blind[5] or 0)} defects"
        if blind else "no blind-set runs recorded in the store yet"
    )
    verified = (f"verdicts from full runs {span[0]}–{span[1]}" if span and span[0] != span[1]
                else f"verdicts from full run {span[0]}" if span else "no full retrieval run yet")
    return (
        "\n<h2>Capability — by the work a question needs</h2>\n"
        '<table><tr><th>#</th><th class="left">category</th><th class="left">example question</th>\n'
        "<th>gated items</th><th>pass rate</th><th></th><th>open defects</th></tr>\n"
        f"{''.join(work_rows)}</table>\n"
        '<p class="note">Pass = every gate metric on the item is 1.0. A check skipped in a run '
        "(no warehouse credentials) never counts as a pass: the item keeps the verdict of the "
        f"last full run that verified it. {verified}. Descriptions are general on purpose so "
        "the rows mean the same thing as the corpus grows; only the payload is judged, never "
        "the answering model's prose.</p>\n"
        "\n<h2>Capability — by mechanism and source (the groups the gate is built on)</h2>\n"
        '<table><tr><th class="left">group</th><th class="left">metric</th><th>gated items</th>\n'
        "<th>pass rate</th><th></th></tr>\n"
        f"{''.join(group_rows)}</table>\n"
        "\n<h2>Open defects</h2>\n"
        f'<ul class="defects">{defect_items}</ul>\n'
        '<p class="note">Each defect is pinned to the category it blocks; a row\'s pass rate can be '
        "100% while a defect found outside the golden set stays open against it.</p>\n"
        "\n<h2>Real questions (blind set, first-run totals)</h2>\n"
        f'<p class="note">{blind_html}</p>\n'
    )


def render(conn: psycopg.Connection, out_path: Path = OUT_PATH) -> Path:
    latest_run = conn.execute(
        "SELECT id, config_label, git_sha, to_char(started_at, 'YYYY-MM-DD HH24:MI') "
        "FROM eval_runs WHERE kind = 'retrieval' AND config_label NOT LIKE '%%SABOTAGE%%' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()

    overall, by_cat, by_source, counts = {}, {}, {}, {}
    if latest_run:
        for cat, metric, val, n, hits in conn.execute(
            "SELECT category, metric, round(avg(value), 3), count(*), sum(value) "
            "FROM eval_scores WHERE run_id = %s GROUP BY 1, 2", (latest_run[0],)
        ).fetchall():
            by_cat.setdefault(cat, {})[metric] = float(val)
            counts[("cat", cat, metric)] = (int(n), int(round(float(hits))))
        for src, metric, val, n, hits in conn.execute(
            "SELECT coalesce(detail->>'source', 'none'), metric, round(avg(value), 3), "
            "count(*), sum(value) FROM eval_scores WHERE run_id = %s GROUP BY 1, 2",
            (latest_run[0],)
        ).fetchall():
            by_source.setdefault(src, {})[metric] = float(val)
            counts[("src", src, metric)] = (int(n), int(round(float(hits))))
        hit_n = conn.execute(
            "SELECT count(*), sum(value) FROM eval_scores WHERE run_id = %s AND metric = 'hit@5'",
            (latest_run[0],)
        ).fetchone()
        if hit_n and hit_n[0]:
            overall["hit@5_ci"] = stats.wilson(int(round(float(hit_n[1]))), int(hit_n[0]))
        for metric, val in conn.execute(
            "SELECT metric, round(avg(value), 3) FROM eval_scores "
            "WHERE run_id = %s GROUP BY 1", (latest_run[0],)
        ).fetchall():
            overall[metric] = float(val)

    latency = {}
    if latest_run:
        by_stage: dict = {}
        for metric, value in conn.execute(
            "SELECT metric, value FROM eval_scores WHERE run_id = %s AND metric LIKE 'latency_%%'",
            (latest_run[0],)
        ).fetchall():
            by_stage.setdefault(metric.removeprefix("latency_"), []).append(float(value))
        for stage, vals in by_stage.items():  # same nearest-rank percentile as the receipt
            latency[stage] = (percentile(vals, 50), percentile(vals, 95))

    rls = conn.execute(
        "SELECT category, max(value) FILTER (WHERE metric = 'recall_mean'), "
        "max(value) FILTER (WHERE metric = 'recall_min'), "
        "max(value) FILTER (WHERE metric = 'underfilled'), max(detail->>'visible'), max(detail->>'total') "
        "FROM eval_scores WHERE run_id = (SELECT max(id) FROM eval_runs WHERE kind = 'recall_rls') "
        "GROUP BY category ORDER BY category"
    ).fetchall()

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
        "SELECT metric, value FROM eval_scores WHERE run_id = "
        "(SELECT max(id) FROM eval_runs WHERE kind = 'deid')"
    ).fetchall())

    # 'abstained' means opposite things on trap vs answerable questions —
    # split it into the two behaviors instead of averaging them together.
    gen = conn.execute(
        "SELECT generator, judge, "
        "CASE WHEN metric = 'abstained' AND category = 'unanswerable' "
        "     THEN 'traps held (must refuse)' "
        "     WHEN metric = 'abstained' THEN 'wrong refusals (must answer)' "
        "     ELSE metric END AS metric, "
        "round(avg(value), 3), count(*) "
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
        ci = overall.get("hit@5_ci")
        rule = f"gate ≥ {THRESHOLDS['hit@5']}" + (f" · 95% CI {ci[0]:.2f}–{ci[1]:.2f}" if ci else "")
        cards.append(_card("hit@5", hit, rule, hit is not None and hit >= THRESHOLDS["hit@5"]))
        gate = overall.get("gate_correct")
        cards.append(_card("scope gate", gate, "gate = 1.0",
                           gate is not None and gate >= 1.0))
        wrong = overall.get("wrong_abstention", 0.0)
        cards.append(_card("wrong abstention", wrong,
                           f"gate ≤ {THRESHOLDS['wrong_abstention_rate']}",
                           wrong <= THRESHOLDS["wrong_abstention_rate"]))
        for key, label in (("deny_clean", "persona: deny clean"),
                           ("allow_answered", "persona: allow served")):
            val = overall.get(key)
            if val is not None:
                cards.append(_card(label, val, "gate = 1.0", val >= 1.0))
    if latency.get("total"):
        p50, p95 = latency["total"]
        cards.append(_card("latency p95 (ms)", int(p95),
                           f"budget ≤ {BUDGET_P95_MS} ms · p50 {p50:.0f} · not gated", None))
    if deid:
        cards.append(_card("de-id recall", float(deid.get("overall_recall", 0)),
                           "measured vs manifest", None))
        cards.append(_card("PHI leakage", float(deid.get("leakage_rate", 0)),
                           "verbatim survivors", None))

    # ---- per-category table -------------------------------------------
    metrics_order = ["hit@5", "precision@5", "source_coverage", "gate_correct",
                     "deny_clean", "allow_answered", "allow_hit", "scope_clean", "version_clean"]
    def _slice_rows(kind: str, table: dict) -> list[str]:
        rows_html = []
        for name in sorted(table):
            cells = [f"<td>{name}</td>"]
            for m in metrics_order:
                v = table[name].get(m)
                if v is None:
                    cells.append("<td>–</td>")
                    continue
                if m == "hit@5":
                    n, hits = counts.get((kind, name, m), (0, 0))
                    low, high = stats.wilson(hits, n) if n else (0.0, 1.0)
                    cells.append(
                        f'<td class="num">{v:.3f}<br><span class="ci">{low:.2f}–{high:.2f} (n={n})</span></td>'
                        f"<td>{_pct_bar(v)}</td>"
                    )
                else:
                    cells.append(f"<td>{v:.3f}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        return rows_html

    cat_rows = _slice_rows("cat", by_cat)
    source_rows = _slice_rows("src", by_source)

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

    capability = _capability_html(conn)
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
{capability}

<h2>Retrieval quality over time</h2>
<div class="chart-box">
  <div class="legend"><span class="k" style="background:var(--teal)"></span>hit@5
  <span class="k" style="background:var(--indigo)"></span>source_coverage
  <span class="k" style="background:var(--amber)"></span>CI gate ({THRESHOLDS['hit@5']})
  &nbsp;·&nbsp; hover a point for the run's config</div>
  {_trend_svg(trend, THRESHOLDS['hit@5'])}
</div>

<p class="note">Latency per stage over the golden set (ms, p50 / p95):
{' · '.join(f"{s} {p50:.0f} / {p95:.0f}" for s, (p50, p95) in sorted(latency.items())) or 'no latency data yet'}
— measured on the machine that ran the eval; the Phase 6 VM is the target.</p>

<p class="note">Vector recall under row-level security, per persona (latest measurement, recall@k vs exact scan as the same persona):
{' · '.join(f"{c} sees {v}/{t}: mean {float(m):.3f}, min {float(mn):.3f}, underfilled {int(float(u))}" for c, m, mn, u, v, t in rls) or 'not measured yet'}</p>

<h2>Latest run — by question category</h2>
<table><tr><th>category</th><th colspan="2">hit@5</th><th>precision@5</th>
<th>coverage</th><th>gate</th><th>deny</th><th>allow</th><th>allow_hit</th></tr>
{''.join(cat_rows)}</table>
<p class="note">unanswerable is scored on the gate; persona_negative on
deny/allow — dashes are metrics that don't apply to a category. Under each
hit@5: 95% Wilson interval and question count — a slice under ~8 questions
is low-power, not hidden.</p>

<h2>Latest run — by expected source</h2>
<table><tr><th>source</th><th colspan="2">hit@5</th><th>precision@5</th>
<th>coverage</th><th>gate</th><th>deny</th><th>allow</th><th>allow_hit</th></tr>
{''.join(source_rows)}</table>
<p class="note">The source(s) a question's expected evidence lives in. A new
source cannot degrade an old one without a number moving here.</p>

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
