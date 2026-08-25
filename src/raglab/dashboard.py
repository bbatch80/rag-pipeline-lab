"""Minimal metrics dashboard: plain HTML tables over the eval store.
Exists from the first eval run; Phase 8 polishes, never builds."""

from pathlib import Path

import psycopg

from raglab import config

OUT_PATH = config.REPO_ROOT / "data" / "eval" / "dashboard.html"

_STYLE = """
body { font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }
table { border-collapse: collapse; margin: 1rem 0 2rem; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: right; }
th { background: #f0f0f0; } td:first-child, th:first-child { text-align: left; }
.bad { background: #ffe0e0; } h2 { margin-top: 2rem; }
"""


def _table(headers: list[str], rows: list[tuple]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td class=\"{'bad' if isinstance(v, float) and v < 0.5 else ''}\">"
            f"{v if not isinstance(v, float) else f'{v:.3f}'}</td>"
            for v in row
        ) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def render(conn: psycopg.Connection, out_path: Path = OUT_PATH) -> Path:
    runs = conn.execute(
        "SELECT r.id, to_char(r.started_at, 'YYYY-MM-DD HH24:MI'), r.kind, "
        "r.config_label, r.git_sha, count(s.id) "
        "FROM eval_runs r LEFT JOIN eval_scores s ON s.run_id = r.id "
        "GROUP BY r.id ORDER BY r.id DESC LIMIT 25"
    ).fetchall()

    latest = conn.execute(
        "SELECT s.category, s.metric, round(avg(s.value), 3) "
        "FROM eval_scores s "
        "WHERE s.run_id = (SELECT max(id) FROM eval_runs WHERE kind = 'retrieval' "
        "                  AND config_label NOT LIKE '%%SABOTAGE%%') "
        "GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()

    trend = conn.execute(
        "SELECT s.run_id, r.config_label, s.metric, round(avg(s.value), 3) "
        "FROM eval_scores s JOIN eval_runs r ON r.id = s.run_id "
        "WHERE s.metric IN ('hit@5', 'precision@5', 'source_coverage') "
        "GROUP BY 1, 2, 3 ORDER BY s.run_id DESC, s.metric LIMIT 60"
    ).fetchall()

    gen = conn.execute(
        "SELECT s.generator, s.judge, s.metric, round(avg(s.value), 3), count(*) "
        "FROM eval_scores s WHERE s.generator IS NOT NULL "
        "GROUP BY 1, 2, 3 ORDER BY 1, 3"
    ).fetchall()

    html = [
        f"<style>{_STYLE}</style>",
        "<h1>raglab metrics</h1>",
        "<h2>Runs</h2>",
        _table(["run", "started", "kind", "config", "sha", "scores"], runs),
        "<h2>Latest retrieval run — by category</h2>",
        _table(["category", "metric", "mean"], latest),
        "<h2>Metric trend by run</h2>",
        _table(["run", "config", "metric", "mean"], trend),
    ]
    if gen:
        html += [
            "<h2>Generation (cross-family judged)</h2>",
            _table(["generator", "judge", "metric", "mean", "n"], gen),
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html))
    return out_path
