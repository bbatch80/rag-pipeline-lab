"""Scheduled eval + drift DAG: the 'continuously monitor' loop. Re-runs the
deterministic retrieval suite (gated), checks embedding drift, refreshes the
dashboard. Generation evals stay manual — checkpoint instrument, not a cron."""

from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

REPO = Path(__file__).resolve().parents[1]


def raglab(cmd: str, task_id: str, group: str | None = None) -> BashOperator:
    runner = f"uv run --group {group}" if group else "uv run"
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {REPO} && {runner} {cmd}",
    )


with DAG(
    dag_id="raglab_eval_drift",
    description="Scheduled retrieval eval (gated) + embedding drift + dashboard refresh",
    schedule="@daily",
    start_date=datetime(2026, 8, 25),
    catchup=False,
    default_args={"retries": 0, "retry_delay": timedelta(minutes=5)},
    tags=["raglab"],
) as dag:
    eval_retrieval = raglab(
        "raglab eval-retrieval --gate --label scheduled", "eval_retrieval",
        group="rerank",
    )
    drift = raglab("raglab drift", "drift")
    dashboard = raglab("raglab dashboard", "dashboard")

    eval_retrieval >> drift >> dashboard
