"""Ingest-sync DAG: keep the corpus fresh against source and config changes.

Tasks map 1:1 onto raglab CLI commands — each task's success is the
command's run receipt exiting zero. Schedule = the stated freshness SLA.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

REPO = Path(__file__).resolve().parents[1]


def raglab(cmd: str, task_id: str, **kwargs) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {REPO} && uv run {cmd}",
        **kwargs,
    )


with DAG(
    dag_id="raglab_ingest_sync",
    description="Download + hash-diff ingest + embed + index refresh (freshness SLA: daily)",
    schedule="@daily",
    start_date=datetime(2026, 8, 25),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["raglab"],
) as dag:
    download = raglab("raglab download --full", "download")
    ingest = raglab("raglab ingest --full", "ingest")
    embed = raglab("raglab embed", "embed")
    index = raglab("raglab index", "index")
    status = raglab("raglab status", "status")

    download >> ingest >> embed >> index >> status
