"""Backfill DAG (manual trigger only): the embedding-model-swap drill.
Wipes Lane 1 derived state and rebuilds from sources under the current
config, honoring bulk-load-then-rebuild-index, then re-baselines the eval.
Everything it destroys is re-derivable; eval history survives by design."""

from datetime import datetime
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

REPO = Path(__file__).resolve().parents[1]


def raglab(cmd: str, task_id: str, group: str | None = None) -> BashOperator:
    runner = f"uv run --group {group}" if group else "uv run"
    return BashOperator(task_id=task_id, bash_command=f"cd {REPO} && {runner} {cmd}")


with DAG(
    dag_id="raglab_backfill",
    description="Full re-derive: wipe Lane 1, re-ingest, re-embed, rebuild index, re-baseline",
    schedule=None,
    start_date=datetime(2026, 8, 25),
    catchup=False,
    tags=["raglab"],
) as dag:
    wipe = raglab("raglab init-db", "wipe_lane1")
    ingest = raglab("raglab ingest --full", "ingest_full")
    embed = raglab("raglab embed", "embed_full")
    index = raglab("raglab index", "rebuild_index")
    rebaseline = raglab(
        "raglab eval-retrieval --gate --label post-backfill", "rebaseline",
        group="rerank",
    )

    wipe >> ingest >> embed >> index >> rebaseline
