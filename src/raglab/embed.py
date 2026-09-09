"""Chunk embedding with the embed-where-NULL resumability contract: the batch
loop always selects chunks whose embedding is NULL, so an interrupted run
resumes exactly where it stopped. Progress commits per batch."""

from dataclasses import dataclass, field

import psycopg

MODEL = "text-embedding-3-small"
BATCH_SIZE = 128
# text-embedding-3-small pricing, USD per 1M tokens.
PRICE_PER_MTOK = 0.02


@dataclass
class EmbedStats:
    embedded: int = 0
    batches: int = 0
    tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return self.tokens / 1e6 * PRICE_PER_MTOK


def _to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def embed_pending(
    conn: psycopg.Connection,
    client,
    batch_size: int = BATCH_SIZE,
    model: str = MODEL,
) -> EmbedStats:
    stats = EmbedStats()
    while True:
        rows = conn.execute(
            "SELECT id, index_text FROM chunks WHERE embedding IS NULL "
            "ORDER BY id LIMIT %s",
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        response = client.embeddings.create(
            model=model, input=[content for _, content in rows]
        )
        for (chunk_id, _), item in zip(rows, response.data, strict=True):
            conn.execute(
                "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                (_to_vector_literal(item.embedding), chunk_id),
            )
        conn.commit()  # per-batch commit: interruption loses at most one batch
        stats.embedded += len(rows)
        stats.batches += 1
        stats.tokens += response.usage.total_tokens
    return stats
