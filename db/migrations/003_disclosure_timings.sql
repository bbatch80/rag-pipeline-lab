-- Per-stage latency of the request that produced each disclosure row:
-- {"route": ms, "translate": ms, "embed": ms, "search": ms, "rerank": ms,
--  "payload": ms, "disclose": ms, "total": ms, "host": "..."}.
ALTER TABLE disclosure_log ADD COLUMN IF NOT EXISTS timings jsonb;
