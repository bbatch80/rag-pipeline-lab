-- OPM carrier letters (P1-PR7): real public regulatory PDFs, fetched by a
-- fixed manifest (raglab.letters). Public tier; year = letter year; the
-- letter's number, date, subject and URL ride as record metadata. A letter
-- is a dated event, not a plan-year edition — issued in year N, usually
-- about plan year N+1 — so it is exempt from the plan-year filter (event).
UPDATE sources SET status = 'ingested', event = true WHERE key = 'carrier_letters';
CREATE INDEX IF NOT EXISTS chunks_bm25_carrier_letter_idx ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'carrier_letter';
