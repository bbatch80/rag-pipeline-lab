-- Clinical policies (P1-PR6): authored, versioned medical policies, public
-- tier. Every version is indexed (a claim is adjudicated under the version
-- in effect on its date of service); default retrieval sees only the
-- version in effect (a hard filter on the record's effective window,
-- driven by the router's as_of date — never a boost). Versioned sources
-- are also event sources: their windows, not the plan-year edition filter,
-- decide which version applies.
ALTER TABLE sources ADD COLUMN IF NOT EXISTS versioned boolean NOT NULL DEFAULT false;
UPDATE sources SET status = 'ingested', event = true, versioned = true WHERE key = 'clinical_policies';
CREATE INDEX IF NOT EXISTS chunks_bm25_clinical_policy_idx ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'clinical_policy';
