-- The search copy defaults to the display copy: any inserter that does not
-- provide index_text gets content, so the column stays NOT NULL without
-- every writer knowing about it. Ingest overrides it for normalized sources.
CREATE OR REPLACE FUNCTION chunks_default_index_text() RETURNS trigger AS $$
BEGIN
    IF NEW.index_text IS NULL THEN
        NEW.index_text := NEW.content;
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_default_index_text ON chunks;
CREATE TRIGGER chunks_default_index_text
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_default_index_text();
