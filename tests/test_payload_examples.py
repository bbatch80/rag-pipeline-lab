"""The payload page shows payloads built by the current code; they must be
the shape the schema defines. Offline."""

import json

import jsonschema

from raglab import handoff, payload

SCHEMA = json.loads(handoff.SCHEMA_PATH.read_text())


def test_single_example_is_the_current_spec():
    example = handoff.example_payload()
    jsonschema.validate(example, SCHEMA)
    assert example["spec_version"] == payload.SPEC_VERSION
    assert example["chunks"][0]["provenance"]["embedding_model"]


def test_composed_example_validates_and_shows_every_part():
    composed = handoff.example_composed_payload()
    jsonschema.validate(composed, SCHEMA)
    assert composed["status"] == "ok" and composed["plan"]["legs"][1]["kind"] == "member_query"
    assert composed["warehouse_results"][0]["rows"] and composed["chunks"][0]["provenance"]["embedding_model"]
    assert composed["chunks"][0]["source"]["doc_type"] == "call_note"
