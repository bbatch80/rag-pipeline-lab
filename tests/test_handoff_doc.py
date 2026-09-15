"""The hand-off specification is generated where it can be and prose where
it must be. These tests keep both honest, offline: every schema field has a
description and appears in the generated table, the example payload the
page shows validates against the schema, the markers in the prose name
real generated sections, and every file the prose points at exists."""

import json
import re

import jsonschema

from raglab import config, handoff, payload

DOC = handoff.DOC_PATH.read_text()
SCHEMA = json.loads(handoff.SCHEMA_PATH.read_text())


def _paths(node, path=""):
    for k, v in node.get("properties", {}).items():
        p = f"{path}.{k}" if path else k
        yield p, v
        yield from _paths(v, p)
        if isinstance(v.get("items"), dict):
            yield from _paths(v["items"], p + "[]")


def test_every_schema_field_is_described_and_tabulated():
    rows = {r["path"]: r for r in handoff.fields()}
    for path, node in _paths(SCHEMA):
        assert node.get("description"), f"{path} has no description in the schema"
        assert path in rows, f"{path} missing from the generated field table"
        assert rows[path]["description"] == node["description"]


def test_required_flags_follow_the_schema():
    rows = {r["path"]: r for r in handoff.fields()}
    for field in SCHEMA["required"]:
        assert rows[field]["required"]
    assert rows["confidence"]["required"] is False  # conditional (allOf), not top-level required
    assert rows["chunks[].acl_basis"]["required"] and rows["chunks[].provenance"]["required"] is False


def test_example_payload_is_the_current_spec():
    example = handoff.example_payload()
    jsonschema.validate(example, SCHEMA)
    assert example["spec_version"] == payload.SPEC_VERSION
    assert example["chunks"][0]["provenance"]["embedding_model"]


def test_markers_name_real_generated_sections_and_each_appears_once():
    found = handoff.markers(DOC)
    assert sorted(found) == sorted(handoff.GENERATED), found


def test_prose_points_at_real_files():
    for path in re.findall(r"`((?:db|eval|deploy|docs|tests)/[A-Za-z0-9_./-]+)`", DOC):
        assert (config.REPO_ROOT / path).exists(), f"{path} named in the hand-off doc does not exist"


def test_mcp_tools_are_read_from_the_server():
    names = {t["name"] for t in handoff.mcp_tools()}
    assert names == {"compose_context", "search_documents", "query_member_data"}
    compose = next(t for t in handoff.mcp_tools() if t["name"] == "compose_context")
    assert ("question", True) in compose["params"] and ("member_id", False) in compose["params"]
