"""docs/HANDOFF.md is the hand-off contract for a team that did not build
the platform. It drifts from the code silently unless a test reads both:
the spec version, every payload field, every chunk field, the entitlement
tiers, the MCP tools, and the identity groups it names must be the ones
the code has. Offline — no database."""

import json
import re

from raglab import config, identity, payload
from raglab.mcp_server import mcp

DOC = (config.REPO_ROOT / "docs" / "HANDOFF.md").read_text()
SCHEMA = json.loads((config.REPO_ROOT / "db" / "payload.schema.json").read_text())


def test_doc_names_the_spec_version_the_code_emits():
    assert f"`{payload.SPEC_VERSION}`" in DOC


def test_doc_names_every_payload_field():
    for field in SCHEMA["properties"]:
        assert f"`{field}`" in DOC, f"payload field {field!r} missing from the hand-off doc"


def test_doc_names_every_chunk_field_and_tier():
    chunk = SCHEMA["properties"]["chunks"]["items"]["properties"]
    for field in chunk:
        assert f"`{field}`" in DOC, f"chunk field {field!r} missing"
    for field in chunk["source"]["properties"]:
        assert f"`{field}`" in DOC, f"chunk source field {field!r} missing"
    for field in chunk["provenance"]["properties"]:
        assert f"`{field}`" in DOC, f"provenance field {field!r} missing"
    for tier in chunk["acl_basis"]["enum"]:
        assert f"`{tier}`" in DOC, f"tier {tier!r} missing"


def test_doc_names_every_mcp_tool():
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    for tool in tools:
        assert f"`{tool.name}(" in DOC, f"MCP tool {tool.name!r} missing"


def test_doc_lists_every_identity_group():
    table = DOC[DOC.index("| Group |"):]
    for group in identity.GROUPS:
        assert re.search(rf"^\| {re.escape(group)} \|", table, re.M), f"identity group {group!r} missing"


def test_doc_points_at_real_files():
    for path in re.findall(r"`((?:db|eval|deploy|docs|tests)/[A-Za-z0-9_./-]+)`", DOC):
        assert (config.REPO_ROOT / path).exists(), f"{path} named in the hand-off doc does not exist"
