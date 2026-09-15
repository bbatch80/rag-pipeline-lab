"""The hand-off specification as a page: generated from the schema, the MCP
server, the source registry, and the identity groups at request time, so it
cannot drift from the code. The prose around the generated parts lives in
docs/HANDOFF.md; markers there say where each generated table goes."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from functools import lru_cache
from pathlib import Path

import markdown
import psycopg

from raglab import config, identity, payload
from raglab.retrieval import Candidate
from raglab.router import Route

SCHEMA_PATH = config.REPO_ROOT / "db" / "payload.schema.json"
DOC_PATH = config.REPO_ROOT / "docs" / "HANDOFF.md"
MARKER = re.compile(r"<!-- generated: ([a-z_]+) -->")
GENERATED = ("payload_fields", "example_payload", "mcp_tools", "sources", "groups")


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _type_of(node: dict) -> str:
    if "enum" in node:
        return "one of " + ", ".join(str(v) for v in node["enum"])
    if "const" in node:
        return repr(node["const"])
    t = node.get("type", "any")
    if isinstance(t, list):
        return " | ".join(t)
    if t == "array":
        items = node.get("items")
        return f"array of {_type_of(items)}" if isinstance(items, dict) else "array"
    return t


def fields(node: dict | None = None, path: str = "") -> list[dict]:
    """Every property in the schema, depth-first in schema order: dotted path,
    type, whether its parent object requires it, its description."""
    node = schema() if node is None else node
    required = set(node.get("required", ()))
    rows: list[dict] = []
    for name, prop in node.get("properties", {}).items():
        dotted = f"{path}.{name}" if path else name
        rows.append({"path": dotted, "type": _type_of(prop), "required": name in required,
                     "description": prop.get("description", ""), "depth": dotted.count(".")})
        rows += fields(prop, dotted)
        items = prop.get("items")
        if isinstance(items, dict) and items.get("properties"):
            rows += fields(items, dotted + "[]")
    return rows


def example_payload() -> dict:
    """One real build of the current spec over a fixture chunk — the example a
    consumer copies, always the shape the code emits today."""
    c = Candidate(
        chunk_id=1, content="This chunk is from GEHA FEHB 71-006 (High, Standard) 2026, section "
        "'Emergency services/accidents'.\n\nEmergency room: In-network you pay 30% of the Plan allowance "
        "after the calendar-year deductible; out-of-network 30% of the Plan allowance plus any difference "
        "between our allowance and the billed amount.",
        section="Emergency services/accidents", doc_title="GEHA FEHB 71-006 (High, Standard) 2026",
        source_path="data/raw/2026/71-006.pdf", plan_code="71-006", year=2026, acl_tag="public",
        pages=[75], vector_rank=1, text_rank=2, rrf_score=0.0323, rerank_score=0.98,
        content_hash="5af5df7a3c15a22187441014399daa5387f8a39ef64ad488451eb2dc92a1c0e4",
        doc_type="brochure", recipe="unstructured-hires|2000/1500/250|template",
        embedding_model="text-embedding-3-small",
    )
    route = Route(scope="in_scope", years=(2026,), plan_codes=("71-006",))
    return payload.build("What would a High Option member pay at the ER?", route, [c])


@lru_cache(maxsize=1)
def mcp_tools() -> list[dict]:
    """The tools the MCP server registers, read from the server itself."""
    from raglab.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    out = []
    for t in tools:
        input_schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        params = list(input_schema.get("properties", {}))
        required = set(input_schema.get("required", ()))
        out.append({"name": t.name, "params": [(p, p in required) for p in params],
                    "description": inspect.cleandoc(t.description or "")})
    return out


def sources(conn: psycopg.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT key, doc_type, acl_tag, phi, versioned, member_scoped, event, chunk_profile, parser, display_name "
        "FROM sources WHERE doc_type IS NOT NULL ORDER BY source_id").fetchall()
    keys = ("key", "doc_type", "tier", "phi", "versioned", "member_scoped", "event", "chunking", "parser", "display_name")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def groups() -> list[dict]:
    return [{"group": g, "persona": persona, "warehouse_role": role or "—", "surfaces": ", ".join(surfaces),
             "description": desc} for g, (persona, role, surfaces, desc) in identity.GROUPS.items()]


def markers(text: str | None = None) -> list[str]:
    text = DOC_PATH.read_text() if text is None else text
    return MARKER.findall(text)


def render(conn: psycopg.Connection, env) -> str:
    """The document's prose as HTML with every generated section in place."""
    text = DOC_PATH.read_text()
    if text.startswith("# "):  # the page supplies the title; the repository copy keeps its own
        text = text.split("\n", 1)[1]
    body = markdown.markdown(text, extensions=["tables"])
    parts = {
        "payload_fields": env.get_template("partials/handoff_fields.html").render(
            fields=fields(), spec_version=payload.SPEC_VERSION),
        "example_payload": "<details class=\"example\"><summary>Example payload (spec %s), built by the current code</summary>"
                           "<pre class=\"chunktext\">%s</pre></details>" % (
                               payload.SPEC_VERSION, _escape(json.dumps(example_payload(), indent=2))),
        "mcp_tools": env.get_template("partials/handoff_tools.html").render(tools=mcp_tools()),
        "sources": env.get_template("partials/handoff_sources.html").render(sources=sources(conn)),
        "groups": env.get_template("partials/handoff_groups.html").render(groups=groups()),
    }
    return MARKER.sub(lambda m: parts.get(m.group(1), m.group(0)), body)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
