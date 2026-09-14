"""Golden-item checks over a composed payload.

Every golden item declares what it expects with `expected_*` fields; each
field present is one check, all must hold for the item to pass, and an item
that declares nothing falls back to hit@5 on its `sources`. The group and
work category decide only where the result reports. Each check returns
`(metric, value, detail)`; the metric is `check_<field>` so a verdict names
the check that failed, not a bare 0.

The payload is the composed dict from `planner.compose` (spec 1.1.0):
`chunks` (text + source), `warehouse_results` (rows per named-query leg),
`plan` (the legs that ran), `router` (years, plan codes, as-of), `coverage`
(the plan-coverage note), `status`, `unresolved_identifiers`.
"""

import re
import statistics

from raglab import ablation, router

CHECK_PREFIX = "check_"
TIE_EPS = 0.005


# ---------------------------------------------------------------- helpers
def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _chunk_text(chunk: dict) -> str:
    return _norm(chunk.get("text") or "")


def _chunk_doc(chunk: dict) -> str:
    """A chunk's document identity: title, and the path it came from."""
    src = chunk.get("source") or {}
    return f"{src.get('title') or ''} | {src.get('path') or ''}"


def _doc_matches(chunk: dict, ref: str) -> bool:
    """`ref` names an internal document by path suffix ("calls/call_C0001.md")
    or by title stem ("call_C0001", "Formulary 2025"); either identifies it."""
    src = chunk.get("source") or {}
    path = str(src.get("path") or "")
    title = str(src.get("title") or "")
    stem = re.sub(r"\.(md|pdf|csv)$", "", ref.split("/")[-1])
    return path.endswith(ref) or title == stem or title == ref or (stem and stem in title)


def _rows(payload: dict, query_name: str) -> list[dict]:
    """Rows of the named-query leg(s) that ran `query_name`, as dicts."""
    out = []
    for w in payload.get("warehouse_results") or []:
        if w.get("query_name") != query_name:
            continue
        cols = w.get("columns") or []
        for row in w.get("rows") or []:
            out.append(dict(zip(cols, row)) if isinstance(row, (list, tuple)) else dict(row))
    return out


def _warehouse_unavailable(payload: dict) -> bool:
    return any(str(w.get("reason") or "").startswith("warehouse unavailable")
               for w in payload.get("warehouse_results") or [])


def _value_eq(got, want) -> bool:
    if got is None:
        return False
    if isinstance(want, bool):
        return bool(got) == want
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return abs(float(got) - float(want)) < 0.005 + 1e-9 * abs(float(want))
        except (TypeError, ValueError):
            return False
    return _norm(got) == _norm(want)


# ---------------------------------------------------------------- legs
def normalize_leg(leg) -> dict:
    """Expected legs are written two ways: dicts ({kind, query_name|source})
    or the shorthand "member_calls", "doc_probe:clinical_policy",
    "aggregate:denials_by_reason". Both become the dict form."""
    if isinstance(leg, dict):
        return leg
    text = str(leg)
    if text.startswith("doc_probe"):
        _, _, source = text.partition(":")
        sources = [x for x in source.split(",") if x]
        return {"kind": "doc_probe", **({"sources": sources} if sources else {})}
    if text.startswith("aggregate:"):
        return {"kind": "member_query", "query_name": text.split(":", 1)[1]}
    return {"kind": "member_query", "query_name": text}


def leg_matches(planned: dict, expected: dict) -> bool:
    """A member leg matches on its query name (`query_name`, or any of
    `query_name_any`); a document leg matches when its hints meet the
    expected `source` / `sources` / `sources_any` — a hint-less planned leg
    or a source-less expectation matches any."""
    if planned.get("kind") != expected.get("kind"):
        return False
    if expected.get("kind") == "member_query":
        allowed = set(expected.get("query_name_any") or [expected.get("query_name")])
        return planned.get("query_name") in allowed
    got = set(planned.get("sources") or [])
    want = set(expected.get("sources") or []) | set(expected.get("sources_any") or [])
    if expected.get("source"):
        want.add(expected["source"])
    return not got or not want or bool(got & want)


def routing_matches(planned: list[dict], expected: list) -> tuple[bool, list]:
    """Member legs: every expected query is matched by a distinct planned leg
    (order-free). Document legs: the plan must SEARCH every expected document
    family — the union of the planned document legs' hints covers the
    expected sources, however many legs carry them; a hint-less planned leg
    searches everything and covers any. Extra planned legs are allowed.
    Returns (ok, missing)."""
    expected = [normalize_leg(e) for e in expected]
    pool = [p for p in planned if p.get("kind") == "member_query"]
    missing = []
    for exp in (e for e in expected if e.get("kind") == "member_query"):
        hit = next((p for p in pool if leg_matches(p, exp)), None)
        if hit is None:
            missing.append(exp)
        else:
            pool.remove(hit)
    doc_expected = [e for e in expected if e.get("kind") == "doc_probe"]
    if doc_expected:
        doc_planned = [p for p in planned if p.get("kind") == "doc_probe"]
        if not doc_planned:
            missing.append({"kind": "doc_probe", "sources": sorted({s for e in doc_expected for s in _doc_sources(e)})})
        elif not any(not (p.get("sources") or []) for p in doc_planned):
            union = {s for p in doc_planned for s in (p.get("sources") or [])}
            absent = sorted({s for e in doc_expected for s in _doc_sources(e)} - union)
            if absent:
                missing.append({"kind": "doc_probe", "sources": absent})
    return (not missing), missing


def _doc_sources(expected_leg: dict) -> set:
    out = set(expected_leg.get("sources") or []) | set(expected_leg.get("sources_any") or [])
    if expected_leg.get("source"):
        out.add(expected_leg["source"])
    return out


# ---------------------------------------------------------------- reported metrics
def source_metrics(item: dict, payload: dict) -> list[tuple]:
    """hit@5 / precision@5 / source_coverage on the item's `sources`, the
    retrieval metrics every run reports whether or not they gate."""
    sources = item.get("sources") or []
    chunks = payload.get("chunks") or []
    if not sources:
        return []
    rel = [_relevant(c, sources) for c in chunks[:5]]
    hit = float(any(rel))
    precision = sum(rel) / len(rel) if rel else 0.0
    covered = sum(1 for s in sources if any(_relevant(c, [s]) for c in chunks[:10]))
    top = [round((c.get("scores") or {}).get("rerank") or 0.0, 4) for c in chunks[:5]]
    return [("hit@5", hit, {"top_scores": top}),
            ("precision@5", precision, {}),
            ("source_coverage", covered / len(sources), {"expected_sources": len(sources)})]


def _relevant(chunk: dict, sources: list[dict]) -> bool:
    src = chunk.get("source") or {}
    for s in sources:
        if "internal" in s:
            if _doc_matches(chunk, s["internal"]):
                return True
            continue
        if src.get("plan_code") != s.get("plan_code") or src.get("year") != s.get("year"):
            continue
        wanted = {p + ablation.PRINTED_TO_PDF_OFFSET + d
                  for p in s.get("printed_pages", [])
                  for d in range(-ablation.PAGE_TOLERANCE, ablation.PAGE_TOLERANCE + 1)}
        if wanted & set(src.get("pages") or []):
            return True
    return False


# ---------------------------------------------------------------- declared checks
def check_expected_values(item: dict, payload: dict) -> tuple:
    """Every side lists strings that must appear in the payload. A side keyed
    "<plan_code>/<year>" is matched inside that edition's chunks and a key
    ending in "/<year>" ("formulary/2025") inside that year's chunks; any
    other key ("sops/current", "CP-0010/as_of_2025-05-01") is matched
    anywhere — the key names the side for the reader, the strings are the
    check. Row values count too. An entry that is a list names alternative
    phrasings of one value; any of them satisfies it."""
    chunks = payload.get("chunks") or []
    rows_text = _norm(" ".join(str(v) for w in payload.get("warehouse_results") or []
                               for r in (w.get("rows") or []) for v in (r if isinstance(r, (list, tuple)) else [r])))
    sides_ok, missing = {}, {}
    for side, wanted in (item.get("expected_values") or {}).items():
        m = re.fullmatch(r"(\d\d-\d\d\d)/(\d{4})", side)
        y = re.fullmatch(r"[^/]+/(\d{4})", side)  # "formulary/2025": that year's chunks only
        pool = [c for c in chunks if
                (m is None or ((c.get("source") or {}).get("plan_code") == m.group(1) and (c.get("source") or {}).get("year") == int(m.group(2))))
                and (y is None or (c.get("source") or {}).get("year") == int(y.group(1)))]
        text = " ".join(_chunk_text(c) for c in pool) + " " + rows_text
        # an entry may list ALTERNATIVE phrasings of one value (a summary page vs a benefits row)
        absent = [w for w in wanted if not any(_norm(alt) in text for alt in (w if isinstance(w, list) else [w]))]
        sides_ok[side] = not absent
        if absent:
            missing[side] = absent
    return (f"{CHECK_PREFIX}expected_values", float(all(sides_ok.values())), {"sides": sides_ok, "missing": missing})


def check_expected_set(item: dict, payload: dict) -> tuple:
    """Every member of the bounded set is in the payload — a document (by
    path suffix or title stem) or, for row sets, a value in any returned row."""
    chunks = payload.get("chunks") or []
    rows_text = _norm(" ".join(str(v) for w in payload.get("warehouse_results") or []
                               for r in (w.get("rows") or []) for v in (r if isinstance(r, (list, tuple)) else [r])))
    present, absent = [], []
    for member in item.get("expected_set") or []:
        if any(_doc_matches(c, member) for c in chunks) or _norm(member) in rows_text:
            present.append(member)
        else:
            absent.append(member)
    total = len(present) + len(absent)
    return (f"{CHECK_PREFIX}expected_set", float(not absent), {"coverage": f"{len(present)}/{total}", "absent": absent})


def check_exclude_sources(item: dict, payload: dict) -> tuple:
    leaked = [s["internal"] for s in item.get("exclude_sources") or []
              if any(_doc_matches(c, s["internal"]) for c in payload.get("chunks") or [])]
    return (f"{CHECK_PREFIX}exclude_sources", float(not leaked), {"leaked": leaked})


def check_absent_titles(item: dict, payload: dict) -> tuple:
    titles = [str((c.get("source") or {}).get("title") or "") for c in payload.get("chunks") or []]
    leaked = [t for t in item.get("absent_titles") or [] if any(t in title for title in titles)]
    return (f"{CHECK_PREFIX}absent_titles", float(not leaked), {"leaked": leaked})


def check_expected_legs(item: dict, payload: dict) -> tuple:
    planned = ((payload.get("plan") or {}).get("legs")) or []
    ok, missing = routing_matches(planned, item.get("expected_legs") or [])
    return (f"{CHECK_PREFIX}expected_legs", float(ok),
            {"planned": [(l.get("kind"), l.get("query_name") or ",".join(l.get("sources") or [])) for l in planned],
             "missing": missing, "origin": (payload.get("plan") or {}).get("origin"),
             "fallback": (payload.get("plan") or {}).get("fallback_reason")})


def check_expected_rows(item: dict, payload: dict) -> tuple:
    """Per named query: `row_count` == n, and/or one row carrying every
    listed column value. A column the query does not return fails (that is
    the semantic-layer measure: a bound the catalog does not expose)."""
    results = {}
    for query_name, want in (item.get("expected_rows") or {}).items():
        # "a|b": either query may carry the row (two legitimate plans for one question)
        names = query_name.split("|")
        query_name = next((n for n in names if _rows(payload, n)), names[0])
        rows = _rows(payload, query_name)
        want = dict(want)
        count = want.pop("row_count", None)
        ok = True
        if count is not None:
            ok &= len(rows) == count
        if want:
            ok &= any(all(_value_eq(r.get(k), v) for k, v in want.items()) for r in rows)
        results[query_name] = {"ok": bool(ok), "rows": len(rows)}
    return (f"{CHECK_PREFIX}expected_rows", float(all(r["ok"] for r in results.values()) if results else 0.0), results)


def check_expected_masked(item: dict, payload: dict) -> tuple:
    """The role's masking on the warehouse legs: every expected column is
    reported masked (an empty expectation means nothing was masked)."""
    masked = sorted({c for w in payload.get("warehouse_results") or [] for c in (w.get("masked_columns") or [])})
    expected = sorted(item.get("expected_masked") or [])
    ran = any(w.get("status") == "ok" for w in payload.get("warehouse_results") or [])
    ok = ran and (set(expected) <= set(masked) if expected else not masked)
    return (f"{CHECK_PREFIX}expected_masked", float(ok), {"masked": masked, "expected": expected, "warehouse_ran": ran})


def check_expected_coverage(item: dict, payload: dict) -> tuple:
    """One evidence chunk per plan of the population, for the year."""
    spec = item.get("expected_coverage") or {}
    year = spec.get("year")
    seen = {(c.get("source") or {}).get("plan_code") for c in payload.get("chunks") or []
            if year is None or (c.get("source") or {}).get("year") == year}
    plans = spec.get("plans") or []
    covered = [p for p in plans if p in seen]
    return (f"{CHECK_PREFIX}expected_coverage", float(len(covered) == len(plans)),
            {"coverage": f"{len(covered)}/{len(plans)}", "missing": [p for p in plans if p not in seen]})


def check_expected_not_offered(item: dict, payload: dict) -> tuple:
    """The coverage note states the missing side: the plan appears under
    `not_offered` (with or without the year — the per-year note is OFM-P3)."""
    note = payload.get("coverage") or {}
    stated = note.get("not_offered") or []
    ok = True
    for want in item.get("expected_not_offered") or []:
        code = want.get("plan_code")
        ok &= any((s == code) or (isinstance(s, dict) and s.get("plan_code") == code) or (isinstance(s, str) and s.startswith(code))
                  for s in stated)
    return (f"{CHECK_PREFIX}expected_not_offered", float(ok), {"stated": stated})


def check_expected_years(item: dict, payload: dict) -> tuple:
    got = sorted((payload.get("router") or {}).get("years") or [])
    want = sorted(item.get("expected_years") or [])
    return (f"{CHECK_PREFIX}expected_years", float(got == want), {"routed": got, "expected": want})


def check_expected_time_points(item: dict, payload: dict) -> tuple:
    """Each time point the question names must be represented in the route:
    an as-of date by the route's as-of (or its year), `current` by the current
    year. Event dates are answer content, not route checks. One as-of slot is
    the known limit (OFM-Q7)."""
    r = payload.get("router") or {}
    years = set(r.get("years") or [])
    as_of = r.get("as_of")
    results = []
    for point in item.get("expected_time_points") or []:
        if point.get("current"):
            results.append(router.CURRENT_YEAR in years)
        elif point.get("as_of"):
            results.append(as_of == point["as_of"] or int(point["as_of"][:4]) in years)
        # an `event` point is a date in the ANSWER (issued / effective / due), not a route requirement
    return (f"{CHECK_PREFIX}expected_time_points", float(all(results)),  # events only: nothing to route, vacuously true
            {"routed_years": sorted(years), "as_of": as_of, "points": results})


def check_expect_status(item: dict, payload: dict) -> tuple:
    if "expect_status_any" in item:
        ok = payload.get("status") in item["expect_status_any"]
    else:
        ok = payload.get("status") == item.get("expect_status")
    return (f"{CHECK_PREFIX}expect_status", float(ok), {"status": payload.get("status"), "confidence": payload.get("confidence")})


def check_required_evidence(item: dict, payload: dict) -> tuple:
    need = item.get("required_evidence") or {}
    ok = all(any(w.get("query_name") in q.split("|") and (w.get("row_count") or 0) >= n
                 for w in payload.get("warehouse_results") or [])
             for q, n in need.get("rows_min", {}).items())
    return (f"{CHECK_PREFIX}required_evidence", float(ok), {"rows_min": need.get("rows_min", {})})


def check_expect_only_plan(item: dict, payload: dict) -> tuple:
    allowed = item["expect_only_plan"]
    allowed = allowed if isinstance(allowed, list) else [allowed]
    brochures = [c for c in payload.get("chunks") or [] if (c.get("source") or {}).get("doc_type") == "brochure"]
    foreign = sorted({(c.get("source") or {}).get("plan_code") for c in brochures} - set(allowed))
    return (f"{CHECK_PREFIX}expect_only_plan", float(not foreign), {"foreign_plans": foreign})


def check_expect_only_member(item: dict, payload: dict, conn=None) -> tuple:
    """Every member-scoped chunk belongs to the member: checked by content
    hash against `documents.member_key` when a connection is given."""
    hashes = [(c.get("source") or {}).get("content_hash") for c in payload.get("chunks") or []]
    hashes = [h for h in hashes if h]
    if conn is None or not hashes:
        return (f"{CHECK_PREFIX}expect_only_member", float(bool(hashes)), {"checked": 0})
    person = conn.execute("SELECT id FROM synthea.patients WHERE member_id = %s", (item["expect_only_member"],)).fetchone()
    others = conn.execute(
        "SELECT count(*) FROM documents WHERE content_hash = ANY(%s) AND member_key IS NOT NULL AND member_key::text <> %s",
        (hashes, person[0] if person else ""),
    ).fetchone()[0]
    return (f"{CHECK_PREFIX}expect_only_member", float(others == 0), {"foreign_chunks": others, "checked": len(hashes)})


def check_allow_titles(item: dict, payload: dict) -> tuple:
    """The entitled persona's payload cites the protected document in its
    top five and answers (status ok)."""
    titles = [str((c.get("source") or {}).get("title") or "") for c in (payload.get("chunks") or [])[:5]]
    hit = any(exp in t for exp in item["allow_titles"] for t in titles)
    ok = hit and payload.get("status") == "ok"
    return (f"{CHECK_PREFIX}allow", float(ok), {"hit": hit, "status": payload.get("status"), "confidence": payload.get("confidence")})


def check_deny_titles(item: dict, payload: dict, persona: str) -> tuple:
    """The denied persona's payload holds none of the protected documents."""
    protected = list(item.get("allow_titles") or []) + list(item.get("deny_titles") or [])
    titles = [str((c.get("source") or {}).get("title") or "") for c in payload.get("chunks") or []]
    leaked = [t for t in titles if any(p in t for p in protected)]
    return (f"{CHECK_PREFIX}deny_{persona}", float(not leaked), {"leaked": leaked, "status": payload.get("status")})


DECLARED = {
    "expected_values": check_expected_values,
    "expected_set": check_expected_set,
    "exclude_sources": check_exclude_sources,
    "absent_titles": check_absent_titles,
    "expected_legs": check_expected_legs,
    "expected_rows": check_expected_rows,
    "expected_masked": check_expected_masked,
    "expected_coverage": check_expected_coverage,
    "expected_not_offered": check_expected_not_offered,
    "expected_years": check_expected_years,
    "expected_time_points": check_expected_time_points,
    "expect_status": check_expect_status,
    "expect_status_any": check_expect_status,
    "required_evidence": check_required_evidence,
    "expect_only_plan": check_expect_only_plan,
}

WAREHOUSE_CHECKS = {"expected_rows", "expected_masked", "required_evidence"}


def declared_checks(item: dict) -> list[str]:
    """The declared fields, deduplicated (expect_status / expect_status_any
    share one check)."""
    seen, out = set(), []
    for field in DECLARED:
        if field in item:
            fn = DECLARED[field].__name__
            if fn not in seen:
                seen.add(fn)
                out.append(field)
    return out


def run_declared(item: dict, payload: dict, conn=None) -> list[tuple]:
    rows = []
    for field in declared_checks(item):
        rows.append(DECLARED[field](item, payload))
    if "expect_only_member" in item:
        rows.append(check_expect_only_member(item, payload, conn))
    return rows


# ---------------------------------------------------------------- payload health
INDEX_MARKERS = re.compile(r"do not rely on this page|table of contents", re.I)


def payload_health(payload: dict) -> dict:
    """The numbers pass/fail cannot see: the top-two gap (ties), duplicate
    chunks, index pages seated, and legs identical after filter binding."""
    chunks = payload.get("chunks") or []
    hashes = [(c.get("source") or {}).get("content_hash") for c in chunks]
    keys = [((c.get("source") or {}).get("title"), str((c.get("source") or {}).get("pages")), _chunk_text(c)[:80]) for c in chunks]
    duplicates = len(keys) - len(set(keys))
    # The top-two gap over DISTINCT chunks: a duplicated seat (P1) is not a tie.
    best: dict = {}
    for key, c in zip(keys, chunks):
        score = float((c.get("scores") or {}).get("rerank") or 0.0)
        best[key] = max(best.get(key, 0.0), score)
    scores = sorted(best.values(), reverse=True)
    gap = (scores[0] - scores[1]) if len(scores) >= 2 else None
    index_seats = sum(1 for c in chunks
                      if INDEX_MARKERS.search(str((c.get("source") or {}).get("section") or "")) or INDEX_MARKERS.search(_chunk_text(c)))
    legs = (payload.get("plan") or {}).get("legs") or []
    sig = [(l.get("kind"), l.get("query_name"), tuple(sorted(l.get("sources") or []))) for l in legs]
    needless_split = len(sig) > 1 and len(set(sig)) < len(sig)
    return {"top_gap": None if gap is None else round(gap, 4),
            "tie": None if gap is None else gap < TIE_EPS,
            "chunks": len(chunks), "duplicates": duplicates, "index_seats": index_seats,
            "legs": len(legs), "needless_split": needless_split, "hashes": len([h for h in hashes if h])}


def summarize_health(details: list[dict]) -> dict:
    gaps = [d["top_gap"] for d in details if d.get("top_gap") is not None]
    with_chunks = [d for d in details if d.get("chunks")]
    with_legs = [d for d in details if d.get("legs", 0) > 1]
    return {
        "tie_rate": round(sum(1 for g in gaps if g < TIE_EPS) / len(gaps), 3) if gaps else None,
        "median_margin": round(statistics.median(gaps), 4) if gaps else None,
        "duplicate_share": round(sum(d["duplicates"] for d in with_chunks) / sum(d["chunks"] for d in with_chunks), 3) if with_chunks else None,
        "index_seat_share": round(sum(d["index_seats"] for d in with_chunks) / sum(d["chunks"] for d in with_chunks), 3) if with_chunks else None,
        "needless_split_rate": round(sum(1 for d in with_legs if d["needless_split"]) / len(with_legs), 3) if with_legs else None,
        "n_payloads": len(details),
    }
