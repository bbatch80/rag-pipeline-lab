"""Every golden check on synthetic payloads: what passes, what fails, and
what the detail says when it fails. No database, no model."""

from raglab import checks, taxonomy


def _chunk(text, *, title="", path="", plan_code=None, year=None, pages=(), section="", score=0.9):
    return {"text": text, "leg": "documents",
            "source": {"title": title, "path": path, "plan_code": plan_code, "year": year,
                       "pages": list(pages), "section": section, "content_hash": f"h:{title}:{text[:8]}", "doc_type": None},
            "scores": {"rerank": score}}


def _payload(chunks=(), warehouse=(), legs=(), router=None, coverage=None, status="ok"):
    return {"status": status, "confidence": 0.9, "chunks": list(chunks), "warehouse_results": list(warehouse),
            "plan": {"legs": list(legs), "origin": "model", "fallback_reason": None},
            "router": router or {"years": [2026], "plan_codes": ["71-006"], "as_of": None},
            "coverage": coverage, "unresolved_identifiers": [], "timings": {}}


# ---------------------------------------------------------------- values
def test_expected_values_matches_inside_the_side_edition_only():
    item = {"expected_values": {"71-006/2026": ["$8,000"], "71-006/2021": ["$6,500"]}}
    good = _payload([_chunk("in-network $8,000 Self Only", plan_code="71-006", year=2026),
                     _chunk("out-of-pocket maximum is $6,500", plan_code="71-006", year=2021)])
    metric, value, detail = checks.check_expected_values(item, good)
    assert metric == "check_expected_values" and value == 1.0 and not detail["missing"]
    # the 2021 value present only in a 2026 chunk does not count for the 2021 side
    wrong_side = _payload([_chunk("$8,000 now, previously $6,500", plan_code="71-006", year=2026)])
    _, value, detail = checks.check_expected_values(item, wrong_side)
    assert value == 0.0 and detail["missing"] == {"71-006/2021": ["$6,500"]}


def test_expected_values_accepts_alternative_phrasings_of_one_value():
    item = {"expected_values": {"71-021/2025": [["$35 copay specialist", "$35 copayment for office visits to specialists"]]}}
    payload = _payload([_chunk("$35 copayment for office visits to specialists (no deductible)", plan_code="71-021", year=2025)])
    assert checks.check_expected_values(item, payload)[1] == 1.0
    assert checks.check_expected_values(item, _payload([_chunk("$35 copay applies", plan_code="71-021", year=2025)]))[1] == 0.0


def test_expected_values_without_an_edition_key_matches_anywhere_including_rows():
    item = {"expected_values": {"sops/current": ["pend the request"], "call_note/first": ["C0006300"]}}
    payload = _payload([_chunk("Missing clinical documentation: pend the request and notify")],
                       warehouse=[{"query_name": "member_calls", "status": "ok", "columns": ["CALL_ID"], "rows": [["C0006300"]]}])
    assert checks.check_expected_values(item, payload)[1] == 1.0


# ---------------------------------------------------------------- sets
def test_expected_set_reports_coverage_and_the_absent_members():
    item = {"expected_set": ["appeal_0167", "letter_0167", "call_C0005810"]}
    payload = _payload([_chunk("case", title="appeal_0167", path="data/internal/appeals/appeal_0167.md"),
                        _chunk("letter", title="letter_0167", path="data/internal/appeals_pdf/letter_0167.pdf")])
    metric, value, detail = checks.check_expected_set(item, payload)
    assert value == 0.0 and detail["coverage"] == "2/3" and detail["absent"] == ["call_C0005810"]


def test_expected_set_accepts_row_values_for_row_sets():
    item = {"expected_set": ["CLM-1", "CLM-2"]}
    payload = _payload(warehouse=[{"query_name": "member_denials", "status": "ok", "columns": ["CLAIM_ID"], "rows": [["CLM-1"], ["CLM-2"]]}])
    assert checks.check_expected_set(item, payload)[1] == 1.0


# ---------------------------------------------------------------- exclusions / versions
def test_exclude_sources_and_absent_titles_catch_a_superseded_version():
    item = {"exclude_sources": [{"internal": "policies/CP-0010_v2.md"}], "absent_titles": ["CP-0010 Continuous Glucose Monitoring v2"]}
    clean = _payload([_chunk("two events", title="CP-0010 Continuous Glucose Monitoring v1", path="data/internal/policies/CP-0010_v1.md")])
    assert checks.check_exclude_sources(item, clean)[1] == 1.0 and checks.check_absent_titles(item, clean)[1] == 1.0
    leaky = _payload([_chunk("one event", title="CP-0010 Continuous Glucose Monitoring v2", path="data/internal/policies/CP-0010_v2.md")])
    assert checks.check_exclude_sources(item, leaky)[1] == 0.0 and checks.check_absent_titles(item, leaky)[1] == 0.0


# ---------------------------------------------------------------- legs
def test_normalize_leg_reads_both_spellings():
    assert checks.normalize_leg("member_calls") == {"kind": "member_query", "query_name": "member_calls"}
    assert checks.normalize_leg("doc_probe:clinical_policy") == {"kind": "doc_probe", "sources": ["clinical_policy"]}
    assert checks.normalize_leg("doc_probe:appeal,clinical_note") == {"kind": "doc_probe", "sources": ["appeal", "clinical_note"]}
    assert checks.normalize_leg("aggregate:denials_by_reason") == {"kind": "member_query", "query_name": "denials_by_reason"}
    assert checks.normalize_leg({"kind": "member_query", "query_name": "appeal_case"})["query_name"] == "appeal_case"


def test_expected_legs_is_order_free_and_names_the_missing_leg():
    planned = [{"kind": "doc_probe", "sources": ["clinical_policy", "appeal"]}, {"kind": "member_query", "query_name": "member_calls"}]
    ok, missing = checks.routing_matches(planned, ["member_calls", "doc_probe:clinical_policy"])
    assert ok and not missing
    ok, missing = checks.routing_matches(planned, ["member_calls", "claim_adjudication"])
    assert not ok and missing == [{"kind": "member_query", "query_name": "claim_adjudication"}]
    hintless = [{"kind": "doc_probe", "sources": []}]
    assert checks.routing_matches(hintless, ["doc_probe:appeal"])[0], "a hint-less planned leg matches any expected source"
    # document legs are matched on SOURCE COVERAGE, not leg count: one planned leg with both
    # hints satisfies two expected families; a family no planned leg hints is named as missing
    one_leg = [{"kind": "doc_probe", "sources": ["appeal", "clinical_note"]}]
    assert checks.routing_matches(one_leg, ["doc_probe:appeal", "doc_probe:clinical_note"])[0]
    ok, missing = checks.routing_matches(one_leg, ["doc_probe:appeal,clinical_policy"])
    assert not ok and missing == [{"kind": "doc_probe", "sources": ["clinical_policy"]}]
    assert not checks.routing_matches([{"kind": "member_query", "query_name": "appeal_case"}], ["doc_probe:appeal"])[0], "no document leg at all"


# ---------------------------------------------------------------- rows / masking
def test_expected_rows_checks_count_and_column_values():
    warehouse = [{"query_name": "appeal_case", "status": "ok", "columns": ["CASE_ID", "DECISION", "FILED_DATE"],
                  "rows": [["APL-1", "overturned", "2024-05-03"]]},
                 {"query_name": "member_calls", "status": "ok", "columns": ["CALL_ID"], "rows": [["C1"], ["C2"], ["C3"]]}]
    payload = _payload(warehouse=warehouse)
    good = {"expected_rows": {"appeal_case": {"DECISION": "Overturned", "FILED_DATE": "2024-05-03"}, "member_calls": {"row_count": 3}}}
    assert checks.check_expected_rows(good, payload)[1] == 1.0
    bad = {"expected_rows": {"appeal_case": {"YEAR": 2024}}}  # a column the query does not return
    metric, value, detail = checks.check_expected_rows(bad, payload)
    assert value == 0.0 and detail["appeal_case"]["ok"] is False


def test_expected_masked_requires_the_role_mask_and_a_run():
    payload = _payload(warehouse=[{"query_name": "member_claims_summary", "status": "ok", "columns": ["TOTAL_COST"], "rows": [[None]],
                                   "masked_columns": ["TOTAL_COST", "PAYER_COVERAGE"]}])
    assert checks.check_expected_masked({"expected_masked": ["TOTAL_COST"]}, payload)[1] == 1.0
    assert checks.check_expected_masked({"expected_masked": []}, payload)[1] == 0.0, "something was masked that should not be"
    assert checks.check_expected_masked({"expected_masked": []}, _payload())[1] == 0.0, "no warehouse leg ran"


# ---------------------------------------------------------------- coverage / years / time
def test_expected_coverage_wants_one_chunk_per_plan_for_the_year():
    item = {"expected_coverage": {"program": "FEHB", "year": 2026, "plans": ["71-006", "71-014", "71-018"]}}
    payload = _payload([_chunk("a", plan_code="71-006", year=2026), _chunk("b", plan_code="71-014", year=2026),
                        _chunk("c", plan_code="71-018", year=2025)])
    metric, value, detail = checks.check_expected_coverage(item, payload)
    assert value == 0.0 and detail["coverage"] == "2/3" and detail["missing"] == ["71-018"]


def test_expected_not_offered_reads_the_coverage_note():
    item = {"expected_not_offered": [{"plan_code": "71-022", "year": 2026}]}
    assert checks.check_expected_not_offered(item, _payload(coverage={"not_offered": ["71-022"]}))[1] == 1.0
    assert checks.check_expected_not_offered(item, _payload(coverage={"not_offered": []}))[1] == 0.0
    assert checks.check_expected_not_offered(item, _payload(coverage=None))[1] == 0.0


def test_expected_years_and_time_points_read_the_route():
    payload = _payload(router={"years": [2025, 2026], "plan_codes": [], "as_of": "2025-07-01"})
    assert checks.check_expected_years({"expected_years": [2026]}, payload)[1] == 0.0
    assert checks.check_expected_years({"expected_years": [2025, 2026]}, payload)[1] == 1.0
    two_points = {"expected_time_points": [{"as_of": "2025-07-01"}, {"current": True}]}
    assert checks.check_expected_time_points(two_points, payload)[1] == 1.0
    only_past = _payload(router={"years": [2024, 2025], "plan_codes": [], "as_of": "2025-07-01"})
    assert checks.check_expected_time_points(two_points, only_past)[1] == 0.0
    events_only = {"expected_time_points": [{"event": "2024-08-31"}]}
    assert checks.check_expected_time_points(events_only, only_past)[1] == 1.0, "an event date is answer content, not a route check"


# ---------------------------------------------------------------- guardrails
def test_status_allow_and_deny_checks():
    assert checks.check_expect_status({"expect_status": "out_of_scope"}, _payload(status="out_of_scope"))[1] == 1.0
    assert checks.check_expect_status({"expect_status_any": ["ok", "insufficient_evidence"]}, _payload(status="ok"))[1] == 1.0
    item = {"allow_titles": ["call_C0001786"], "deny_titles": None}
    served = _payload([_chunk("note", title="call_C0001786")], status="ok")
    assert checks.check_allow_titles(item, served)[1] == 1.0
    assert checks.check_allow_titles(item, _payload([_chunk("note", title="call_C0001786")], status="insufficient_evidence"))[1] == 0.0
    metric, value, detail = checks.check_deny_titles(item, served, "public")
    assert metric == "check_deny_public" and value == 0.0 and detail["leaked"] == ["call_C0001786"]
    assert checks.check_deny_titles(item, _payload([_chunk("brochure text", title="GEHA FEHB 71-006")]), "public")[1] == 1.0


def test_expect_only_plan_flags_a_foreign_brochure():
    item = {"expect_only_plan": ["71-006", "71-021"]}
    ok = _payload([{**_chunk("x", plan_code="71-006", year=2026), "source": {**_chunk("x", plan_code="71-006", year=2026)["source"], "doc_type": "brochure"}}])
    assert checks.check_expect_only_plan(item, ok)[1] == 1.0
    foreign = _payload([{**_chunk("x", plan_code="71-018", year=2026), "source": {**_chunk("x", plan_code="71-018", year=2026)["source"], "doc_type": "brochure"}}])
    metric, value, detail = checks.check_expect_only_plan(item, foreign)
    assert value == 0.0 and detail["foreign_plans"] == ["71-018"]


# ---------------------------------------------------------------- reported metrics + fallback
def test_source_metrics_report_hit_precision_and_coverage():
    item = {"sources": [{"plan_code": "71-006", "year": 2026, "printed_pages": [14]}, {"plan_code": "71-006", "year": 2025, "printed_pages": [143]}]}
    payload = _payload([_chunk("a", plan_code="71-006", year=2026, pages=[16]), _chunk("b", plan_code="71-006", year=2026, pages=[40])])
    rows = dict((m, v) for m, v, _ in checks.source_metrics(item, payload))
    assert rows["hit@5"] == 1.0 and rows["precision@5"] == 0.5 and rows["source_coverage"] == 0.5


def test_declared_checks_dedupe_status_and_gate_prefix_is_a_gate():
    assert checks.declared_checks({"expect_status": "ok", "expect_status_any": ["ok"], "expected_set": []}) == ["expected_set", "expect_status"]
    assert taxonomy.is_gate_metric("check_expected_set") and taxonomy.is_gate_metric("hit@5") and not taxonomy.is_gate_metric("precision@5")
    verdicts = taxonomy.item_verdicts([("A", "yoy", "check_expected_values", 1.0, {}), ("A", "yoy", "check_expected_years", 0.0, {}),
                                       ("B", "named_query", "warehouse_skipped", 1.0, {}), ("C", "factual", "check_hit", 1.0, {})])
    assert verdicts["A"] == (False, []) and verdicts["B"] == (None, ["warehouse_skipped"]) and verdicts["C"] == (True, [])


# ---------------------------------------------------------------- health
def test_payload_health_sees_ties_duplicates_index_pages_and_needless_splits():
    dup = _chunk("same text here", title="t", pages=[3], score=0.99)
    payload = _payload([dup, dict(dup), _chunk("Index — Do not rely on this page", title="i", section="Index", score=0.988)],
                       legs=[{"kind": "doc_probe", "sources": ["brochure"]}, {"kind": "doc_probe", "sources": ["brochure"]}])
    h = checks.payload_health(payload)
    assert h["tie"] is True and h["duplicates"] == 1 and h["index_seats"] == 1 and h["needless_split"] is True
    summary = checks.summarize_health([h, checks.payload_health(_payload([_chunk("a", score=0.9), _chunk("b", score=0.4)]))])
    assert summary["tie_rate"] == 0.5 and summary["n_payloads"] == 2


def test_expected_rows_accepts_either_of_two_named_queries():
    """call_note-06 (2026-09-14): member_denials or member_calls both carry
    the claim; whichever the plan ran is judged, the other is not demanded."""
    from raglab import checks

    item = {"expected_rows": {"member_denials|member_calls": {"CLAIM_ID": "CLM-1"}}}
    calls = {"warehouse_results": [{"query_name": "member_calls", "columns": ["CALL_ID", "CLAIM_ID"], "rows": [["C1", "CLM-1"]]}]}
    denials = {"warehouse_results": [{"query_name": "member_denials", "columns": ["CLAIM_ID"], "rows": [["CLM-1"]]}]}
    neither = {"warehouse_results": [{"query_name": "member_calls", "columns": ["CALL_ID", "CLAIM_ID"], "rows": [["C1", "CLM-9"]]}]}
    assert checks.check_expected_rows(item, calls)[1] == 1.0
    assert checks.check_expected_rows(item, denials)[1] == 1.0
    assert checks.check_expected_rows(item, neither)[1] == 0.0
    assert checks.check_expected_rows(item, {"warehouse_results": []})[1] == 0.0
