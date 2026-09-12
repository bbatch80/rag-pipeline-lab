"""Clinical policies: authored, versioned; as-of routing; version precedence."""

import json

from raglab import router
from raglab.synth import policies


def test_policies_cover_the_cited_ids_and_versions(tmp_path):
    stats = policies.generate(tmp_path / "policies", tmp_path / "m.jsonl")
    assert stats == {"policies": 40, "with_second_version": 32, "with_third_version": 14, "documents": 86}
    recs = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert {r["policy_id"] for r in recs} == {f"CP-{i:04d}" for i in range(1, 41)}
    by_policy = {}
    for r in recs:
        by_policy.setdefault(r["policy_id"], []).append(r)
    for pid, versions in by_policy.items():
        versions.sort(key=lambda r: r["version"])
        for earlier, later in zip(versions, versions[1:]):  # a chain: each version ends where the next begins
            assert earlier["status"] == "superseded" and earlier["effective_to"] == later["effective_from"], pid
            assert later["supersedes"] == f"{pid}_v{earlier['version']}"
            t1 = (tmp_path / "policies" / earlier["doc"]).read_text(); t2 = (tmp_path / "policies" / later["doc"]).read_text()
            assert t1 != t2 and "Revision history" in t2 and "criterion changed" in t2
        assert versions[-1]["status"] == "current" and versions[-1]["effective_to"] is None
        assert all(r["year"] == int(r["effective_from"][:4]) for r in versions)
    # The golden-cited policies keep the versions the labels name.
    assert [r["version"] for r in by_policy["CP-0003"]] == [1, 2]
    assert [r["version"] for r in by_policy["CP-0011"]] == [1]
    v3 = (tmp_path / "policies" / "CP-0038_v3.md").read_text()
    assert "Version 3, effective 2026-01-01" in v3 and "'6 months' to '3 months'" in v3
    assert stats == policies.generate(tmp_path / "p2", tmp_path / "m2.jsonl") and (tmp_path / "m.jsonl").read_text() == (tmp_path / "m2.jsonl").read_text()


def test_as_of_date_parses_the_forms_people_write():
    assert router.as_of_date("What were the CP-0003 criteria as of August 1, 2024?") == "2024-08-01"
    assert router.as_of_date("CP-0008 criteria in effect on 01/06/2026") == "2026-01-06"
    assert router.as_of_date("criteria effective 2025-03-15") == "2025-03-15"
    assert router.as_of_date("What are the criteria for a sleep study under CP-0003?") is None
    r = router.route("What were the CP-0003 criteria as of August 1, 2024?")
    assert r.as_of == "2024-08-01" and not r.change
    assert router.route("How did CP-0003 change between versions?").change


def test_versioned_filter_and_policy_context():
    from raglab.retrieval import _filters
    from raglab.router import Route

    where, params = _filters(Route(scope="in_scope", years=(2026,)), None, (), (), None, ("clinical_policy",))
    assert "effective_from' <= %s" in where and "2026-12-31" in params
    where, params = _filters(Route(scope="in_scope", years=(2024,), as_of="2024-08-01"), None, (), (), None, ("clinical_policy",))
    assert "2024-08-01" in params and "2024-12-31" not in params
    where, params = _filters(Route(scope="in_scope"), None, (), (), {"policy_id": "CP-0003"}, ())
    assert "c.metadata->'record'->>%s IS NULL OR" in where and "CP-0003" in params
