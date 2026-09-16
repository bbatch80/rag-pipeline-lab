"""Cross-encoder reranking over fused candidates. The reranker reads query
and chunk jointly — catching negation, plan/option distinctions, and
answers-vs-discusses — and its scores double as the abstention gauge:
best-score-below-threshold means the corpus likely lacks an answer.

Model is local (no-paid-API constraint), pre-fetched by
`huggingface_hub.snapshot_download` as an explicit setup step.
"""

import os
import threading

from raglab.retrieval import Candidate

# Rerankers under comparison; one ships. Cross-encoder scores are not
# calibrated across models, so each carries its own abstention threshold,
# derived from the golden set's answerable/unanswerable score separation
# (v1 method for bge-base: answerable best scores bottomed at 0.72, the
# absent-topic question peaked at 0.30 → midpoint 0.5. KNOWN LIMIT:
# redirect-style unanswerables score high on genuinely relevant chunks and
# are caught at the generation layer instead).
# One reranker ships. gte-reranker-modernbert-base (2025, 149M) was measured
# against it on 2026-09-08 and lost: hit@5 0.931 vs 0.966, coverage 0.787 vs
# 0.868, an entitled persona wrongly blocked (allow_answered 0.8), rerank
# p50 1964 vs 1157 ms; its answerable/unanswerable margin was 0.06 vs 0.42.
RERANKERS = {
    # ---- bake-off v2 candidates (2026-09-12): thresholds provisional until
    # re-derived from each model's own score separation (raglab rerank-bakeoff).
    "bge-v2-m3": {"model": "BAAI/bge-reranker-v2-m3", "kind": "cross-encoder", "threshold": 0.5,
                  "thresholds": {"call_note": 0.1, "appeal": 0.1, "clinical_note": 0.1}, "size_gb": 2.2},
    "mxbai-large-v2": {"model": "mixedbread-ai/mxbai-rerank-large-v2", "kind": "cross-encoder", "threshold": 0.5,
                       "thresholds": {"call_note": 0.1, "appeal": 0.1, "clinical_note": 0.1}, "size_gb": 3.0},
    # Derived 2026-09-16 from run 879 (full golden set): expected refusals score
    # 0.516 / 0.500 / 0.0 / 0.0 (plus one false-answer kind at 0.99 that fools
    # every model); answered prose items score >= 0.899 -> the midpoint 0.7.
    # Record bars unchanged (record-lane items behaved identically to bge-base).
    "qwen3-0.6b": {"model": "Qwen/Qwen3-Reranker-0.6B", "kind": "qwen3", "threshold": 0.7,
                   "thresholds": {"call_note": 0.1, "appeal": 0.1, "clinical_note": 0.1}},  # ships
    "bge-base": {
        "model": "BAAI/bge-reranker-base",  # 2023, 278M
        "kind": "cross-encoder",
        "threshold": 0.5,  # prose sources (calibrated on brochures, Phase 0)
        # Records: a terse, de-identified note scores lower in absolute
        # terms even when it is the answer. Calibrated on the call-note
        # golden slice (2026-09-09, 10 items): correct notes 0.16–0.99,
        # unrelated notes ≈ 0.00–0.01; a threshold of 0.1 sits under every
        # answered item with margin and above the noise floor.
        # Clinical notes are records of the same shape (member-scoped,
        # de-identified, one document per event) and were added after this
        # table was calibrated; they carried the prose bar until 2026-09-12
        # (persona_negative-07: the right note ranked first at 0.30 and the
        # payload abstained). Set by category, not by measurement; the
        # reranker bake-off re-derives every bar from each model's scores.
        "thresholds": {"call_note": 0.1, "appeal": 0.1, "clinical_note": 0.1},
    },
}
# Qwen3 ships (2026-09-16: full golden set 121/163 vs bge-base 119, member
# wording read as meaning); bge-base stays in the image as the one-line
# fallback: RAGLAB_RERANKER=bge-base and a restart (deploy.sh reranker).
RERANKER = os.environ.get("RAGLAB_RERANKER", "qwen3-0.6b")
MODEL_NAME = RERANKERS[RERANKER]["model"]
ABSTAIN_THRESHOLD = RERANKERS[RERANKER]["threshold"]
ABSTAIN_BY_SOURCE = RERANKERS[RERANKER].get("thresholds", {})


def threshold_for(doc_type: str) -> float:
    return ABSTAIN_BY_SOURCE.get(doc_type, ABSTAIN_THRESHOLD)
TOP_N_OUT = 10

_model = None


# ---------------------------------------------------------------- cache
# A cross-encoder score is a pure function of (model, query text, chunk
# text). The cache memoizes exactly that: the model name + the weight
# snapshot it loaded, the final ranking query string, the sha256 of the
# exact text scored (prefix + record header + search copy, per text mode).
# No chunk id, no timestamp — a re-ingest, a search-copy rebuild, a header
# rule, a query-side change, or a model swap all change the key and miss
# honestly. Thresholds are applied after scoring and are not in the key.
RERANK_CACHE = os.environ.get("RAGLAB_RERANK_CACHE", "on") != "off"
# Inference precision. bf16 runs the cross-encoder under CPU autocast — on a
# CPU with AMX (the deployed VM) 3.4x faster; scores move in the fourth
# decimal (measured 0.0403 -> 0.0402), far inside the abstention bars. Off
# by default: the Mac, the baseline, and the CI gate stay fp32. Part of the
# cache key, so scores from the two precisions never mix.
RERANK_DTYPE = os.environ.get("RAGLAB_RERANK_DTYPE", "fp32")
if RERANK_DTYPE not in ("fp32", "bf16"):
    raise SystemExit(f"RAGLAB_RERANK_DTYPE={RERANK_DTYPE!r}: expected fp32 or bf16")


def _predict(model, pairs: list[tuple[str, str]]) -> list[float]:
    """model.predict in the configured precision."""
    if not pairs:
        return []
    with _predict_lock:
        if RERANK_DTYPE == "bf16":
            import torch

            with torch.autocast("cpu", dtype=torch.bfloat16):
                return [float(x) for x in model.predict(pairs)]
        return [float(x) for x in model.predict(pairs)]
CACHE_STATS = {"hits": 0, "misses": 0}
_cache_local = threading.local()  # one autocommit cache connection per thread (the eval runs items in parallel)
_predict_lock = threading.Lock()  # inference is CPU-bound: one prediction at a time, network waits overlap elsewhere
_model_key: str | None = None


def model_key() -> str:
    """MODEL_NAME plus the weight snapshot revision the local cache holds."""
    global _model_key
    if _model_key is None:
        rev = "local"
        try:
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_id == MODEL_NAME:
                    revs = sorted(repo.revisions, key=lambda r: r.last_modified)
                    rev = revs[-1].commit_hash[:12] if revs else "local"
        except Exception:  # no hub cache metadata: still keyed on the name
            pass
        _model_key = f"{MODEL_NAME}@{rev}"
    return _model_key + ("+reid" if RERANK_REIDENTIFY else "") + ("+bf16" if RERANK_DTYPE == "bf16" else "")


def _cache_connection():
    """An owner connection of its own (autocommit): the pipeline's session
    may be running under a persona role that cannot write. Tests patch
    this to hand in the rolled-back fixture connection."""
    conn = getattr(_cache_local, "conn", None)
    if conn is None or conn.closed:
        from raglab import db

        conn = db.connect()
        conn.autocommit = True
        _cache_local.conn = conn
    return conn


def score_pairs(model, pairs: list[tuple[str, str]], conn=None) -> list[float]:
    """model.predict through the rerank_scores table: hits are read, misses
    are predicted in ONE batched call and stored. Without the table (a
    fresh database before any eval) or with the cache off, plain predict."""
    import hashlib

    if not RERANK_CACHE or not pairs:
        return _predict(model, pairs)
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    keys = [(h(q), h(t)) for q, t in pairs]
    mk = model_key()
    try:
        conn = conn or _cache_connection()
        with conn.transaction():
            rows = conn.execute(
                "SELECT query_hash, text_hash, score FROM rerank_scores "
                "WHERE model = %s AND query_hash = ANY(%s) AND text_hash = ANY(%s)",
                (mk, sorted({q for q, _ in keys}), sorted({t for _, t in keys})),
            ).fetchall()
    except Exception:  # no table / no database: score everything
        return _predict(model, pairs)
    known = {(q, t): float(s) for q, t, s in rows if s == s}  # a cached NaN is a miss, never a score
    scores: list[float | None] = [known.get(k) for k in keys]
    miss_idx = [i for i, s in enumerate(scores) if s is None]
    CACHE_STATS["hits"] += len(pairs) - len(miss_idx)
    CACHE_STATS["misses"] += len(miss_idx)
    if miss_idx:
        fresh = _predict(model, [pairs[i] for i in miss_idx])
        for i, s in zip(miss_idx, fresh, strict=True):
            scores[i] = s
        try:
            # One statement for the whole batch (unnest), not one per row:
            # ~500 misses per question would otherwise be ~500 round trips —
            # the cold pass measured +2.7 s/question with executemany.
            keep = [(i, s) for i, s in zip(miss_idx, fresh, strict=True) if s == s]  # NaN is never cached
            with conn.transaction():
                conn.execute(
                    "INSERT INTO rerank_scores (model, query_hash, text_hash, score) "
                    "SELECT %s, q, t, s FROM unnest(%s::text[], %s::text[], %s::real[]) AS u(q, t, s) "
                    "ON CONFLICT DO NOTHING",
                    (mk, [keys[i][0] for i, _ in keep], [keys[i][1] for i, _ in keep], [s for _, s in keep]),
                )
        except Exception:
            pass  # a cache write failure never fails a query
    return [float(s) for s in scores]  # type: ignore[arg-type]


class _Qwen3Reranker:
    """Qwen3-Reranker as a predict(pairs) scorer: a causal LM asked whether
    the document answers the query; score = P(yes) from the last-token
    logits, as the model card prescribes. Same interface as CrossEncoder."""

    _PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
               'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _INSTRUCT = ("Given a health-plan question, judge whether the passage STATES the specific fact the question asks for. "
                 "Being on the same topic is not an answer: a formulary that does not list the drug asked about, or a "
                 "directory page that does not name the provider asked about, does not answer. Shorthand, abbreviations, "
                 "and paraphrase count when the fact is the same.")

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left", local_files_only=True)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # bf16 on the GPU (the VM already runs the shipped reranker in bf16, verified
        # verdict-for-verdict on run 623); fp32 on CPU. The same scorer is measured and shipped.
        dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=dtype).eval()
        self.model.to(self.device)
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")

    # The longest chunk in the corpus is ~480 tokens and the prompt ~145 (measured
    # 2026-09-15), so 1024 truncates nothing; batches are length-sorted so padding
    # is minimal — each pair's score does not depend on what it was batched with.
    MAX_LEN = 1024

    def predict(self, pairs, batch_size: int = 8):
        texts = [self._PREFIX + f"<Instruct>: {self._INSTRUCT}\n<Query>: {q}\n<Document>: {d}" + self._SUFFIX
                 for q, d in pairs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scores = [0.0] * len(texts)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=self.MAX_LEN,
                           return_tensors="pt").to(self.device)
            probs = self._probs(self.model, enc)
            if any(s != s for s in probs):  # bf16 on the GPU overflows on some inputs: rescore those in fp32
                bad = [k for k, s in enumerate(probs) if s != s]
                enc32 = self.tok([texts[idx[k]] for k in bad], padding=True, truncation=True, max_length=self.MAX_LEN,
                                 return_tensors="pt").to(self.device)
                for k, s in zip(bad, self._probs(self._fp32(), enc32), strict=True):
                    probs[k] = s
            for j, s in zip(idx, probs, strict=True):
                scores[j] = float(s)
            if self.device == "mps":
                # The MPS allocator caches every batch shape it has seen and never
                # returns it; over thousands of variably padded batches that grew to
                # the whole machine (15 GB, swapping) and stalled the bake-off.
                self.torch.mps.empty_cache()
        return scores

    def _probs(self, model, enc) -> list[float]:
        # On a CPU with AMX (the VM) bf16 matmuls are several times faster; the
        # NaN guard above rescores any overflow in fp32. RAGLAB_RERANK_DTYPE=bf16.
        cpu_bf16 = self.device == "cpu" and RERANK_DTYPE == "bf16"
        with self.torch.no_grad(), self.torch.autocast("cpu", dtype=self.torch.bfloat16, enabled=cpu_bf16):
            logits = model(**enc).logits[:, -1, :].float()
            two = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            return self.torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()

    def _fp32(self):
        """A full-precision copy, loaded on the first NaN and kept."""
        if getattr(self, "_model32", None) is None:
            with _load_lock:
                if getattr(self, "_model32", None) is None:
                    from transformers import AutoModelForCausalLM

                    self._model32 = AutoModelForCausalLM.from_pretrained(
                        self.model.config._name_or_path, local_files_only=True, dtype=self.torch.float32).eval().to(self.device)
        return self._model32


def load_reranker(key: str):
    """The scorer for a RERANKERS entry, from local weights only."""
    spec = RERANKERS[key]
    if spec.get("kind") == "qwen3":
        return _Qwen3Reranker(spec["model"])
    from sentence_transformers import CrossEncoder

    return CrossEncoder(spec["model"], local_files_only=True, trust_remote_code=True)


_load_lock = threading.Lock()  # one loader at a time: the warm-up thread and the first request


def _get_model():
    """The shipped reranker, loaded once. The deployed app warms it on a
    thread at startup while requests may already be arriving; two threads
    importing transformers' lazy modules at the same moment left one with a
    half-initialized module (ImportError: cannot import name
    'AutoModelForCausalLM' — the v2.0.21 smoke failure, 2026-09-16)."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                _model = load_reranker(RERANKER)
    return _model


# What the cross-encoder reads (A/B, RAGLAB_RERANK_TEXT):
#   index — the whole search copy, template prefix and source header included
#   header — for call notes, minus the note header line (call id, date,
#           rep, reason, member, identity verification): metadata the
#           lexical and vector arms use, noise to a relevance judge
#   body  — header, and minus the template prefix line ("This chunk is
#           from …") for every source
#   max   — index AND the header-less body for call notes, higher score wins
RERANK_TEXT = os.environ.get("RAGLAB_RERANK_TEXT", "max")
# Bake-off axis 2 (2026-09-12): score on RE-IDENTIFIED text — vault tokens
# ([MEMBER_ID-0384]) replaced by their originals at scoring time. The
# reranker is local, so no boundary is crossed; the substitution needs the
# vault (owner connection) and is off by default.
RERANK_REIDENTIFY = os.environ.get("RAGLAB_RERANK_REIDENTIFY", "off") == "on"
_vault: dict[str, str] | None = None


def _reidentify(text: str) -> str:
    global _vault
    if _vault is None:
        try:
            conn = _cache_connection()
            with conn.transaction():
                _vault = {p: o for o, p in conn.execute("SELECT original, pseudonym FROM deid_vault").fetchall()}
        except Exception:
            _vault = {}
    if not _vault or "[" not in text:
        return text
    import re as _re
    return _re.sub(r"\[[A-Z_]+-\d{4}\]", lambda m: _vault.get(m.group(0), m.group(0)), text)
_HEADER_SOURCES = ("call_note", "appeal")
_HEADER_PREFIXES = ("record:", "CALL NOTE", "Appeal case", "Case ", "GEHA APPEALS DETERMINATION", "Case:", "Member:")


def record_body(c: Candidate) -> str:
    """A record's search copy without its header line."""
    text = c.index_text or c.content
    lines = [l for l in text.split("\n") if not l.lstrip().startswith(_HEADER_PREFIXES)]
    return "\n".join(lines).strip() or text


def _rerank_text(c: Candidate) -> str:
    text = c.index_text or c.content
    if RERANK_TEXT not in ("body", "header"):
        return text
    lines = text.split("\n")
    if RERANK_TEXT == "body" and lines and lines[0].startswith("This chunk is from"):
        lines = lines[1:]
    if c.doc_type in _HEADER_SOURCES:
        lines = [l for l in lines if not l.lstrip().startswith(_HEADER_PREFIXES)]
    return "\n".join(lines).strip() or text


def rerank_text(c: Candidate) -> str:
    text = _rerank_text(c)
    return _reidentify(text) if RERANK_REIDENTIFY else text


def rerank(
    query: str, candidates: list[Candidate], top_n: int = TOP_N_OUT, plan_seats: bool = True,
    stratify_years: tuple[int, ...] = (),
    stratify_plans: tuple[str, ...] = (),
) -> list[Candidate]:
    """Returns the top_n candidates by cross-encoder score, scores attached.

    stratify_years: when the router resolved MULTIPLE years (YoY questions),
    similarity ranking alone lets one year's near-identical chunks crowd out
    the other's. Each year is scored against its own query and the years
    are interleaved by rank (latest year first), so every routed year has
    its best chunks at the top of the list.
    """
    if not candidates:
        return []
    model = _get_model()
    # Multi-year routes: each candidate is scored against ITS year's
    # year-neutral query (router.year_neutral), not the change-worded
    # question, so "changes this year" sections lose their built-in edge.
    from raglab import router as router_mod

    by_year = router_mod.year_queries(query, stratify_years)
    # Score the search copy (shorthand expanded, boilerplate suppressed,
    # identifiers canonical) — a cross-encoder reads plain language, not rep
    # shorthand. The display copy is what the payload cites.
    pairs = [(by_year.get(c.year, query), rerank_text(c)) for c in candidates]
    # sentence-transformers >= 3 applies sigmoid activation in predict();
    # scores arrive in 0..1 already.
    scores = score_pairs(model, pairs)
    if RERANK_TEXT == "max":
        # Records (call notes) are scored twice — the whole search copy and
        # the body without the note header — and take the higher. The
        # header's member/reason tokens carry an identifier-shaped question;
        # for a question with no identifier the same header buries a
        # near-verbatim body (0.99 body-only vs 0.01 with it). Cost: a
        # second pair per call note in the pool (a handful under a member
        # context).
        idx = [i for i, c in enumerate(candidates) if c.doc_type in _HEADER_SOURCES]
        if idx:
            bodies = [(pairs[i][0], record_body(candidates[i])) for i in idx]
            for i, score in zip(idx, score_pairs(model, bodies), strict=True):
                scores[i] = max(scores[i], float(score))
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = score
    ordered = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

    if len(stratify_plans) >= 2 and len(stratify_years) < 2 and not plan_seats:
        # Every current plan covered because NOTHING was named (router
        # cover_level "all"): no plan is owed a seat — score order, with at
        # most two chunks per plan so five plans' near-identical sections
        # do not crowd the rest out. Sources with no plan compete as they
        # are (run 667: guaranteed lanes took a bulletin's seats).
        picked, per_plan = [], {}
        for c in ordered:
            if c.plan_code in stratify_plans:
                if per_plan.get(c.plan_code, 0) >= 2:
                    continue
                per_plan[c.plan_code] = per_plan.get(c.plan_code, 0) + 1
            picked.append(c)
            if len(picked) >= top_n:
                break
        return picked

    if len(stratify_plans) >= 2 and len(stratify_years) < 2:
        # The coverage rule: every covered plan's best chunk near the top, by
        # rank within its plan (scores ARE comparable here — same query).
        lanes = {code: [c for c in ordered if c.plan_code == code] for code in stratify_plans}
        # Chunks the rule does not classify — sources with no plan (bulletins,
        # letters, policies, records) — keep their own lane and compete on
        # score: the rule arranges the covered source, it never demotes the
        # others (P4: a claims bulletin fell outside every plan lane and was
        # dropped, blocking an entitled answer).
        lanes["*"] = [c for c in ordered if c.plan_code not in stratify_plans]
        # strongest lane leads; every plan lane gets exactly ONE seat (its best chunk)
        lanes = dict(sorted(lanes.items(), key=lambda kv: -(kv[1][0].rerank_score if kv[1] else -1)))
        return _interleave(ordered, lanes, top_n)

    if len(stratify_years) < 2:
        return ordered[:top_n]

    # Years were scored against different queries (see above), so their
    # scores are not comparable: interleave by RANK within each year, latest
    # year first, so the top of every year is near the top of the list.
    lanes = {year: [c for c in ordered if c.year == year] for year in sorted(stratify_years, reverse=True)}
    return _share(ordered, lanes, top_n)


def _share(ordered: list[Candidate], lanes: dict, top_n: int) -> list[Candidate]:
    """Round-robin: every lane holds an equal share of the list. For YEARS,
    where the question asked for both sides ("how did X change"), the
    asker wants each year's evidence in depth, not one chunk per year."""
    picked, picked_ids = [], set()
    while len(picked) < top_n and any(lanes.values()):
        for key in list(lanes):
            if lanes[key] and len(picked) < top_n:
                c = lanes[key].pop(0)
                if c.chunk_id not in picked_ids:
                    picked.append(c)
                    picked_ids.add(c.chunk_id)
    for c in ordered:
        if len(picked) >= top_n:
            break
        if c.chunk_id not in picked_ids:
            picked.append(c)
            picked_ids.add(c.chunk_id)
    return picked[:top_n]


def _interleave(ordered: list[Candidate], lanes: dict, top_n: int) -> list[Candidate]:
    """PLAN lanes: one seat per lane, then score order. A covered plan is a
    plausible reading of an ambiguous question and is REPRESENTED by its
    best chunk at the top; coverage never means an equal share of the list.
    (Years use _share: a change question asked for both sides.)
    The chunks outside every lane (key "*": bulletins, policies, letters —
    sources without the covered field) own no seat, but are never demoted
    below a seat they outscore (P4: a claims bulletin at 0.95 leads).
    Measured 2026-09-11 before this rule: a round-robin split the top five
    between two brochures saying the same thing (F1, T4 lost; Y1, Y8 lost a
    year) and gave a 0.47 carrier letter the third seat above 0.85 chunks."""
    seats = [lane[0] for key, lane in lanes.items() if key != "*" and lane]
    seat_ids = {c.chunk_id for c in seats}
    floor = min((c.rerank_score or 0.0) for c in seats) if seats else 0.0
    outside = [c for c in lanes.get("*", []) if (c.rerank_score or 0.0) >= floor and c.chunk_id not in seat_ids]
    head = sorted(seats + outside, key=lambda c: -(c.rerank_score or 0.0))
    picked, picked_ids = [], set()
    for c in head + ordered:
        if len(picked) >= top_n:
            break
        if c.chunk_id not in picked_ids:
            picked.append(c)
            picked_ids.add(c.chunk_id)
    return picked[:top_n]


def abstention_verdict(reranked: list[Candidate]) -> tuple[bool, float]:
    """(should_abstain, best_score). Abstain when no candidate clears its
    own source's threshold (threshold_for). Callers decide what to do with
    it; the payload maps it to insufficient_evidence."""
    if not reranked:
        return True, 0.0
    # The best score in the list, not the first chunk's: year and plan
    # interleaving put a lane's top chunk first, and it understated
    # confidence (0.33 shown with 0.67 in the list; backlog 9d).
    best = max((c.rerank_score or 0.0) for c in reranked)
    cleared = any((c.rerank_score or 0.0) >= threshold_for(c.doc_type) for c in reranked)
    return not cleared, best
