# The app image: the API and the surfaces over the pipeline's query side.
# Serve-only dependencies (no parsers, no de-identification models), the
# both rerankers' weights baked in at pinned revisions — no runtime downloads.
# Built by the release workflow on a tag and published to GHCR; the same
# image runs on the VM and, for the smoke test, on the Mac.

FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/
WORKDIR /app

# Dependencies first (cached until the lock changes): the serve core plus
# the reranker group; CPU torch on Linux comes from the lock's index fork.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --group rerank --no-install-project

# The reranker weights at the revision the baseline was measured with.
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ARG RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70
RUN HF_HUB_OFFLINE=0 /app/.venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('${RERANKER_MODEL}', revision='${RERANKER_REVISION}', \
     allow_patterns=['*.json', 'model.safetensors', 'sentencepiece.bpe.model', 'tokenizer*'])" \
    && mkdir -p "/opt/hf/hub/models--$(echo ${RERANKER_MODEL} | sed 's|/|--|g')/refs" \
    && echo -n "${RERANKER_REVISION}" > "/opt/hf/hub/models--$(echo ${RERANKER_MODEL} | sed 's|/|--|g')/refs/main"
# one weight format, no ONNX/bin copies; refs/main pinned so a load by name resolves offline

# The shipped reranker (Qwen3, a yes/no judge) at the revision the baseline was
# measured with. Both models ride in the image so the switch between them is
# RAGLAB_RERANKER and a restart, never a rebuild (deploy.sh reranker <name>).
ARG QWEN_MODEL=Qwen/Qwen3-Reranker-0.6B
ARG QWEN_REVISION=e61197ed45024b0ed8a2d74b80b4d909f1255473
RUN HF_HUB_OFFLINE=0 /app/.venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('${QWEN_MODEL}', revision='${QWEN_REVISION}', \
     allow_patterns=['*.json', 'model.safetensors', 'tokenizer*', 'merges.txt', 'vocab.json', 'chat_template.jinja'])" \
    && mkdir -p "/opt/hf/hub/models--$(echo ${QWEN_MODEL} | sed 's|/|--|g')/refs" \
    && echo -n "${QWEN_REVISION}" > "/opt/hf/hub/models--$(echo ${QWEN_MODEL} | sed 's|/|--|g')/refs/main"

# The package, the golden set and baseline the Console reports, the payload schema.
COPY src ./src
COPY eval/golden.jsonl eval/baseline.json ./eval/
COPY db ./db
COPY docs ./docs
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --group rerank

ENV PATH="/app/.venv/bin:$PATH" RAGLAB_RERANKER=qwen3-0.6b
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/login').status_code == 200 else 1)"
COPY deploy/app-entrypoint.sh /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
