# The vector-lane database: pgvector's pg17 image plus pg_textsearch (BM25).
# Built by `docker compose up -d --build` locally and by CI.
FROM pgvector/pgvector:pg17
ARG PG_TEXTSEARCH_VERSION=1.4.0
ARG TARGETARCH
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends curl unzip ca-certificates; \
    curl -fsSL -o /tmp/pgts.zip \
      "https://github.com/timescale/pg_textsearch/releases/download/v${PG_TEXTSEARCH_VERSION}/pg-textsearch-v${PG_TEXTSEARCH_VERSION}-pg17-${TARGETARCH}.zip"; \
    cd /tmp && unzip -o pgts.zip && dpkg -i ./pg-textsearch-postgresql-17_*.deb; \
    rm -rf /tmp/pgts.zip /tmp/*.deb /var/lib/apt/lists/*
# BM25 needs a preloaded library; everything else about the server is stock.
CMD ["postgres", "-c", "shared_preload_libraries=pg_textsearch"]
