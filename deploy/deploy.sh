#!/usr/bin/env bash
# The host side of a release. Runs on the deploy host in /opt/raglab, where
# deploy/.env holds the site's configuration (never in git) and the
# snapshot lives at deploy/snapshot/raglab.dump.
#
#   deploy.sh up <tag> [--no-pull]   pull the images for <tag>, restart, remember the previous tag
#                                    (--no-pull: a rehearsal on images built locally)
#   deploy.sh rollback     restart on the previous tag (after a failed smoke test)
#   deploy.sh status       what is running
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/deploy/.env"
PREVIOUS="$ROOT/deploy/.previous-tag"
COMPOSE=(docker compose -f "$ROOT/deploy/compose.yml" --env-file "$ENV_FILE")

current_tag() { grep -E '^TAG=' "$ENV_FILE" | head -1 | cut -d= -f2 | tr -d '[:space:]'; }

set_tag() {
  local tag="$1"
  if grep -qE '^TAG=' "$ENV_FILE"; then
    sed -i.bak -E "s|^TAG=.*|TAG=${tag}|" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
  else
    echo "TAG=${tag}" >> "$ENV_FILE"
  fi
}

# Every release image carries the reranker weights (~4.4 GB); fifteen of them
# filled the host's disk and a deploy failed mid-pull (2026-09-15). Before
# pulling a new tag, keep only the images still needed: the tags passed in
# (the running one, so a failed smoke test can still roll back) — the
# registry holds every release for anything older.
prune_images() {
  local keep=" $* "
  { docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null || true; } | { grep -F 'rag-pipeline-lab:' || true; } | while read -r image; do
    local t="${image##*:}"
    case "$keep" in *" $t "*) ;; *) docker rmi -f "$image" >/dev/null 2>&1 || true ;; esac
  done
  docker image prune -f >/dev/null 2>&1 || true
}

case "${1:-}" in
  up)
    tag="${2:?tag}"
    [ -f "$ENV_FILE" ] || { echo "no $ENV_FILE on this host (copy deploy/.env.example and fill it in)"; exit 2; }
    previous="$(current_tag)"
    echo "deploying ${tag} (previous: ${previous:-none})"
    echo "${previous:-}" > "$PREVIOUS"
    set_tag "$tag"
    prune_images "$tag" "${previous:-}"
    if [ "${3:-}" != "--no-pull" ]; then "${COMPOSE[@]}" pull --quiet app db; fi
    "${COMPOSE[@]}" up -d --remove-orphans
    "${COMPOSE[@]}" ps --format '{{.Name}} {{.Status}}'
    ;;
  rollback)
    previous="$(cat "$PREVIOUS" 2>/dev/null || true)"
    [ -n "$previous" ] || { echo "no previous tag recorded: nothing to roll back to"; exit 2; }
    echo "rolling back to ${previous}"
    set_tag "$previous"
    "${COMPOSE[@]}" up -d --remove-orphans
    "${COMPOSE[@]}" ps --format '{{.Name}} {{.Status}}'
    ;;
  prune)
    prune_images "$(current_tag)" "$(cat "$PREVIOUS" 2>/dev/null || true)"
    docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -F 'rag-pipeline-lab' || true
    ;;
  status)
    echo "TAG=$(current_tag)  previous=$(cat "$PREVIOUS" 2>/dev/null || echo none)"
    "${COMPOSE[@]}" ps --format '{{.Name}} {{.Status}}'
    ;;
  *)
    echo "usage: deploy.sh up <tag> | rollback | status"; exit 2 ;;
esac
