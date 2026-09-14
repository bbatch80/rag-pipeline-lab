#!/bin/bash
# First start: restore the snapshot (pg_dump custom format, --no-owner) if one
# is mounted at /snapshot/raglab.dump. Runs once, by the postgres image's
# init convention, before the server accepts connections.
set -euo pipefail
if [ -f /snapshot/raglab.dump ]; then
  echo "restoring /snapshot/raglab.dump into ${POSTGRES_DB}"
  pg_restore -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" --no-owner --exit-on-error /snapshot/raglab.dump
  echo "snapshot restored"
else
  echo "no snapshot mounted: empty database (apply db/*.sql and migrations to use it)"
fi
