#!/bin/sh
# Wait for the database, refresh the Console's scorecard from the eval store
# (the file is not in the image; the runs are in the snapshot), then serve.
set -eu
python - <<'PY'
import os, time, psycopg
url = os.environ["RAGLAB_DATABASE_URL"]
for attempt in range(300):  # a first-start restore can take minutes
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"waiting for the database ({exc.__class__.__name__})", flush=True)
        time.sleep(2)
else:
    raise SystemExit("database never became reachable")
PY
raglab seed-identity || echo "identity not seeded (the accounts stay as they were)"
raglab dashboard || echo "dashboard not rendered (the Console will say so)"
exec raglab serve --host 0.0.0.0 --port 8000
