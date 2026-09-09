"""Lane 2: Synthea claims served from Snowflake under row access policies
and dynamic masking, audited by ACCESS_HISTORY.

Postgres (synthea schema) stays the system of record; Snowflake holds a
governed serving copy, fully rebuilt by `raglab snowflake-setup` — a lapsed
trial is recreated in minutes. Snowflake data is never mutated in place.
"""

import gzip
import os
from pathlib import Path

import psycopg

from raglab import config

SETUP_SQL_PATH = config.REPO_ROOT / "db" / "snowflake" / "setup.sql"
STAGE_DIR = config.REPO_ROOT / "data" / "snowflake_stage"

ROLES = ("CLAIMS_EXAMINER", "PSHB_EXAMINER", "CARE_MANAGER", "ACTUARY")

# LOB assignment is synthetic (Synthea has no plan codes): a deterministic
# hash of the patient id splits ~80/20 FEHB/PSHB, denormalized onto both
# tables so the row access policy needs no join.
LOB_EXPR = (
    "CASE WHEN abs(hashtext({col})) % 5 = 0 THEN 'PSHB' ELSE 'FEHB' END"
)

EXPORTS = {
    "patients": (
        "SELECT id, member_id, mrn, birthdate, ssn, first, last, gender, race, ethnicity, "
        "city, state, zip, " + LOB_EXPR.format(col="id")
        + " FROM synthea.patients"
    ),
    "enrollment": (
        "SELECT patient, member_id, year, line_of_business, plan_code, plan_option, "
        "tier, enrollment_code FROM synthea.enrollment"
    ),
    "claim_lines": (
        "SELECT id, patient, start::date, encounterclass, code, description, "
        "reasondescription, base_encounter_cost, total_claim_cost, "
        "payer_coverage, " + LOB_EXPR.format(col="patient")
        + " FROM synthea.encounters"
    ),
}


def connect(role: str | None = None, bootstrap: bool = False):
    """bootstrap=True skips USE statements — on a fresh trial nothing exists
    until setup.sql runs (which creates and USEs everything itself)."""
    import snowflake.connector

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
    )
    if not bootstrap:
        cur = conn.cursor()
        if role:
            cur.execute(f"USE ROLE {role}")
            # Pin the session to ONE role, like a production app connection.
            # Trial users default to secondary roles ALL, which would make
            # IS_ROLE_IN_SESSION() true for every granted role and fire every
            # masking policy for everyone.
            cur.execute("USE SECONDARY ROLES NONE")
        cur.execute("USE WAREHOUSE RAGLAB_WH")
        cur.execute("USE SCHEMA RAGLAB.CLAIMS")
    return conn


def export_csvs(pg: psycopg.Connection) -> dict[str, int]:
    """Postgres -> gzipped CSVs in data/snowflake_stage/."""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, sql in EXPORTS.items():
        path = STAGE_DIR / f"{name}.csv.gz"
        rows = 0
        with gzip.open(path, "wb") as out:
            with pg.cursor().copy(
                f"COPY ({sql}) TO STDOUT WITH (FORMAT csv)"
            ) as copy:
                for chunk in copy:
                    out.write(bytes(chunk))
                    rows += bytes(chunk).count(b"\n")
        counts[name] = rows
    return counts


def run_setup(sf_cursor) -> int:
    """Execute setup.sql statement by statement; returns statement count.
    Fails loudly on the first masking-policy DDL if the account is not
    Enterprise edition."""
    # Strip comment lines BEFORE splitting on ';' — a semicolon inside a
    # comment would otherwise shear a statement in two.
    sql = "\n".join(
        line for line in SETUP_SQL_PATH.read_text().splitlines()
        if not line.strip().startswith("--")
    )
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for statement in statements:
        sf_cursor.execute(statement)
    return len(statements)


def load(sf_cursor) -> dict[str, int]:
    """PUT the exported CSVs to the internal stage and COPY INTO tables."""
    counts = {}
    for name in EXPORTS:
        path = STAGE_DIR / f"{name}.csv.gz"
        sf_cursor.execute(
            f"PUT file://{path} @LOAD_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        )
        # DELETE, not TRUNCATE: tables carry a row access policy, and
        # ACCOUNTADMIN is entitled to every row.
        sf_cursor.execute(f"DELETE FROM {name}")
        sf_cursor.execute(
            f"COPY INTO {name} FROM @LOAD_STAGE/{path.name} "
            "FILE_FORMAT = (TYPE=CSV FIELD_OPTIONALLY_ENCLOSED_BY='\"' "
            "EMPTY_FIELD_AS_NULL=TRUE) PURGE=TRUE"
        )
        counts[name] = sf_cursor.execute(
            f"SELECT count(*) FROM {name}"
        ).fetchone()[0]
    return counts


def grant_roles_to_user(sf_cursor, user: str) -> None:
    for role in ROLES:
        sf_cursor.execute(f"GRANT ROLE {role} TO USER {user}")


DEMO_QUERY = """
SELECT FIRST_NAME, LAST_NAME, SSN, BIRTHDATE, DESCRIPTION,
       TOTAL_CLAIM_COST, LINE_OF_BUSINESS
FROM CLAIM_DETAIL
ORDER BY SERVICE_DATE DESC
LIMIT 5
"""


def verify(sf_conn) -> dict:
    """The three-persona entitlement test (plus the row-scoped examiner):
    same SELECT, per-role result shapes, asserted not eyeballed."""
    cur = sf_conn.cursor()
    report = {}
    for role in ("ACCOUNTADMIN",) + ROLES:
        cur.execute(f"USE ROLE {role}")
        cur.execute("USE SECONDARY ROLES NONE")
        cur.execute("USE WAREHOUSE RAGLAB_WH")
        cur.execute("USE SCHEMA RAGLAB.CLAIMS")
        rows = cur.execute(DEMO_QUERY).fetchall()
        stats = cur.execute(
            "SELECT count(*), count(DISTINCT LINE_OF_BUSINESS), "
            "count(SSN), count(TOTAL_CLAIM_COST), "
            "count(DISTINCT PATIENT_ID) FROM CLAIM_DETAIL"
        ).fetchone()
        sample = rows[0] if rows else None
        report[role] = {
            "rows": stats[0],
            "lobs": stats[1],
            "ssn_visible": stats[2] > 0,
            "cost_visible": stats[3] > 0,
            "distinct_patients": stats[4],
            "sample_name": sample[0] if sample else None,
        }
    cur.execute("USE ROLE ACCOUNTADMIN")
    return report


# The semantic contract: member data is reachable ONLY through these named,
# parameterized queries — grain and meaning authored once, here. No freeform
# SQL crosses the tool boundary (an agent-written query is an injection
# surface and a wrong-number machine). Warehouse policies still apply on
# top: what each role sees inside these results is masked/trimmed by
# Snowflake, not by this code.
NAMED_QUERIES = {
    "member_claims_summary": {
        "doc": ("One row per member matched by member ID or by name: claim-line "
                "count, total claim cost, payer coverage, and service-date span."),
        "params": {"member_id": type(None), "last_name": type(None), "first_name": type(None)},
        "sql": """
            SELECT MEMBER_ID, PATIENT_ID, FIRST_NAME, LAST_NAME,
                   COUNT(*) AS CLAIM_LINES,
                   SUM(TOTAL_CLAIM_COST) AS TOTAL_COST,
                   SUM(PAYER_COVERAGE) AS PAYER_COVERAGE,
                   MIN(SERVICE_DATE) AS FIRST_SERVICE,
                   MAX(SERVICE_DATE) AS LAST_SERVICE
            FROM CLAIM_DETAIL
            WHERE (%(member_id)s IS NOT NULL AND MEMBER_ID = %(member_id)s)
               OR (%(member_id)s IS NULL AND UPPER(LAST_NAME) = UPPER(%(last_name)s)
                   AND (%(first_name)s IS NULL OR UPPER(FIRST_NAME) = UPPER(%(first_name)s)))
            GROUP BY MEMBER_ID, PATIENT_ID, FIRST_NAME, LAST_NAME
            ORDER BY CLAIM_LINES DESC
        """,
    },
    "member_recent_claims": {
        "doc": ("One row per claim line for the member (by member ID or last "
                "name), newest first: service date, encounter class, description, costs."),
        "params": {"member_id": type(None), "last_name": type(None), "limit": int},
        "sql": """
            SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, SERVICE_DATE, ENCOUNTER_CLASS,
                   DESCRIPTION, REASON, TOTAL_CLAIM_COST, PAYER_COVERAGE
            FROM CLAIM_DETAIL
            WHERE (%(member_id)s IS NOT NULL AND MEMBER_ID = %(member_id)s)
               OR (%(member_id)s IS NULL AND UPPER(LAST_NAME) = UPPER(%(last_name)s))
            ORDER BY SERVICE_DATE DESC
            LIMIT %(limit)s
        """,
    },
    "member_enrollment": {
        "doc": ("The member's enrollment by plan year: line of business, plan, "
                "option, tier, enrollment code — what Agent Assist uses to pick "
                "the member's brochure."),
        "params": {"member_id": str},
        "sql": """
            SELECT MEMBER_ID, YEAR, LINE_OF_BUSINESS, PLAN_CODE, PLAN_OPTION, TIER, ENROLLMENT_CODE
            FROM ENROLLMENT
            WHERE MEMBER_ID = %(member_id)s
            ORDER BY YEAR
        """,
    },
    "cost_by_condition": {
        "doc": ("Aggregate across members: claim-line count, average and "
                "total cost, grouped by encounter description matching the "
                "pattern. De-identified by role policy where applicable."),
        "params": {"description_like": str},
        "sql": """
            SELECT DESCRIPTION,
                   COUNT(*) AS CLAIM_LINES,
                   COUNT(DISTINCT PATIENT_ID) AS MEMBERS,
                   AVG(TOTAL_CLAIM_COST) AS AVG_COST,
                   SUM(TOTAL_CLAIM_COST) AS TOTAL_COST
            FROM CLAIM_DETAIL
            WHERE DESCRIPTION ILIKE %(description_like)s
            GROUP BY DESCRIPTION
            ORDER BY CLAIM_LINES DESC
        """,
    },
}


# Which result columns each role's masking policies alter (NULLed, hashed,
# or truncated). Reported per response so a consumer can tell policy-NULL
# from data-NULL — Synthea leaves e.g. REASON genuinely empty on many rows.
MASKED_FOR_ROLE = {
    "CARE_MANAGER": {"BASE_COST", "TOTAL_COST", "TOTAL_CLAIM_COST",
                     "PAYER_COVERAGE", "AVG_COST"},
    "ACTUARY": {"SSN", "FIRST_NAME", "LAST_NAME", "BIRTHDATE", "MEMBER_ID", "MRN"},
}


def run_named_query(sf_conn, query_name: str, params: dict) -> dict:
    """Execute a catalog query with bound parameters; returns column names,
    rows, the query's documented meaning, and which columns the session
    role's policies masked. Unknown names are refused — the catalog IS the
    surface area."""
    if query_name not in NAMED_QUERIES:
        return {
            "status": "unknown_query",
            "known_queries": {
                name: spec["doc"] for name, spec in NAMED_QUERIES.items()
            },
        }
    spec = NAMED_QUERIES[query_name]
    bound = {"member_id": None, "last_name": None, "first_name": None, "limit": 20}
    bound.update({k: v for k, v in params.items() if v is not None})
    cur = sf_conn.cursor()
    role = cur.execute("SELECT CURRENT_ROLE()").fetchone()[0]
    cur.execute(spec["sql"], bound)
    columns = [d[0] for d in cur.description]
    rows = [
        [v.isoformat() if hasattr(v, "isoformat") else
         float(v) if hasattr(v, "as_tuple") else v for v in row]
        for row in cur.fetchall()
    ]
    return {
        "status": "ok",
        "query_name": query_name,
        "meaning": spec["doc"],
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "masked_columns": sorted(
            MASKED_FOR_ROLE.get(role, set()) & set(columns)
        ),
    }


def access_history_peek(sf_cursor, limit: int = 5) -> list[tuple]:
    """Platform audit, consumed not built (the deliberate contrast with
    Lane 1's hand-built disclosure log). ACCOUNT_USAGE has up to ~3h
    ingestion latency — early runs legitimately return nothing."""
    return sf_cursor.execute(
        """
        SELECT QUERY_START_TIME, USER_NAME,
               OBJ.VALUE:"objectName"::STRING AS OBJECT_NAME
        FROM SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY,
             LATERAL FLATTEN(BASE_OBJECTS_ACCESSED) OBJ
        WHERE OBJ.VALUE:"objectName"::STRING LIKE 'RAGLAB.CLAIMS.%'
        ORDER BY QUERY_START_TIME DESC
        LIMIT """ + str(int(limit))
    ).fetchall()
