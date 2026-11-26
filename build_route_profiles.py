"""
Extracts real 2024 U.S. DOT BTS scheduled flight timetables & empirical hourly delay profiles
per (ORIGIN, DEST, CARRIER) and (ORIGIN, DEST) from the 7.08M-row DuckDB warehouse.
"""

from __future__ import annotations
import json
import duckdb
from config import AIRPORTS_REGISTRY, OUTPUTS_DIR, DUCKDB_PATH


def build_route_profiles() -> dict:
    hubs = tuple(a["IATA"] for a in AIRPORTS_REGISTRY)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        # 1. Exact (ORIGIN, DEST, CARRIER) scheduled hours & modal departure minute
        df_odc = con.execute(
            f"""
            SELECT
                ORIGIN, DEST, OP_UNIQUE_CARRIER AS CARRIER,
                CAST(CRS_DEP_TIME / 100 AS INT) AS hr,
                COUNT(*) AS n,
                CAST(MODE(CRS_DEP_TIME % 100) AS INT) AS typ_min,
                ROUND(AVG(CASE WHEN DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 1) AS emp_pct
            FROM bts_flights
            WHERE CANCELLED = 0 AND DIVERTED = 0 AND DEP_DEL15 IS NOT NULL
              AND ORIGIN IN {hubs} AND DEST IN {hubs}
              AND CRS_DEP_TIME >= 0 AND CRS_DEP_TIME < 2400
            GROUP BY 1, 2, 3, 4
            HAVING COUNT(*) >= 10
            """
        ).fetchdf()

        # 2. Corridor-level (ORIGIN, DEST) across all airlines
        df_od = con.execute(
            f"""
            SELECT
                ORIGIN, DEST,
                CAST(CRS_DEP_TIME / 100 AS INT) AS hr,
                COUNT(*) AS n,
                CAST(MODE(CRS_DEP_TIME % 100) AS INT) AS typ_min,
                ROUND(AVG(CASE WHEN DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 1) AS emp_pct
            FROM bts_flights
            WHERE CANCELLED = 0 AND DIVERTED = 0 AND DEP_DEL15 IS NOT NULL
              AND ORIGIN IN {hubs} AND DEST IN {hubs}
              AND CRS_DEP_TIME >= 0 AND CRS_DEP_TIME < 2400
            GROUP BY 1, 2, 3
            HAVING COUNT(*) >= 12
            """
        ).fetchdf()

        # 3. Origin hub hourly profile fallback (for hub pairs without nonstop flights)
        df_o = con.execute(
            f"""
            SELECT
                ORIGIN,
                CAST(CRS_DEP_TIME / 100 AS INT) AS hr,
                CAST(MODE(CRS_DEP_TIME % 100) AS INT) AS typ_min,
                ROUND(AVG(CASE WHEN DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 1) AS emp_pct
            FROM bts_flights
            WHERE CANCELLED = 0 AND DIVERTED = 0 AND DEP_DEL15 IS NOT NULL
              AND ORIGIN IN {hubs}
              AND CRS_DEP_TIME >= 500 AND CRS_DEP_TIME < 2359
            GROUP BY 1, 2
            HAVING COUNT(*) >= 50
            """
        ).fetchdf()
    finally:
        con.close()

    odc_map: dict[str, dict] = {}
    for _, r in df_odc.iterrows():
        key = f"{r['ORIGIN']}_{r['DEST']}_{r['CARRIER']}"
        odc_map.setdefault(key, {})[str(int(r["hr"]))] = [int(r["typ_min"]), float(r["emp_pct"])]

    # Keep only ODC entries with at least 2 distinct operating hours
    odc_map = {k: v for k, v in odc_map.items() if len(v) >= 2}

    od_map: dict[str, dict] = {}
    for _, r in df_od.iterrows():
        key = f"{r['ORIGIN']}_{r['DEST']}"
        od_map.setdefault(key, {})[str(int(r["hr"]))] = [int(r["typ_min"]), float(r["emp_pct"])]

    o_map: dict[str, dict] = {}
    for _, r in df_o.iterrows():
        o_map.setdefault(r["ORIGIN"], {})[str(int(r["hr"]))] = [int(r["typ_min"]), float(r["emp_pct"])]

    payload = {"odc": odc_map, "od": od_map, "origin": o_map}
    out_path = OUTPUTS_DIR / "route_slot_profiles.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"[build_route_profiles] Saved {len(odc_map):,} airline-routes & {len(od_map):,} nonstop corridors "
        f"({round(out_path.stat().st_size / 1024, 1)} KB)"
    )
    return payload


if __name__ == "__main__":
    build_route_profiles()
