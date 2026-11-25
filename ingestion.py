"""
Stage 1 — Out-of-Core Multi-Million-Row Dataset Ingestion & Indexing (ingestion.py)
Module Owner: Parth (S24CSEU2470)

Responsibilities:
  1. Discovers all raw monthly U.S. DOT BTS flight CSV partitions (~7.0M rows across 12 months),
     NOAA ASOS hourly surface weather (~525K rows across 60 U.S. hubs), and metadata registries.
  2. Streams multi-million-row CSV partitions out-of-core into DuckDB columnar analytical storage
     (data/db/aviation_warehouse.duckdb) — avoiding out-of-memory crashes on 6M+ flight tables.
  3. Performs SQL schema integrity validation, NULL-record auditing per monthly partition, and
     creates composite indexes on (ORIGIN, CRS_DEP_TIMESTAMP) and (STATION_IATA, OBS_TIMESTAMP).
"""

from __future__ import annotations
import json
import time
import duckdb

from config import DUCKDB_PATH, RAW_DIR, OUTPUTS_DIR

REQUIRED_FLIGHT_COLS = {
    "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_DEP_TIMESTAMP",
    "DEP_TIME", "DEP_DELAY", "DEP_DEL15", "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "DISTANCE"
}

REQUIRED_WEATHER_COLS = {
    "STATION_IATA", "OBS_TIMESTAMP", "TEMP_F", "DEW_POINT_F",
    "WIND_SPEED_KT", "VISIBILITY_MI", "PRECIP_IN", "PRESSURE_MB", "WEATHER_CONDITION"
}


def run_ingestion(force_regenerate: bool = False) -> dict:
    t0 = time.perf_counter()
    partition_paths = sorted(RAW_DIR.glob("bts_flights_2024_*.csv"))
    if not partition_paths:
        partition_paths = sorted(RAW_DIR.glob("bts_flights_*.csv"))
    weather_path = RAW_DIR / "noaa_weather_hourly.csv"

    if force_regenerate or not partition_paths or not weather_path.exists():
        from download_real_bigdata import main as download_main
        download_main()
        partition_paths = sorted(RAW_DIR.glob("bts_flights_2024_*.csv"))

    # Remove stale lock/file before full rebuild
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        # 1. Ingest Tertiary Metadata Registries
        con.execute(
            f"CREATE OR REPLACE TABLE airports_metadata AS SELECT * FROM read_csv_auto('{ (RAW_DIR / 'airports_metadata.csv').as_posix() }', header=true)"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE airlines_metadata AS SELECT * FROM read_csv_auto('{ (RAW_DIR / 'airlines_metadata.csv').as_posix() }', header=true)"
        )

        # 2. Ingest Secondary NOAA ASOS Hourly Surface Weather Observations
        con.execute(
            f"""
            CREATE OR REPLACE TABLE noaa_weather_hourly AS
            SELECT
                CAST(STATION_IATA AS VARCHAR) AS STATION_IATA,
                CAST(OBS_TIMESTAMP AS VARCHAR) AS OBS_TIMESTAMP,
                CAST(TEMP_F AS DOUBLE) AS TEMP_F,
                CAST(DEW_POINT_F AS DOUBLE) AS DEW_POINT_F,
                CAST(WIND_SPEED_KT AS DOUBLE) AS WIND_SPEED_KT,
                CAST(VISIBILITY_MI AS DOUBLE) AS VISIBILITY_MI,
                CAST(PRECIP_IN AS DOUBLE) AS PRECIP_IN,
                CAST(PRESSURE_MB AS DOUBLE) AS PRESSURE_MB,
                CAST(WEATHER_CONDITION AS VARCHAR) AS WEATHER_CONDITION
            FROM read_csv_auto('{ weather_path.as_posix() }', header=true)
            """
        )
        wx_cols = {r[0] for r in con.execute("DESCRIBE noaa_weather_hourly").fetchall()}
        if REQUIRED_WEATHER_COLS - wx_cols:
            raise ValueError(f"Weather schema missing: {REQUIRED_WEATHER_COLS - wx_cols}")

        # 3. Stream All Monthly BTS CSV Partitions Out-of-Core into DuckDB
        glob_pattern = (RAW_DIR / "bts_flights_2024_*.csv").as_posix()
        if not list(RAW_DIR.glob("bts_flights_2024_*.csv")):
            glob_pattern = (RAW_DIR / "bts_flights_*.csv").as_posix()

        con.execute(
            f"""
            CREATE OR REPLACE TABLE bts_flights AS
            SELECT
                CAST(FL_DATE AS VARCHAR) AS FL_DATE,
                CAST(OP_UNIQUE_CARRIER AS VARCHAR) AS OP_UNIQUE_CARRIER,
                CAST(TAIL_NUM AS VARCHAR) AS TAIL_NUM,
                TRY_CAST(OP_CARRIER_FL_NUM AS INTEGER) AS OP_CARRIER_FL_NUM,
                CAST(ORIGIN AS VARCHAR) AS ORIGIN,
                CAST(DEST AS VARCHAR) AS DEST,
                TRY_CAST(CRS_DEP_TIME AS INTEGER) AS CRS_DEP_TIME,
                CAST(CRS_DEP_TIMESTAMP AS VARCHAR) AS CRS_DEP_TIMESTAMP,
                TRY_CAST(DEP_TIME AS DOUBLE) AS DEP_TIME,
                TRY_CAST(DEP_DELAY AS DOUBLE) AS DEP_DELAY,
                TRY_CAST(DEP_DEL15 AS INTEGER) AS DEP_DEL15,
                COALESCE(TRY_CAST(CANCELLED AS INTEGER), 0) AS CANCELLED,
                CAST(CANCELLATION_CODE AS VARCHAR) AS CANCELLATION_CODE,
                COALESCE(TRY_CAST(DIVERTED AS INTEGER), 0) AS DIVERTED,
                TRY_CAST(CRS_ELAPSED_TIME AS DOUBLE) AS CRS_ELAPSED_TIME,
                TRY_CAST(DISTANCE AS DOUBLE) AS DISTANCE
            FROM read_csv_auto('{glob_pattern}', header=true, union_by_name=true)
            """
        )

        fl_cols = {r[0] for r in con.execute("DESCRIBE bts_flights").fetchall()}
        if REQUIRED_FLIGHT_COLS - fl_cols:
            raise ValueError(f"BTS schema missing: {REQUIRED_FLIGHT_COLS - fl_cols}")

        # 4. Build Composite Analytical Indexes
        con.execute("CREATE INDEX IF NOT EXISTS idx_flights_origin_ts ON bts_flights (ORIGIN, CRS_DEP_TIMESTAMP)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_weather_station_ts ON noaa_weather_hourly (STATION_IATA, OBS_TIMESTAMP)")

        # 5. Collect Partition Integrity & NULL Audit Telemetry
        total_flights = int(con.execute("SELECT COUNT(*) FROM bts_flights").fetchone()[0])
        wx_rows = int(con.execute("SELECT COUNT(*) FROM noaa_weather_hourly").fetchone()[0])
        ap_rows = int(con.execute("SELECT COUNT(*) FROM airports_metadata").fetchone()[0])
        al_rows = int(con.execute("SELECT COUNT(*) FROM airlines_metadata").fetchone()[0])

        part_rows = con.execute(
            """
            SELECT
                SUBSTR(FL_DATE, 1, 7) AS month_partition,
                COUNT(*) AS rows,
                SUM(CASE WHEN CANCELLED = 1 THEN 1 ELSE 0 END) AS cancelled_rows,
                SUM(CASE WHEN CANCELLED = 0 AND DEP_DELAY IS NULL THEN 1 ELSE 0 END) AS null_delay_rows
            FROM bts_flights
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf().to_dict(orient="records")
    finally:
        con.close()

    raw_size_mb = round(sum(p.stat().st_size for p in partition_paths) / (1024 * 1024) + weather_path.stat().st_size / (1024 * 1024), 1)
    elapsed = round(time.perf_counter() - t0, 2)

    summary = {
        "database_engine": "DuckDB Columnar Analytical Warehouse (PostgreSQL DDL Compatible)",
        "warehouse_file": DUCKDB_PATH.name,
        "dataset_source": "Official U.S. DOT BTS TranStats (2024 Full Year) + NOAA/NWS ASOS Hourly METAR",
        "raw_disk_footprint_mb": raw_size_mb,
        "partitions_loaded": len(partition_paths),
        "total_flights_ingested": total_flights,
        "weather_rows_ingested": wx_rows,
        "airports_ingested": ap_rows,
        "airlines_ingested": al_rows,
        "indexes_created": 2,
        "elapsed_sec": elapsed,
        "monthly_partition_audits": part_rows,
    }

    with open(OUTPUTS_DIR / "ingestion_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[ingestion.py] Streamed {total_flights:,} real BTS flights ({len(partition_paths)} monthly CSVs, "
        f"{raw_size_mb} MB) & {wx_rows:,} NOAA ASOS weather rows into DuckDB in {elapsed}s."
    )
    return summary


if __name__ == "__main__":
    run_ingestion()
