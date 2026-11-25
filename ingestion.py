"""
Stage 1 — Raw Dataset Ingestion & Database Schema Integrity Module (ingestion.py)
Module Owner: Parth (S24CSEU2470)

Responsibilities:
  1. Discovers raw monthly BTS flight CSV partitions, NOAA hourly weather logs, and metadata tables.
  2. Performs schema validation, column integrity checks, and raw NULL/anomaly auditing.
  3. Streams validated partitions into PostgreSQL (via DATABASE_URL) or indexed SQLite analytical
     storage, building composite B-tree indexes on (ORIGIN, CRS_DEP_TIMESTAMP) and (STATION_IATA, OBS_TIMESTAMP).
"""

from __future__ import annotations
import json
import time
from sqlalchemy import create_engine, text
import pandas as pd

from config import DATABASE_URL, RAW_DIR, OUTPUTS_DIR
from generate_datasets import run_dataset_generation

REQUIRED_FLIGHT_COLS = {
    "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_DEP_TIMESTAMP",
    "DEP_TIME", "DEP_DELAY", "DEP_DEL15", "CANCELLED", "DIVERTED", "DISTANCE"
}

REQUIRED_WEATHER_COLS = {
    "STATION_IATA", "OBS_TIMESTAMP", "TEMP_F", "DEW_POINT_F",
    "WIND_SPEED_KT", "VISIBILITY_MI", "PRECIP_IN", "PRESSURE_MB", "WEATHER_CONDITION"
}


def validate_schema(df: pd.DataFrame, expected_cols: set[str], source_name: str) -> dict:
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"[Schema Error] {source_name} missing required columns: {sorted(missing)}")
    null_counts = {col: int(df[col].isna().sum()) for col in sorted(expected_cols) if df[col].isna().sum() > 0}
    return {
        "source": source_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "null_columns": null_counts,
    }


def run_ingestion(force_regenerate: bool = False) -> dict:
    t0 = time.perf_counter()
    partition_paths = sorted(RAW_DIR.glob("bts_flights_*.csv"))
    weather_path = RAW_DIR / "noaa_weather_hourly.csv"

    if force_regenerate or not partition_paths or not weather_path.exists():
        run_dataset_generation()
        partition_paths = sorted(RAW_DIR.glob("bts_flights_*.csv"))

    engine = create_engine(DATABASE_URL)
    dialect = engine.dialect.name
    audit_reports = []

    # 1. Ingest Tertiary Metadata Registries
    airports_df = pd.read_csv(RAW_DIR / "airports_metadata.csv")
    airlines_df = pd.read_csv(RAW_DIR / "airlines_metadata.csv")
    airports_df.to_sql("airports_metadata", engine, if_exists="replace", index=False)
    airlines_df.to_sql("airlines_metadata", engine, if_exists="replace", index=False)

    # 2. Ingest Secondary NOAA Weather Observations
    weather_df = pd.read_csv(weather_path)
    audit_reports.append(validate_schema(weather_df, REQUIRED_WEATHER_COLS, weather_path.name))
    weather_df.to_sql("noaa_weather_hourly", engine, if_exists="replace", index=False, chunksize=10000)

    # 3. Stream Monthly Primary BTS Flight Partitions into Database
    total_ingested = 0
    for idx, p_path in enumerate(partition_paths):
        part_df = pd.read_csv(p_path)
        audit_reports.append(validate_schema(part_df, REQUIRED_FLIGHT_COLS, p_path.name))
        part_df.to_sql(
            "bts_flights",
            engine,
            if_exists="replace" if idx == 0 else "append",
            index=False,
            chunksize=15000,
        )
        total_ingested += len(part_df)

    # 4. Create Composite Indexes on Airport IATA & Timestamps (Chapter 2 Stage 1)
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_flights_origin_ts ON bts_flights (ORIGIN, CRS_DEP_TIMESTAMP)",
        "CREATE INDEX IF NOT EXISTS idx_flights_chrono ON bts_flights (CRS_DEP_TIMESTAMP)",
        "CREATE INDEX IF NOT EXISTS idx_weather_station_ts ON noaa_weather_hourly (STATION_IATA, OBS_TIMESTAMP)",
        "CREATE INDEX IF NOT EXISTS idx_airports_iata ON airports_metadata (IATA)",
    ]
    with engine.begin() as conn:
        for stmt in index_statements:
            conn.execute(text(stmt))

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "database_dialect": dialect,
        "database_url_masked": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        "partitions_loaded": len(partition_paths),
        "total_flights_ingested": total_ingested,
        "weather_rows_ingested": int(len(weather_df)),
        "airports_ingested": int(len(airports_df)),
        "airlines_ingested": int(len(airlines_df)),
        "indexes_created": len(index_statements),
        "elapsed_sec": elapsed,
        "partition_audits": audit_reports,
    }

    with open(OUTPUTS_DIR / "ingestion_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ingestion.py] Loaded {total_ingested:,} flight records ({len(partition_paths)} partitions) "
          f"& {len(weather_df):,} weather logs into {dialect.upper()} in {elapsed}s.")
    return summary


if __name__ == "__main__":
    run_ingestion()
