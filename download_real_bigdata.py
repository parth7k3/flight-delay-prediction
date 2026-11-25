"""
Real Multi-Million-Row Big Data Acquisition Script (download_real_bigdata.py)
Course: Big Data Analytics and Business Intelligence (CBCA 293) — Bennett University

Downloads and extracts the three authentic empirical datasets specified in Chapter 2:
  1. PRIMARY TIER: Full-Year 2024 U.S. DOT Bureau of Transportation Statistics (BTS)
     On-Time Performance Logs across all 12 monthly partitions (~7,000,000+ real domestic flights)
     from https://transtats.bts.gov/PREZIP/
  2. SECONDARY TIER: Full-Year 2024 NOAA / NWS Automated Surface Observing System (ASOS)
     Hourly Airport Weather Observations across 60 major U.S. hubs (~525,000+ real METAR readings)
     from https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py, converted from UTC to each
     airport's exact IANA local wall-clock timezone so hourly observations align with BTS CRSDepTime.
  3. TERTIARY TIER: U.S. Commercial Airport & Reporting Airline Metadata Registries.
"""

from __future__ import annotations
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import requests
import urllib3

from config import RAW_DIR, AIRPORTS_REGISTRY, AIRLINES_REGISTRY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BTS_YEAR = 2024
BTS_BASE_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
ASOS_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

BTS_SOURCE_COLS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
]


def save_metadata_registries() -> None:
    ap_df = pd.DataFrame(AIRPORTS_REGISTRY)
    al_df = pd.DataFrame(AIRLINES_REGISTRY)
    ap_df.to_csv(RAW_DIR / "airports_metadata.csv", index=False)
    al_df.to_csv(RAW_DIR / "airlines_metadata.csv", index=False)
    print(f"[Tier 3] Saved {len(ap_df)} U.S. hub airports & {len(al_df)} reporting airlines to {RAW_DIR}")


def _download_single_bts_month(year: int, month: int) -> tuple[int, int, str]:
    out_csv = RAW_DIR / f"bts_flights_{year}_{month:02d}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 15_000_000:
        # Count lines quickly
        with open(out_csv, "rb") as f:
            n_rows = sum(buf.count(b"\n") for buf in iter(lambda: f.raw.read(1024 * 1024), b"")) - 1
        return month, n_rows, str(out_csv)

    url = BTS_BASE_URL.format(year=year, month=month)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=180, verify=False)
            resp.raise_for_status()
            break
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed downloading BTS {year}-{month:02d}: {exc}") from exc
            time.sleep(2.0 * (attempt + 1))

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_member = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_member) as raw_stream:
            convert_opts = pacsv.ConvertOptions(
                include_columns=BTS_SOURCE_COLS,
                strings_can_be_null=True,
                column_types={
                    "FlightDate": pa.string(),
                    "Reporting_Airline": pa.string(),
                    "Tail_Number": pa.string(),
                    "Flight_Number_Reporting_Airline": pa.string(),
                    "Origin": pa.string(),
                    "Dest": pa.string(),
                    "CRSDepTime": pa.string(),
                    "DepTime": pa.string(),
                    "DepDelay": pa.float32(),
                    "DepDel15": pa.float32(),
                    "Cancelled": pa.float32(),
                    "CancellationCode": pa.string(),
                    "Diverted": pa.float32(),
                    "CRSElapsedTime": pa.float32(),
                    "Distance": pa.float32(),
                },
            )
            read_opts = pacsv.ReadOptions(block_size=16 * 1024 * 1024)
            table = pacsv.read_csv(raw_stream, read_options=read_opts, convert_options=convert_opts)

    df = table.to_pandas()
    # Standardize column names to match pipeline schema + build CRS_DEP_TIMESTAMP
    crs_str = df["CRSDepTime"].fillna("0000").astype(str).str.strip().str.zfill(4)
    # Handle '2400' edge case in BTS -> '2359'
    crs_str = np.where(crs_str == "2400", "2359", crs_str)
    crs_series = pd.Series(crs_str, index=df.index)
    hh = crs_series.str.slice(0, 2)
    mm = crs_series.str.slice(2, 4)

    clean_part = pd.DataFrame({
        "FL_DATE": df["FlightDate"],
        "OP_UNIQUE_CARRIER": df["Reporting_Airline"],
        "TAIL_NUM": df["Tail_Number"].fillna(""),
        "OP_CARRIER_FL_NUM": pd.to_numeric(df["Flight_Number_Reporting_Airline"], errors="coerce").fillna(0).astype(np.int32),
        "ORIGIN": df["Origin"],
        "DEST": df["Dest"],
        "CRS_DEP_TIME": pd.to_numeric(crs_series, errors="coerce").fillna(0).astype(np.int32),
        "CRS_DEP_TIMESTAMP": df["FlightDate"].astype(str) + " " + hh + ":" + mm + ":00",
        "DEP_TIME": pd.to_numeric(df["DepTime"], errors="coerce"),
        "DEP_DELAY": df["DepDelay"],
        "DEP_DEL15": df["DepDel15"],
        "CANCELLED": df["Cancelled"].fillna(0).astype(np.int8),
        "CANCELLATION_CODE": df["CancellationCode"].fillna(""),
        "DIVERTED": df["Diverted"].fillna(0).astype(np.int8),
        "CRS_ELAPSED_TIME": df["CRSElapsedTime"],
        "DISTANCE": df["Distance"],
    })
    clean_part.to_csv(out_csv, index=False)
    return month, len(clean_part), str(out_csv)


def download_all_bts_partitions(year: int = BTS_YEAR) -> int:
    # Remove old small 90k synthetic CSVs if present
    for old_f in RAW_DIR.glob("bts_flights_2025_*.csv"):
        old_f.unlink(missing_ok=True)

    print(f"[Tier 1] Downloading all 12 monthly U.S. DOT BTS flight partitions for {year} from transtats.bts.gov...")
    t0 = time.perf_counter()
    total_rows = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_download_single_bts_month, year, m): m for m in range(1, 13)}
        for fut in as_completed(futures):
            m, rows, path = fut.result()
            total_rows += rows
            print(f"  -> Saved Partition {year}-{m:02d}: {rows:,} real flights ({path})")

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"[Tier 1] Completed 12 BTS monthly partitions: {total_rows:,} total real domestic flight records in {elapsed}s.")
    return total_rows


def _classify_metar_wx(wxcodes: pd.Series, temp_f: pd.Series, vis_mi: pd.Series, precip_in: pd.Series) -> np.ndarray:
    wx = wxcodes.fillna("").astype(str).str.upper()
    is_ts = wx.str.contains("TS")
    is_snow_ice = wx.str.contains("SN|FZ|PL|IC|BLSN|GR") | ((temp_f <= 32.5) & (precip_in >= 0.02))
    is_fog = wx.str.contains("FG") | (vis_mi <= 1.0)
    is_rain = wx.str.contains("RA|DZ|SH") | (precip_in >= 0.01)

    return np.where(
        is_ts,
        "THUNDERSTORM",
        np.where(
            is_snow_ice,
            "SNOW_ICE",
            np.where(is_fog, "DENSE_FOG", np.where(is_rain, "RAIN", "CLEAR_VFR")),
        ),
    )


def download_noaa_asos_weather(year: int = BTS_YEAR) -> int:
    out_wx = RAW_DIR / "noaa_weather_hourly.csv"
    tz_map = {a["IATA"]: a["TZ"] for a in AIRPORTS_REGISTRY}
    stations = [a["IATA"] for a in AIRPORTS_REGISTRY]

    print(f"[Tier 2] Downloading full-year {year} NOAA/NWS ASOS hourly surface weather for {len(stations)} U.S. hubs...")
    t0 = time.perf_counter()
    batch_size = 10
    station_frames = []

    for i in range(0, len(stations), batch_size):
        batch = stations[i : i + batch_size]
        params = [
            ("year1", str(year)), ("month1", "1"), ("day1", "1"),
            ("year2", str(year + 1)), ("month2", "1"), ("day2", "2"),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("report_type", "3"),
        ]
        for st_code in batch:
            params.append(("station", st_code))
        for col in ["tmpf", "dwpf", "sknt", "vsby", "p01i", "mslp", "wxcodes"]:
            params.append(("data", col))

        for attempt in range(3):
            try:
                resp = requests.get(ASOS_BASE_URL, params=params, timeout=120)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Failed ASOS batch {batch}: {exc}") from exc
                time.sleep(2.5 * (attempt + 1))

        raw_df = pd.read_csv(
            io.StringIO(resp.text),
            na_values=["null", "M", ""],
            low_memory=False,
        )
        station_frames.append(raw_df)
        print(f"  -> Downloaded ASOS batch {i // batch_size + 1}/{(len(stations) + batch_size - 1) // batch_size}: {len(raw_df):,} raw METAR observations ({', '.join(batch)})")
        time.sleep(1.1)  # Respect IEM 1-sec rate limit policy

    all_wx = pd.concat(station_frames, ignore_index=True)
    all_wx = all_wx.dropna(subset=["station", "valid"]).copy()
    all_wx["valid_utc"] = pd.to_datetime(all_wx["valid"], utc=True, errors="coerce")
    all_wx = all_wx.dropna(subset=["valid_utc"]).copy()

    # Convert each airport's UTC METAR observation to local airport wall-clock hour
    converted_parts = []
    full_local_hours = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    full_hour_strs = full_local_hours.strftime("%Y-%m-%d %H:%M:%S")

    for st_code, grp in all_wx.groupby("station"):
        if st_code not in tz_map:
            continue
        local_tz = ZoneInfo(tz_map[st_code])
        local_dt = grp["valid_utc"].dt.tz_convert(local_tz).dt.tz_localize(None)
        # Round to nearest hour so :51/:53/:54 routine hourly METAR aligns cleanly to HH:00:00
        grp = grp.copy()
        grp["OBS_TIMESTAMP"] = local_dt.dt.round("h").dt.strftime("%Y-%m-%d %H:%M:%S")
        grp["tmpf"] = pd.to_numeric(grp["tmpf"], errors="coerce")
        grp["dwpf"] = pd.to_numeric(grp["dwpf"], errors="coerce")
        grp["sknt"] = pd.to_numeric(grp["sknt"], errors="coerce")
        grp["vsby"] = pd.to_numeric(grp["vsby"], errors="coerce")
        grp["p01i"] = pd.to_numeric(grp["p01i"], errors="coerce").fillna(0.0)
        grp["mslp"] = pd.to_numeric(grp["mslp"], errors="coerce")

        # Keep closest reading per local hour
        dedup = grp.sort_values("valid_utc").drop_duplicates(subset=["OBS_TIMESTAMP"], keep="last")

        # Reindex onto complete 2024 hourly grid so flight join has 100% hourly coverage
        grid = pd.DataFrame({"OBS_TIMESTAMP": full_hour_strs})
        merged_st = grid.merge(
            dedup[["OBS_TIMESTAMP", "tmpf", "dwpf", "sknt", "vsby", "p01i", "mslp", "wxcodes"]],
            on="OBS_TIMESTAMP",
            how="left",
        )
        merged_st["STATION_IATA"] = st_code
        merged_st["TEMP_F"] = merged_st["tmpf"].ffill().bfill().fillna(58.0).round(1)
        merged_st["DEW_POINT_F"] = merged_st["dwpf"].ffill().bfill().fillna(44.0).round(1)
        merged_st["WIND_SPEED_KT"] = merged_st["sknt"].ffill().bfill().fillna(7.0).clip(0.0, 75.0).round(1)
        merged_st["VISIBILITY_MI"] = merged_st["vsby"].ffill().bfill().fillna(10.0).clip(0.1, 10.0).round(2)
        merged_st["PRECIP_IN"] = merged_st["p01i"].fillna(0.0).clip(0.0, 5.0).round(3)
        merged_st["PRESSURE_MB"] = merged_st["mslp"].ffill().bfill().fillna(1015.0).clip(960.0, 1055.0).round(1)
        merged_st["WEATHER_CONDITION"] = _classify_metar_wx(
            merged_st["wxcodes"], merged_st["TEMP_F"], merged_st["VISIBILITY_MI"], merged_st["PRECIP_IN"]
        )

        converted_parts.append(merged_st[[
            "STATION_IATA", "OBS_TIMESTAMP", "TEMP_F", "DEW_POINT_F",
            "WIND_SPEED_KT", "VISIBILITY_MI", "PRECIP_IN", "PRESSURE_MB", "WEATHER_CONDITION"
        ]])

    final_wx = pd.concat(converted_parts, ignore_index=True)
    final_wx.to_csv(out_wx, index=False)
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"[Tier 2] Saved {len(final_wx):,} hourly local-aligned NOAA ASOS observations across {len(converted_parts)} hubs in {elapsed}s.")
    return len(final_wx)


def main() -> None:
    save_metadata_registries()
    n_wx = download_noaa_asos_weather(BTS_YEAR)
    n_bts = download_all_bts_partitions(BTS_YEAR)
    print(f"\n[download_real_bigdata] COMPLETE: {n_bts:,} real BTS flights + {n_wx:,} real NOAA ASOS weather rows ready!")


if __name__ == "__main__":
    main()
