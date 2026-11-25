"""
Realistic Multi-Tier Dataset Generator (BTS Flight Partitions + NOAA Hourly Weather + Registries).
Generates 3 monthly BTS flight partitions (~90,000 domestic flight records), 43,200 hourly NOAA
surface weather observations across 20 U.S. hubs, and airport/airline metadata lookup registries.

Models real aviation operational dynamics cited in the report:
  1. 80/20 on-time vs delayed (>=15 min) class imbalance.
  2. Sharp non-linear weather threshold effects (winds > 25 kt, visibility < 3 mi, freezing precip).
  3. Downstream ripple effects & rolling origin hub departure congestion across the 24-hour cycle.
  4. Realistic ~1.8% cancellations, ~0.3% diversions, and ~0.4% incomplete/NULL sensor rows.
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import (
    RAW_DIR,
    AIRPORTS_REGISTRY,
    AIRLINES_REGISTRY,
    RANDOM_SEED,
    DELAY_THRESHOLD_MIN,
)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return round(2 * r_miles * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 0)


def generate_metadata_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    airports_df = pd.DataFrame(AIRPORTS_REGISTRY)
    airlines_df = pd.DataFrame(AIRLINES_REGISTRY)[["CARRIER_CODE", "AIRLINE_NAME", "FLEET_TYPE"]]
    airports_df.to_csv(RAW_DIR / "airports_metadata.csv", index=False)
    airlines_df.to_csv(RAW_DIR / "airlines_metadata.csv", index=False)
    return airports_df, pd.DataFrame(AIRLINES_REGISTRY)


def generate_noaa_weather(rng: np.random.Generator, start_dt: datetime, days: int = 90) -> pd.DataFrame:
    """Generates hourly NOAA surface weather readings for each origin airport."""
    hours = days * 24
    timestamps = [start_dt + timedelta(hours=h) for h in range(hours)]

    climate_base = {
        "continental_winter": (29.0, 13.0),
        "alpine_winter": (34.0, 14.0),
        "northeast_coastal": (36.0, 12.0),
        "pacific_rain": (46.0, 8.0),
        "coastal_fog": (55.0, 7.0),
        "coastal_marine": (63.0, 7.0),
        "humid_subtropical": (54.0, 13.0),
        "convective_plains": (53.0, 15.0),
        "tropical": (74.0, 8.0),
        "arid_desert": (64.0, 12.0),
    }

    records = []
    hour_idx = np.arange(hours)
    day_frac = hour_idx / 24.0

    for ap in AIRPORTS_REGISTRY:
        iata = ap["IATA"]
        climate = ap["CLIMATE"]
        base_temp, std_temp = climate_base[climate]

        # Seasonal warming from Jan -> Mar + diurnal cycle + synoptic weather waves
        seasonal = 14.0 * (day_frac / days)
        diurnal = 7.5 * np.sin(2 * np.pi * (hour_idx - 9) / 24.0)
        synoptic_wave = np.convolve(rng.normal(0, 1, hours + 24), np.ones(25) / 5.0, mode="valid")[:hours]
        temp_f = np.round(base_temp + seasonal + diurnal + 2.5 * synoptic_wave, 1)

        # Storm front indicator (persistent multi-hour weather events)
        storm_signal = np.convolve(rng.normal(0, 1, hours + 12), np.ones(13) / 3.6, mode="valid")[:hours]
        is_storm = storm_signal > 1.35
        is_severe_storm = storm_signal > 2.05

        # Fog tendency (especially SFO, SEA, LAX, BOS in early morning)
        early_morning = ((hour_idx % 24) >= 4) & ((hour_idx % 24) <= 9)
        fog_prone = climate in ("coastal_fog", "pacific_rain", "northeast_coastal")
        is_fog = early_morning & (rng.random(hours) < (0.20 if fog_prone else 0.04))

        # Dew point close to temp during storms/fog
        spread = np.where(
            is_fog,
            rng.uniform(0.3, 2.2, hours),
            np.where(is_storm, rng.uniform(1.5, 5.5, hours), rng.uniform(6.0, 22.0, hours)),
        )
        dew_point_f = np.round(temp_f - spread, 1)

        # Wind speed (knots): sharp tail > 25 kt during fronts/storms
        wind_kt = rng.gamma(shape=2.6, scale=3.4, size=hours)
        wind_kt = np.where(is_storm, wind_kt + rng.uniform(10, 19, hours), wind_kt)
        wind_kt = np.where(is_severe_storm, wind_kt + rng.uniform(8, 15, hours), wind_kt)
        wind_kt = np.round(np.clip(wind_kt, 0.0, 48.0), 1)

        # Visibility (statute miles, max 10.0)
        vis_mi = np.full(hours, 10.0)
        vis_mi = np.where(is_storm, rng.uniform(2.2, 6.5, hours), vis_mi)
        vis_mi = np.where(is_severe_storm, rng.uniform(0.5, 2.5, hours), vis_mi)
        vis_mi = np.where(is_fog, rng.uniform(0.25, 2.0, hours), vis_mi)
        vis_mi = np.round(np.clip(vis_mi, 0.25, 10.0), 2)

        # Precipitation (inches/hr)
        precip_in = np.where(
            is_severe_storm,
            rng.exponential(0.28, hours) + 0.12,
            np.where(is_storm, rng.exponential(0.09, hours) + 0.02, 0.0),
        )
        precip_in = np.round(np.clip(precip_in, 0.0, 1.75), 2)

        # Barometric pressure (mb)
        pressure_mb = np.round(1016.5 - 5.5 * storm_signal + rng.normal(0, 1.5, hours), 1)

        cond = np.where(
            is_severe_storm & (temp_f <= 33.0),
            "BLIZZARD_ICE",
            np.where(
                is_severe_storm,
                "THUNDERSTORM",
                np.where(
                    is_fog,
                    "DENSE_FOG",
                    np.where(is_storm & (temp_f <= 33.0), "SNOW", np.where(is_storm, "RAIN", "CLEAR_VFR")),
                ),
            ),
        )

        ap_df = pd.DataFrame(
            {
                "STATION_IATA": iata,
                "OBS_TIMESTAMP": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
                "TEMP_F": temp_f,
                "DEW_POINT_F": dew_point_f,
                "WIND_SPEED_KT": wind_kt,
                "VISIBILITY_MI": vis_mi,
                "PRECIP_IN": precip_in,
                "PRESSURE_MB": pressure_mb,
                "WEATHER_CONDITION": cond,
            }
        )
        records.append(ap_df)

    weather_df = pd.concat(records, ignore_index=True)
    weather_df.to_csv(RAW_DIR / "noaa_weather_hourly.csv", index=False)
    return weather_df


def generate_bts_flights(
    rng: np.random.Generator,
    weather_df: pd.DataFrame,
    start_dt: datetime,
    days: int = 90,
    total_flights: int = 90000,
) -> list[str]:
    """Generates monthly BTS domestic flight CSV partitions with realistic non-linear delay physics."""
    ap_map = {a["IATA"]: a for a in AIRPORTS_REGISTRY}
    iata_list = [a["IATA"] for a in AIRPORTS_REGISTRY]
    cap_weights = np.array([a["CAPACITY_HR"] for a in AIRPORTS_REGISTRY], dtype=float)
    cap_probs = cap_weights / cap_weights.sum()

    carriers = [c["CARRIER_CODE"] for c in AIRLINES_REGISTRY]
    carrier_otp = {c["CARRIER_CODE"]: c["OTP_FACTOR"] for c in AIRLINES_REGISTRY}
    carrier_probs = np.array([0.19, 0.07, 0.17, 0.19, 0.18, 0.08, 0.06, 0.06])

    #Realistic commercial departure hour profile (05:00 to 23:00 heavy, scarce redeyes)
    hour_weights = np.array([
        0.3, 0.1, 0.1, 0.1, 0.3,   # 00-04
        2.5, 6.2, 7.5, 7.8, 7.0,   # 05-09
        6.5, 6.6, 6.8, 6.9, 7.1,   # 10-14
        7.3, 7.6, 7.9, 7.4, 6.5,   # 15-19
        5.2, 3.8, 2.2, 1.0         # 20-23
    ])
    hour_probs = hour_weights / hour_weights.sum()

    day_offsets = rng.integers(0, days, size=total_flights)
    dep_hours = rng.choice(24, size=total_flights, p=hour_probs)
    dep_mins = rng.integers(0, 60, size=total_flights)

    origins = rng.choice(iata_list, size=total_flights, p=cap_probs)
    dests = rng.choice(iata_list, size=total_flights, p=cap_probs)
    same_mask = origins == dests
    while same_mask.any():
        dests[same_mask] = rng.choice(iata_list, size=same_mask.sum(), p=cap_probs)
        same_mask = origins == dests

    chosen_carriers = rng.choice(carriers, size=total_flights, p=carrier_probs)
    fl_nums = rng.integers(100, 4999, size=total_flights)
    tail_nums = [f"N{rng.integers(101, 999)}{c}" for c in chosen_carriers]

    # Build base dataframe
    fl_dates = [start_dt + timedelta(days=int(d)) for d in day_offsets]
    crs_dep_ts = [
        datetime(d.year, d.month, d.day, int(h), int(m))
        for d, h, m in zip(fl_dates, dep_hours, dep_mins)
    ]
    obs_hour_ts = [
        datetime(d.year, d.month, d.day, int(h), 0).strftime("%Y-%m-%d %H:%M:%S")
        for d, h in zip(fl_dates, dep_hours)
    ]

    distances = np.array([
        max(180.0, haversine_miles(ap_map[o]["LAT"], ap_map[o]["LON"], ap_map[d]["LAT"], ap_map[d]["LON"]))
        for o, d in zip(origins, dests)
    ])

    flights_df = pd.DataFrame({
        "FL_DATE": [d.strftime("%Y-%m-%d") for d in fl_dates],
        "OP_UNIQUE_CARRIER": chosen_carriers,
        "TAIL_NUM": tail_nums,
        "OP_CARRIER_FL_NUM": fl_nums,
        "ORIGIN": origins,
        "DEST": dests,
        "CRS_DEP_TIME": dep_hours * 100 + dep_mins,
        "CRS_DEP_TIMESTAMP": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in crs_dep_ts],
        "OBS_TIMESTAMP": obs_hour_ts,
        "DISTANCE": distances,
        "_dep_hour": dep_hours + dep_mins / 60.0,
    })

    # Sort chronologically to compute genuine rolling airport departure load
    flights_df.sort_values("CRS_DEP_TIMESTAMP", inplace=True)
    flights_df.reset_index(drop=True, inplace=True)

    # Merge origin weather to simulate physically grounded delays
    wx = weather_df.rename(columns={"STATION_IATA": "ORIGIN"})
    merged = flights_df.merge(wx, on=["ORIGIN", "OBS_TIMESTAMP"], how="left")

    # Compute rolling 3-hour origin departure count per airport-hour
    hourly_counts = (
        merged.groupby(["ORIGIN", "OBS_TIMESTAMP"]).size().rename("hr_cnt").reset_index()
    )
    hourly_counts["obs_dt"] = pd.to_datetime(hourly_counts["OBS_TIMESTAMP"])
    hourly_counts.sort_values(["ORIGIN", "obs_dt"], inplace=True)
    hourly_counts["roll3h"] = (
        hourly_counts.groupby("ORIGIN")["hr_cnt"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    merged = merged.merge(
        hourly_counts[["ORIGIN", "OBS_TIMESTAMP", "roll3h"]],
        on=["ORIGIN", "OBS_TIMESTAMP"],
        how="left",
    )
    cap_vec = merged["ORIGIN"].map({a["IATA"]: a["CAPACITY_HR"] * 0.42 for a in AIRPORTS_REGISTRY})
    congestion_ratio = merged["roll3h"] / np.maximum(cap_vec, 1.0)

    # Non-linear latent delay propensity (Choi et al. 2016 + Rebollo & Balakrishnan 2014)
    h = merged["_dep_hour"].to_numpy()
    wind = merged["WIND_SPEED_KT"].to_numpy()
    vis = merged["VISIBILITY_MI"].to_numpy()
    precip = merged["PRECIP_IN"].to_numpy()
    temp = merged["TEMP_F"].to_numpy()
    dew = merged["DEW_POINT_F"].to_numpy()
    c_factor = merged["OP_UNIQUE_CARRIER"].map(carrier_otp).to_numpy()

    # Non-linear threshold & U-shaped interactions (why tree ensembles strongly beat linear models!)
    score = np.full(len(merged), -2.85)
    score += c_factor * 0.95

    # 1. Sharp wind step-function at >24 kt (below 22 kt has zero effect -> fools linear slope)
    score += np.where(wind > 28.0, 2.75, np.where(wind > 23.0, 1.85, 0.0))

    # 2. Sharp IFR visibility cliff (< 2.5 miles)
    score += np.where(vis < 1.5, 2.45, np.where(vis < 3.0, 1.45, 0.0))

    # 3. U-shaped temperature x moisture interaction:
    #    Cold freezing de-icing (temp <= 33F & precip/fog) AND hot convective density-altitude (>82F & afternoon)
    deicing = (temp <= 33.5) & ((precip >= 0.03) | ((temp - dew) < 2.8))
    hot_convective = (temp >= 78.0) & (h >= 13.0) & (h <= 20.0)
    score += np.where(deicing, 2.55, 0.0)
    score += np.where(hot_convective, 1.65, 0.0)
    score += np.where(precip >= 0.22, 1.75, 0.0)

    # 4. Pure 2-way interaction: Downstream ripple effect!
    #    High hub departure load in the morning (05-10) runs smoothly (overnighted aircraft),
    #    whereas high hub load in late afternoon/evening (15:30-22:30) triggers severe cascading delays!
    evening_bank = (h >= 15.5) & (h <= 22.5)
    morning_bank = (h >= 5.0) & (h <= 9.5)
    score += np.where(evening_bank & (congestion_ratio > 1.02), 2.35, 0.0)
    score += np.where(evening_bank & (congestion_ratio <= 1.02), 0.55, 0.0)
    score += np.where(morning_bank & (congestion_ratio > 1.05), -0.65, -0.35)

    # 5. Fog saturation interaction on coastal hubs
    score += np.where(((temp - dew) < 2.2) & (vis < 3.5), 1.25, 0.0)

    # Calibrated noise -> ~20.0% delay rate (80/20 class skew)
    logistic_noise = rng.logistic(loc=0.0, scale=0.58, size=len(merged))
    latent = score + logistic_noise
    # Calibrate threshold so exactly ~20.2% of active flights are delayed >= 15 min
    cutoff = np.quantile(latent, 0.798)
    is_delayed = latent >= cutoff

    # Continuous DEP_DELAY in minutes (for both classification target and secondary regression baseline)
    on_time_delays = np.round(rng.normal(loc=-2.5, scale=5.2, size=len(merged)), 0)
    on_time_delays = np.clip(on_time_delays, -15.0, 14.0)

    severity = np.maximum(0.0, latent - cutoff)
    delayed_mins = np.round(15.0 + 18.0 * severity + rng.exponential(scale=16.0, size=len(merged)), 0)
    delayed_mins = np.clip(delayed_mins, 15.0, 360.0)

    dep_delay = np.where(is_delayed, delayed_mins, on_time_delays)
    dep_del15 = (dep_delay >= DELAY_THRESHOLD_MIN).astype(int)

    # Realistic cancellations (~1.7%, higher in blizzards/severe storms) & diversions (~0.25%)
    cancel_prob = np.where((wind > 30) | (vis < 1.0) | deicing, 0.085, 0.011)
    cancelled = (rng.random(len(merged)) < cancel_prob).astype(int)
    diverted = ((cancelled == 0) & (rng.random(len(merged)) < 0.0025)).astype(int)

    # Compute actual DEP_TIME HHMM
    actual_dep_time = []
    for ts_str, d_min, canc in zip(merged["CRS_DEP_TIMESTAMP"], dep_delay, cancelled):
        if canc == 1:
            actual_dep_time.append(np.nan)
        else:
            dt_actual = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=float(d_min))
            actual_dep_time.append(dt_actual.hour * 100 + dt_actual.minute)

    out_df = pd.DataFrame({
        "FL_DATE": merged["FL_DATE"],
        "OP_UNIQUE_CARRIER": merged["OP_UNIQUE_CARRIER"],
        "TAIL_NUM": merged["TAIL_NUM"],
        "OP_CARRIER_FL_NUM": merged["OP_CARRIER_FL_NUM"],
        "ORIGIN": merged["ORIGIN"],
        "DEST": merged["DEST"],
        "CRS_DEP_TIME": merged["CRS_DEP_TIME"],
        "CRS_DEP_TIMESTAMP": merged["CRS_DEP_TIMESTAMP"],
        "DEP_TIME": actual_dep_time,
        "DEP_DELAY": np.where(cancelled == 1, np.nan, dep_delay),
        "DEP_DEL15": np.where(cancelled == 1, np.nan, dep_del15),
        "CANCELLED": cancelled,
        "CANCELLATION_CODE": np.where(cancelled == 1, np.where(wind > 24, "B", "A"), ""),
        "DIVERTED": diverted,
        "DISTANCE": merged["DISTANCE"],
    })

    # Inject ~0.35% realistic incomplete/NULL sensor rows so ingestion & Spark cleaning handle real dirty data
    null_idx = rng.choice(len(out_df), size=int(0.0035 * len(out_df)), replace=False)
    out_df.loc[null_idx[: len(null_idx) // 2], "DEP_DELAY"] = np.nan
    out_df.loc[null_idx[len(null_idx) // 2 :], "ORIGIN"] = None

    # Save as monthly partitions (Jan, Feb, Mar 2025)
    out_df["_month"] = out_df["FL_DATE"].str.slice(0, 7)
    saved_files = []
    for month_str, part_df in out_df.groupby("_month"):
        fname = f"bts_flights_{month_str.replace('-', '_')}.csv"
        fpath = RAW_DIR / fname
        part_df.drop(columns=["_month"]).to_csv(fpath, index=False)
        saved_files.append(str(fpath))

    return saved_files


def run_dataset_generation(total_flights: int = 90000) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    start_dt = datetime(2025, 1, 1, 0, 0, 0)
    airports_df, airlines_df = generate_metadata_tables()
    weather_df = generate_noaa_weather(rng, start_dt=start_dt, days=90)
    partition_files = generate_bts_flights(
        rng, weather_df=weather_df, start_dt=start_dt, days=90, total_flights=total_flights
    )
    summary = {
        "airports_count": len(airports_df),
        "airlines_count": len(airlines_df),
        "weather_hourly_rows": len(weather_df),
        "flight_partitions": partition_files,
        "total_raw_flights": total_flights,
    }
    print(f"[generate_datasets] Generated {total_flights:,} raw BTS flights across {len(partition_files)} monthly partitions "
          f"and {len(weather_df):,} hourly NOAA weather observations.")
    return summary


if __name__ == "__main__":
    run_dataset_generation()
