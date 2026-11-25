"""
Stage 7 — Interactive Streamlit Live Demo & Business Intelligence Console (app.py)
Project: Predicting Flight Delays: A Big Data and Machine Learning Approach
Course: Big Data Analytics and Business Intelligence (CBCA 293) — Bennett University
Module Owner: Abhijeet Rana (S24CSEU1419) | Full Team: Parth, Vipul Gupta, Krishna Sharma, Abhijeet Rana
"""

from __future__ import annotations
import json
import math
from datetime import date, time as dt_time
import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import (
    AIRPORTS_REGISTRY,
    AIRLINES_REGISTRY,
    DUCKDB_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLANS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SQL_DIR,
)
from feature_engineering import encode_cyclical_time
from generate_datasets import haversine_miles

st.set_page_config(
    page_title="Flight Delay Intelligence | Big Data & BI Live Demo",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise Aviation & Presentation Styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2.2rem; }
    .hero-card {
        background: linear-gradient(135deg, #090d16 0%, #1e3a8a 55%, #0284c7 100%);
        color: #f8fafc;
        padding: 1.35rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.0rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.22);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.2);
        border: 1px solid rgba(56, 189, 248, 0.5);
        color: #bae6fd;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero-title { font-size: 1.8rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
    .hero-sub { font-size: 0.94rem; opacity: 0.92; margin-top: 0.35rem; }
    .verdict-ontime {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        color: white; padding: 1.2rem 1.45rem; border-radius: 14px;
        border-left: 7px solid #34d399;
        box-shadow: 0 6px 18px rgba(5, 150, 105, 0.22);
        margin-bottom: 0.75rem;
    }
    .verdict-delayed {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        color: white; padding: 1.2rem 1.45rem; border-radius: 14px;
        border-left: 7px solid #fca5a5;
        box-shadow: 0 6px 18px rgba(220, 38, 38, 0.25);
        margin-bottom: 0.75rem;
    }
    .case-banner {
        background: linear-gradient(90deg, rgba(30, 58, 138, 0.08), rgba(14, 165, 233, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.28);
        border-radius: 12px;
        padding: 0.7rem 1.0rem;
        margin-bottom: 0.85rem;
    }
    .viva-tip {
        background: rgba(245, 158, 11, 0.10);
        border-left: 4px solid #f59e0b;
        padding: 0.65rem 0.95rem;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AP_OPTIONS = {f"{a['IATA']} — {a['CITY']} ({a['NAME']})": a["IATA"] for a in AIRPORTS_REGISTRY}
AP_LABELS = list(AP_OPTIONS.keys())
IATA_TO_LABEL = {v: k for k, v in AP_OPTIONS.items()}

AL_OPTIONS = {f"{c['CARRIER_CODE']} — {c['AIRLINE_NAME']}": c["CARRIER_CODE"] for c in AIRLINES_REGISTRY}
AL_LABELS = list(AL_OPTIONS.keys())
CARRIER_TO_LABEL = {v: k for k, v in AL_OPTIONS.items()}

TRAFFIC_OPTIONS = [
    "Light Off-Peak (0.75x)",
    "Standard Schedule (1.0x)",
    "Busy Bank (1.25x)",
    "Peak Holiday Bank (1.50x)",
]
TRAFFIC_MULT_MAP = {
    "Light Off-Peak (0.75x)": 0.75,
    "Standard Schedule (1.0x)": 1.0,
    "Busy Bank (1.25x)": 1.25,
    "Peak Holiday Bank (1.50x)": 1.50,
}

WEATHER_PRESETS = {
    "Clear VFR Day (Nominal)": {"temp": 64.0, "dew": 45.0, "wind": 8.5, "vis": 10.0, "precip": 0.00, "press": 1018.0},
    "High Crosswinds (>25 kt Runway Alert)": {"temp": 52.0, "dew": 38.0, "wind": 31.5, "vis": 8.5, "precip": 0.02, "press": 1006.0},
    "Summer Convective Thunderstorm": {"temp": 82.0, "dew": 75.5, "wind": 29.5, "vis": 1.6, "precip": 0.46, "press": 1002.0},
    "Winter Blizzard / Freezing De-icing": {"temp": 24.0, "dew": 22.0, "wind": 27.5, "vis": 0.8, "precip": 0.32, "press": 1004.0},
    "Dense Coastal Morning Fog (IFR)": {"temp": 52.0, "dew": 51.0, "wind": 5.5, "vis": 0.5, "precip": 0.01, "press": 1016.0},
}

DEMO_CASE_STUDIES = {
    "☀️ Case 1: Clear Morning VFR": {
        "origin": "ATL", "dest": "MCO", "carrier": "DL",
        "dep_hour": 7, "dep_min": 15, "traffic": "Standard Schedule (1.0x)",
        "prev_delay": 0.0, "hub_t2h_rate": 0.08,
        "wx": {"temp": 65.0, "dew": 47.0, "wind": 7.0, "vis": 10.0, "precip": 0.00, "press": 1019.0},
        "story": "Nominal morning hub departure (ATL → MCO, Delta 07:15): overnighted aircraft + VFR weather -> very low delay risk.",
    },
    "🌬️ Case 2: >25 kt Crosswind Cliff": {
        "origin": "ORD", "dest": "LGA", "carrier": "UA",
        "dep_hour": 17, "dep_min": 30, "traffic": "Busy Bank (1.25x)",
        "prev_delay": 18.0, "hub_t2h_rate": 0.36,
        "wx": {"temp": 50.0, "dew": 37.0, "wind": 31.5, "vis": 8.5, "precip": 0.02, "press": 1005.0},
        "story": "Choi et al. (2016) Wind Threshold Demo (ORD → LGA, 31.5 kt gusts): crosses FAA 25 kt runway limit with rising T-2h hub delays.",
    },
    "❄️ Case 3: Blizzard & De-icing": {
        "origin": "DEN", "dest": "MSP", "carrier": "F9",
        "dep_hour": 19, "dep_min": 15, "traffic": "Busy Bank (1.25x)",
        "prev_delay": 42.0, "hub_t2h_rate": 0.54,
        "wx": {"temp": 24.0, "dew": 22.5, "wind": 27.5, "vis": 0.75, "precip": 0.34, "press": 1003.0},
        "story": "Severe Winter De-icing + IFR (DEN → MSP, 24°F + heavy snow + 42m inbound tail lag): triggers glycol pad queue + turnaround compound delay.",
    },
    "🌫️ Case 4: SFO Marine Fog (IFR)": {
        "origin": "SFO", "dest": "SEA", "carrier": "AS",
        "dep_hour": 8, "dep_min": 30, "traffic": "Busy Bank (1.25x)",
        "prev_delay": 14.0, "hub_t2h_rate": 0.38,
        "wx": {"temp": 52.0, "dew": 51.0, "wind": 6.0, "vis": 0.50, "precip": 0.01, "press": 1016.0},
        "story": "Coastal Fog Saturation (SFO → SEA, T-Td spread = 1.0°F, 0.5 mi visibility): forces single-runway IFR separation at SFO.",
    },
    "🌇 Case 5: Evening Cascade (Clear Sky)": {
        "origin": "EWR", "dest": "MIA", "carrier": "B6",
        "dep_hour": 20, "dep_min": 15, "traffic": "Peak Holiday Bank (1.50x)",
        "prev_delay": 48.0, "hub_t2h_rate": 0.45,
        "wx": {"temp": 67.0, "dew": 49.0, "wind": 9.0, "vis": 10.0, "precip": 0.00, "press": 1017.0},
        "story": "Rebollo & Balakrishnan (2014) Ripple Demo (EWR → MIA, 20:15, +48m inbound aircraft delay): proves late inbound tails + hub congestion trigger delays even in clear sky!",
    },
}

PRESET_SQL_QUERIES = {
    "1. Top 10 Origin Hubs by Delay Rate & Congestion": """SELECT
    f.ORIGIN AS hub_iata,
    ap.CITY AS city,
    COUNT(*) AS total_flights,
    ROUND(AVG(CASE WHEN f.DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 2) AS delay_rate_pct,
    ROUND(AVG(f.DEP_DELAY), 1) AS avg_delay_min
FROM bts_flights f
JOIN airports_metadata ap ON f.ORIGIN = ap.IATA
WHERE f.CANCELLED = 0 AND f.DIVERTED = 0 AND f.DEP_DELAY IS NOT NULL
GROUP BY f.ORIGIN, ap.CITY
ORDER BY delay_rate_pct DESC
LIMIT 10;""",
    "2. NOAA Surface Weather Condition Impact (SQL JOIN)": """SELECT
    w.WEATHER_CONDITION AS weather_regime,
    COUNT(*) AS flights_observed,
    ROUND(AVG(w.WIND_SPEED_KT), 1) AS avg_wind_kt,
    ROUND(AVG(w.VISIBILITY_MI), 2) AS avg_vis_mi,
    ROUND(AVG(CASE WHEN f.DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 2) AS delay_rate_pct,
    ROUND(AVG(f.DEP_DELAY), 1) AS avg_delay_min
FROM bts_flights f
INNER JOIN noaa_weather_hourly w
  ON w.STATION_IATA = f.ORIGIN
 AND w.OBS_TIMESTAMP = SUBSTR(f.CRS_DEP_TIMESTAMP, 1, 13) || ':00:00'
WHERE f.CANCELLED = 0 AND f.DIVERTED = 0 AND f.DEP_DELAY IS NOT NULL
GROUP BY w.WEATHER_CONDITION
ORDER BY delay_rate_pct DESC;""",
    "3. Airline Carrier On-Time Performance Leaderboard": """SELECT
    f.OP_UNIQUE_CARRIER AS carrier,
    al.AIRLINE_NAME AS airline_name,
    al.FLEET_TYPE AS operation_model,
    COUNT(*) AS active_flights,
    ROUND(AVG(CASE WHEN f.DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 2) AS delay_rate_pct,
    ROUND(AVG(f.DEP_DELAY), 1) AS avg_delay_min
FROM bts_flights f
JOIN airlines_metadata al ON f.OP_UNIQUE_CARRIER = al.CARRIER_CODE
WHERE f.CANCELLED = 0 AND f.DIVERTED = 0 AND f.DEP_DELAY IS NOT NULL
GROUP BY f.OP_UNIQUE_CARRIER, al.AIRLINE_NAME, al.FLEET_TYPE
ORDER BY delay_rate_pct DESC;""",
    "4. Time-of-Day Downstream Ripple Wave Analysis": """SELECT
    CAST(f.CRS_DEP_TIME / 100 AS INTEGER) AS dep_hour_24h,
    COUNT(*) AS departures,
    ROUND(AVG(CASE WHEN f.DEP_DEL15 = 1 THEN 100.0 ELSE 0.0 END), 2) AS delay_rate_pct,
    ROUND(AVG(f.DEP_DELAY), 1) AS avg_delay_min
FROM bts_flights f
WHERE f.CANCELLED = 0 AND f.DIVERTED = 0 AND f.DEP_DELAY IS NOT NULL
GROUP BY dep_hour_24h
ORDER BY dep_hour_24h ASC;""",
}


@st.cache_resource
def load_model_bundle() -> dict:
    model_path = MODELS_DIR / "flight_delay_models.joblib"
    if not model_path.exists():
        from run_pipeline import main as run_full_pipeline
        run_full_pipeline()
    bundle = joblib.load(model_path)
    # Single-thread online inference avoids joblib/OpenMP thread-spawn latency on small interactive queries
    for clf in bundle["classifiers"].values():
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = 1
    if hasattr(bundle["delay_regressor"], "n_jobs"):
        bundle["delay_regressor"].n_jobs = 1
    return bundle


@st.cache_data
def load_pipeline_artifacts() -> dict:
    with open(OUTPUTS_DIR / "ingestion_audit.json", "r", encoding="utf-8") as f:
        ingestion = json.load(f)
    with open(OUTPUTS_DIR / "clean_join_summary.json", "r", encoding="utf-8") as f:
        clean_join = json.load(f)
    with open(OUTPUTS_DIR / "smote_summary.json", "r", encoding="utf-8") as f:
        smote = json.load(f)
    with open(OUTPUTS_DIR / "evaluation_report.json", "r", encoding="utf-8") as f:
        eval_rep = json.load(f)

    plan_text = (PLANS_DIR / "pyspark_execution_plan.txt").read_text(encoding="utf-8")
    sql_ddl = (SQL_DIR / "schema_postgres.sql").read_text(encoding="utf-8")
    test_npz = np.load(OUTPUTS_DIR / "xgb_test_predictions.npz")

    sample_pq = PROCESSED_DIR / "dashboard_sample.parquet"
    target_pq = sample_pq if sample_pq.exists() else (PROCESSED_DIR / "featured_flights.parquet")
    df_feat = pd.read_parquet(
        target_pq,
        columns=[
            "ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "dep_hour_float",
            "WIND_SPEED_KT", "VISIBILITY_MI", "PRECIP_IN", "TEMP_F",
            "WEATHER_CONDITION", "origin_congestion_ratio", "DEP_DELAY", "DEP_DEL15",
        ],
    )
    return {
        "ingestion": ingestion,
        "clean_join": clean_join,
        "smote": smote,
        "eval": eval_rep,
        "plan_text": plan_text,
        "sql_ddl": sql_ddl,
        "y_true_sample": test_npz["y_true"],
        "y_prob_sample": test_npz["y_prob"],
        "featured_df": df_feat,
    }


@st.cache_data(show_spinner=False)
def execute_cached_sql(sql_query: str) -> pd.DataFrame:
    if DUCKDB_PATH.exists():
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            return con.execute(sql_query).fetchdf()
        finally:
            con.close()
    # Cloud fallback using portable dashboard_sample.parquet + metadata CSVs
    con = duckdb.connect()
    try:
        spq = (PROCESSED_DIR / "dashboard_sample.parquet").as_posix()
        con.execute(
            f"""
            CREATE VIEW bts_flights AS
            SELECT *, 0 AS CANCELLED, 0 AS DIVERTED, CAST(dep_hour_float * 100 AS INT) AS CRS_DEP_TIME
            FROM read_parquet('{spq}')
            """
        )
        con.execute(
            f"""
            CREATE VIEW noaa_weather_hourly AS
            SELECT ORIGIN AS STATION_IATA, SUBSTR(CRS_DEP_TIMESTAMP,1,13)||':00:00' AS OBS_TIMESTAMP,
                   WIND_SPEED_KT, VISIBILITY_MI, WEATHER_CONDITION
            FROM read_parquet('{spq}')
            """
        )
        con.execute(f"CREATE VIEW airports_metadata AS SELECT * FROM read_csv_auto('{(RAW_DIR / 'airports_metadata.csv').as_posix()}', header=true)")
        con.execute(f"CREATE VIEW airlines_metadata AS SELECT * FROM read_csv_auto('{(RAW_DIR / 'airlines_metadata.csv').as_posix()}', header=true)")
        return con.execute(sql_query).fetchdf()
    finally:
        con.close()


def fetch_live_airport_weather(lat: float, lon: float) -> dict | None:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,dew_point_2m,precipitation,pressure_msl,wind_speed_10m,visibility"
        "&temperature_unit=fahrenheit&wind_speed_unit=kn&precipitation_unit=inch"
    )
    try:
        resp = requests.get(url, timeout=4.5)
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        vis_meters = float(cur.get("visibility", 16093.0))
        vis_miles = round(min(10.0, max(0.25, vis_meters / 1609.34)), 2)
        return {
            "temp_f": round(float(cur.get("temperature_2m", 62.0)), 1),
            "dew_f": round(float(cur.get("dew_point_2m", 46.0)), 1),
            "wind_kt": round(float(cur.get("wind_speed_10m", 9.0)), 1),
            "vis_mi": vis_miles,
            "precip_in": round(float(cur.get("precipitation", 0.0)), 2),
            "pressure_mb": round(float(cur.get("pressure_msl", 1015.0)), 1),
        }
    except Exception:
        return None


def build_single_feature_vector(
    bundle: dict,
    origin_iata: str,
    dest_iata: str,
    carrier_code: str,
    fl_date: date,
    dep_time_val: dt_time,
    temp_f: float,
    dew_f: float,
    wind_kt: float,
    vis_mi: float,
    precip_in: float,
    pressure_mb: float,
    traffic_mult: float = 1.0,
    prev_aircraft_delay: float | None = None,
    origin_t2h_rate: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    ap_map = {a["IATA"]: a for a in AIRPORTS_REGISTRY}
    orig_ap = ap_map[origin_iata]
    dest_ap = ap_map[dest_iata]
    meta = bundle["feature_metadata"]

    dist = max(180.0, haversine_miles(orig_ap["LAT"], orig_ap["LON"], dest_ap["LAT"], dest_ap["LON"]))
    crs_elapsed = round(dist / 7.5 + 35.0, 1)
    dep_hr_float = dep_time_val.hour + dep_time_val.minute / 60.0
    dow = fl_date.weekday()
    cyc = encode_cyclical_time(dep_hr_float, dow, fl_date.month)

    base_vol = meta["hub_hourly_vol_lookup"].get(f"{origin_iata}_{dep_time_val.hour}", orig_ap["CAPACITY_HR"] * 1.2)
    roll_3h_vol = round(float(base_vol * traffic_mult), 1)
    expected_cap = max(orig_ap["CAPACITY_HR"] * 1.6, 10.0)
    cong_ratio = round(roll_3h_vol / expected_cap, 3)

    prior = meta["global_delay_prior"]
    orig_prior = float(meta["origin_hist_delay_rate"].get(origin_iata, prior))
    dest_prior = float(meta["dest_hist_delay_rate"].get(dest_iata, prior))

    has_inbound = 1.0 if dep_time_val.hour >= 9 else 0.0
    if prev_aircraft_delay is None:
        prev_aircraft_delay = 0.0 if dep_time_val.hour < 9 else (14.0 if (wind_kt > 24 or vis_mi < 2.5 or traffic_mult > 1.2) else 0.0)
    if origin_t2h_rate is None:
        wx_stress = (0.14 if wind_kt > 24 else 0.0) + (0.14 if vis_mi < 2.5 else 0.0) + (0.10 if precip_in >= 0.1 else 0.0)
        origin_t2h_rate = min(0.75, orig_prior * traffic_mult + wx_stress)

    row = {
        "DISTANCE": round(dist),
        "CRS_ELAPSED_TIME": round(crs_elapsed),
        "dep_hour_sin": float(cyc["dep_hour_sin"]),
        "dep_hour_cos": float(cyc["dep_hour_cos"]),
        "dow_sin": float(cyc["dow_sin"]),
        "dow_cos": float(cyc["dow_cos"]),
        "month_sin": float(cyc["month_sin"]),
        "month_cos": float(cyc["month_cos"]),
        "is_weekend": int(dow in (5, 6)),
        "TEMP_F": round(float(temp_f)),
        "DEW_POINT_F": round(float(dew_f)),
        "temp_dew_spread": round(float(temp_f)) - round(float(dew_f)),
        "WIND_SPEED_KT": round(float(wind_kt)),
        "VISIBILITY_MI": round(float(vis_mi) * 2.0) / 2.0,
        "PRECIP_IN": round(float(precip_in), 2),
        "PRESSURE_MB": round(float(pressure_mb)),
        "wind_max_3h": round(float(wind_kt) * (1.12 if wind_kt > 18.0 else 1.0)),
        "precip_sum_3h": round(float(precip_in) * 2.2, 2),
        "vis_min_3h": round(float(vis_mi) * 2.0) / 2.0,
        "origin_rolling_3h_vol": round(roll_3h_vol),
        "origin_congestion_ratio": cong_ratio,
        "prev_aircraft_delay_t2h": round(float(prev_aircraft_delay)),
        "has_inbound_leg": has_inbound,
        "origin_t2h_delay_rate": float(origin_t2h_rate),
        "dest_t2h_delay_rate": dest_prior,
        "carrier_hist_delay_rate": float(meta["carrier_hist_delay_rate"].get(carrier_code, prior)),
        "origin_hist_delay_rate": orig_prior,
        "dest_hist_delay_rate": dest_prior,
    }
    X_one = pd.DataFrame([row])[bundle["feature_columns"]].astype(np.float32)
    return X_one, {
        "distance_mi": dist,
        "dep_hour_float": dep_hr_float,
        "roll_3h_vol": roll_3h_vol,
        "congestion_ratio": cong_ratio,
        "orig_ap": orig_ap,
        "dest_ap": dest_ap,
    }


def compute_xai_waterfall(
    bundle: dict,
    origin_iata: str,
    dest_iata: str,
    carrier_code: str,
    fl_date: date,
    dep_time_val: dt_time,
    temp_f: float,
    dew_f: float,
    wind_kt: float,
    vis_mi: float,
    precip_in: float,
    pressure_mb: float,
    traffic_mult: float,
    prev_delay: float,
    hub_t2h_rate: float,
) -> pd.DataFrame:
    """Computes telescoping marginal probability attributions from a nominal reference flight to active query."""
    clf = bundle["classifiers"]["XGBoost Classifier (Tuned)"]

    X0, _ = build_single_feature_vector(bundle, "ATL", "CLT", "DL", date(2024, 9, 18), dt_time(8, 0), 64.0, 45.0, 8.0, 10.0, 0.0, 1018.0, 0.85, 0.0, 0.12)
    X1, _ = build_single_feature_vector(bundle, origin_iata, dest_iata, carrier_code, fl_date, dt_time(8, 0), 64.0, 45.0, 8.0, 10.0, 0.0, 1018.0, 0.85, 0.0, 0.12)
    X2, _ = build_single_feature_vector(bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val, 64.0, 45.0, 8.0, 10.0, 0.0, 1018.0, traffic_mult, 0.0, 0.12)
    X3, _ = build_single_feature_vector(bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val, 64.0, 45.0, 8.0, 10.0, 0.0, 1018.0, traffic_mult, prev_delay, hub_t2h_rate)
    X4, _ = build_single_feature_vector(bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val, temp_f, dew_f, wind_kt, vis_mi, precip_in, pressure_mb, traffic_mult, prev_delay, hub_t2h_rate)

    probs = clf.predict_proba(pd.concat([X0, X1, X2, X3, X4], ignore_index=True))[:, 1] * 100.0
    p0, p1, p2, p3, p4 = [float(x) for x in probs]

    return pd.DataFrame([
        {"Factor": "Nominal VFR Baseline", "Delta": round(p0, 1), "Measure": "absolute"},
        {"Factor": f"Route & Carrier ({carrier_code})", "Delta": round(p1 - p0, 1), "Measure": "relative"},
        {"Factor": f"Slot & Hub Bank ({dep_time_val.strftime('%H:%M')})", "Delta": round(p2 - p1, 1), "Measure": "relative"},
        {"Factor": f"T-2h Tail & Hub Cascade ({prev_delay:+.0f}m)", "Delta": round(p3 - p2, 1), "Measure": "relative"},
        {"Factor": "Origin Surface Weather (NOAA)", "Delta": round(p4 - p3, 1), "Measure": "relative"},
        {"Factor": "Final Predicted Risk", "Delta": round(p4, 1), "Measure": "total"},
    ])


def _init_predictor_state() -> None:
    if "pred_initialized" not in st.session_state:
        default_case = DEMO_CASE_STUDIES["🌬️ Case 2: >25 kt Crosswind Cliff"]
        st.session_state["orig_label_k"] = IATA_TO_LABEL[default_case["origin"]]
        st.session_state["dest_label_k"] = IATA_TO_LABEL[default_case["dest"]]
        st.session_state["carrier_label_k"] = CARRIER_TO_LABEL[default_case["carrier"]]
        st.session_state["dep_hour_k"] = default_case["dep_hour"]
        st.session_state["dep_min_k"] = default_case["dep_min"]
        st.session_state["traffic_label_k"] = default_case["traffic"]
        st.session_state["prev_delay_k"] = float(default_case["prev_delay"])
        st.session_state["hub_t2h_k"] = int(round(default_case["hub_t2h_rate"] * 100))
        st.session_state["wx_preset_k"] = "High Crosswinds (>25 kt Runway Alert)"
        st.session_state["temp_k"] = default_case["wx"]["temp"]
        st.session_state["dew_k"] = default_case["wx"]["dew"]
        st.session_state["wind_k"] = default_case["wx"]["wind"]
        st.session_state["vis_k"] = default_case["wx"]["vis"]
        st.session_state["precip_k"] = default_case["wx"]["precip"]
        st.session_state["press_k"] = default_case["wx"]["press"]
        st.session_state["active_story"] = default_case["story"]
        st.session_state["wx_source"] = "Case Study: >25 kt Crosswind Cliff"
        st.session_state["pred_initialized"] = True


def _apply_case_study(case_key: str) -> None:
    c = DEMO_CASE_STUDIES[case_key]
    st.session_state["orig_label_k"] = IATA_TO_LABEL[c["origin"]]
    st.session_state["dest_label_k"] = IATA_TO_LABEL[c["dest"]]
    st.session_state["carrier_label_k"] = CARRIER_TO_LABEL[c["carrier"]]
    st.session_state["dep_hour_k"] = c["dep_hour"]
    st.session_state["dep_min_k"] = c["dep_min"]
    st.session_state["traffic_label_k"] = c["traffic"]
    st.session_state["prev_delay_k"] = float(c["prev_delay"])
    st.session_state["hub_t2h_k"] = int(round(c["hub_t2h_rate"] * 100))
    st.session_state["temp_k"] = float(c["wx"]["temp"])
    st.session_state["dew_k"] = float(c["wx"]["dew"])
    st.session_state["wind_k"] = float(c["wx"]["wind"])
    st.session_state["vis_k"] = float(c["wx"]["vis"])
    st.session_state["precip_k"] = float(c["wx"]["precip"])
    st.session_state["press_k"] = float(c["wx"]["press"])
    st.session_state["active_story"] = c["story"]
    st.session_state["wx_source"] = f"Loaded {case_key}"


def _apply_wx_preset() -> None:
    pname = st.session_state["wx_preset_k"]
    wx = WEATHER_PRESETS[pname]
    st.session_state["temp_k"] = float(wx["temp"])
    st.session_state["dew_k"] = float(wx["dew"])
    st.session_state["wind_k"] = float(wx["wind"])
    st.session_state["vis_k"] = float(wx["vis"])
    st.session_state["precip_k"] = float(wx["precip"])
    st.session_state["press_k"] = float(wx["press"])
    st.session_state["wx_source"] = f"Preset: {pname}"


def main() -> None:
    _init_predictor_state()
    bundle = load_model_bundle()
    art = load_pipeline_artifacts()

    xgb_bench = next(
        b for b in art["eval"]["benchmarks"] if "XGBoost" in b["Model Architecture"]
    )

    # Hero Banner
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">CBCA 293 • Big Data Analytics & Business Intelligence • Live Demonstration</div>
            <div class="hero-title">✈️ Flight Delay Intelligence & AOCC Decision Console</div>
            <div class="hero-sub">
                Distributed PySpark Cleaning • Partitioned Warehouse • Cyclical Time Geometry • Training-Fold SMOTE • Explainable Tuned XGBoost
                <br/><b>Bennett University</b> — Parth (S24CSEU2470), Vipul Gupta (S24CSEU2526), Krishna Sharma (S24CSEU2756), Abhijeet Rana (S24CSEU1419)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Pipeline KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Raw BTS Flight Logs", f"{art['ingestion']['total_flights_ingested']:,}", f"{art['ingestion']['partitions_loaded']} Monthly CSVs")
    k2.metric("PySpark Clean & Joined", f"{art['clean_join']['joined_clean_rows']:,}", f"{art['clean_join']['delay_rate_pct']}% Delayed (≥15m)")
    k3.metric("Tuned XGBoost ROC-AUC", f"{xgb_bench['ROC-AUC']:.3f}", f"{art['eval'].get('test_rows_evaluated', 1130888):,} Test Flights")
    k4.metric("Delay Catch Recall", f"{xgb_bench['Recall'] * 100:.1f}%", f"Precision: {xgb_bench['Precision'] * 100:.1f}%")
    k5.metric("Delay Regressor MAE", f"{art['eval']['secondary_regression_baseline']['mae_minutes']:.1f} min", "Secondary Baseline")

    # Sidebar Dispatch Controls & Presentation Mode
    with st.sidebar:
        st.header("🎛️ AOCC Dispatch Controls")
        viva_mode = st.toggle("🎓 Guided Viva Walkthrough Notes", value=True, help="Displays concise talking points explaining the Big Data & ML concepts on each screen during a live presentation.")
        decision_threshold = st.slider(
            "Classification Decision Threshold",
            min_value=0.20,
            max_value=0.80,
            value=float(bundle.get("default_threshold", 0.45)),
            step=0.02,
            help="Lower threshold boosts Recall (catching more delays to protect passenger connections); higher threshold boosts Precision.",
        )
        opt_f2 = art["eval"]["optimal_f2_ops_threshold"]["threshold"]
        st.caption(
            f"FAA Threshold: **Delay ≥ 15 min (Class 1)**\n\n"
            f"Recall-Optimal Cutoff (F2): **`{opt_f2:.2f}`** | Cost Benchmark: **`$101/min`**"
        )
        st.divider()
        st.subheader("👥 Module Ownership")
        st.markdown(
            "- **Parth** (`S24CSEU2470`): `ingestion.py`, `spark_clean_join.py`\n"
            "- **Vipul Gupta** (`S24CSEU2526`): `feature_engineering.py`\n"
            "- **Krishna Sharma** (`S24CSEU2756`): `balance_classes.py`, `train_models.py`\n"
            "- **Abhijeet Rana** (`S24CSEU1419`): `evaluate.py`, `app.py`"
        )

    tabs = st.tabs([
        "🛫 Live Flight Predictor & XAI",
        "📡 Hub Wave Radar & Batch CSV",
        "📊 Model Benchmarks & Threshold Lab",
        "⚖️ SMOTE & Weather Physics",
        "🗄️ SQL Warehouse Lab & Spark Plan",
    ])

    # =========================================================================
    # TAB 1: LIVE FLIGHT PREDICTOR, 1-CLICK CASE STUDIES, XAI & WHAT-IF
    # =========================================================================
    with tabs[0]:
        st.markdown("##### 🎬 1-Click Live Presentation Scenarios *(Click any scenario to auto-configure Flight + NOAA Weather)*")
        cs_cols = st.columns(len(DEMO_CASE_STUDIES))
        for col_btn, case_title in zip(cs_cols, DEMO_CASE_STUDIES.keys()):
            col_btn.button(
                case_title,
                key=f"btn_{case_title}",
                on_click=_apply_case_study,
                args=(case_title,),
                width="stretch",
            )

        if viva_mode and st.session_state.get("active_story"):
            st.markdown(
                f'<div class="viva-tip">🎙️ <b>Demo Spotlight:</b> {st.session_state["active_story"]}</div>',
                unsafe_allow_html=True,
            )

        c_left, c_right = st.columns([1.05, 1.35], gap="large")

        with c_left:
            st.subheader("1. Flight Schedule & Corridor")
            r1c1, r1c2 = st.columns(2)
            orig_label = r1c1.selectbox("Origin Hub Airport", AP_LABELS, key="orig_label_k")
            dest_label = r1c2.selectbox("Destination Airport", AP_LABELS, key="dest_label_k")
            origin_iata = AP_OPTIONS[orig_label]
            dest_iata = AP_OPTIONS[dest_label]

            if origin_iata == dest_iata:
                st.warning("Origin and Destination must differ — routing to ATL/LAX fallback.")
                dest_iata = "ATL" if origin_iata != "ATL" else "LAX"

            r2c1, r2c2 = st.columns(2)
            carrier_label = r2c1.selectbox("Operating Airline", AL_LABELS, key="carrier_label_k")
            carrier_code = AL_OPTIONS[carrier_label]
            fl_date = r2c2.date_input("Scheduled Date", value=date(2025, 3, 21))

            r3c1, r3c2 = st.columns(2)
            dep_hour = r3c1.slider("Departure Hour (24h)", 0, 23, key="dep_hour_k")
            dep_min = r3c2.selectbox("Minute Slot", [0, 15, 30, 45], key="dep_min_k")
            dep_time_val = dt_time(int(dep_hour), int(dep_min))

            traffic_label = st.select_slider(
                "Origin Hub Departure Bank Load",
                options=TRAFFIC_OPTIONS,
                key="traffic_label_k",
            )
            traffic_mult = TRAFFIC_MULT_MAP[traffic_label]

            t2_c1, t2_c2 = st.columns(2)
            prev_tail_delay = t2_c1.slider(
                "Inbound Tail Prev-Leg Delay @ T-2h (min)",
                min_value=-15.0, max_value=140.0, step=1.0, key="prev_delay_k",
                help="Known delay of this aircraft's prior flight leg on the same day (>=2 hours prior to departure).",
            )
            hub_t2h_pct = t2_c2.slider(
                "Origin Hub T-2h Delay Rate (%)",
                min_value=5, max_value=75, step=1, key="hub_t2h_k",
                help="Observed airport delay rate across departures 2–5 hours earlier.",
            )
            hub_t2h_rate = float(hub_t2h_pct) / 100.0

            st.subheader("2. Origin Surface Weather (NOAA / METAR)")
            wx_col1, wx_col2 = st.columns([1.55, 1.05])
            wx_col1.selectbox(
                "Weather Regime Preset",
                list(WEATHER_PRESETS.keys()),
                key="wx_preset_k",
                on_change=_apply_wx_preset,
            )

            orig_coords = next(a for a in AIRPORTS_REGISTRY if a["IATA"] == origin_iata)
            if wx_col2.button(f"🌐 Live {origin_iata} METAR", width="stretch"):
                live_res = fetch_live_airport_weather(orig_coords["LAT"], orig_coords["LON"])
                if live_res:
                    st.session_state["temp_k"] = live_res["temp_f"]
                    st.session_state["dew_k"] = live_res["dew_f"]
                    st.session_state["wind_k"] = live_res["wind_kt"]
                    st.session_state["vis_k"] = live_res["vis_mi"]
                    st.session_state["precip_k"] = live_res["precip_in"]
                    st.session_state["press_k"] = live_res["pressure_mb"]
                    st.session_state["wx_source"] = f"Live Open-Meteo Aviation Feed ({origin_iata})"
                    st.rerun()
                else:
                    st.info("Live feed offline in current network — using interactive sliders below.")

            st.caption(f"Weather Source: **{st.session_state.get('wx_source', 'Custom')}**")

            w1, w2, w3 = st.columns(3)
            temp_f = w1.number_input("Temp (°F)", min_value=-20.0, max_value=120.0, step=1.0, key="temp_k")
            dew_f = w2.number_input("Dew Point (°F)", min_value=-30.0, max_value=95.0, step=1.0, key="dew_k")
            wind_kt = w3.slider("Wind Speed (kt)", 0.0, 48.0, step=0.5, key="wind_k")

            w4, w5, w6 = st.columns(3)
            vis_mi = w4.slider("Visibility (mi)", 0.25, 10.0, step=0.25, key="vis_k")
            precip_in = w5.slider("Precip (in/hr)", 0.0, 1.5, step=0.02, key="precip_k")
            pressure_mb = w6.number_input("Pressure (mb)", min_value=975.0, max_value=1045.0, step=1.0, key="press_k")

        # Live Inference across all 4 classifiers + secondary delay-minutes regressor
        eff_dew = min(float(dew_f), float(temp_f))
        X_query, ctx = build_single_feature_vector(
            bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val,
            float(temp_f), eff_dew, float(wind_kt), float(vis_mi), float(precip_in), float(pressure_mb),
            traffic_mult, float(prev_tail_delay), hub_t2h_rate
        )

        model_probs = {
            m_name: float(clf.predict_proba(X_query)[0, 1])
            for m_name, clf in bundle["classifiers"].items()
        }
        primary_prob = model_probs["XGBoost Classifier (Tuned)"]
        lr_prob = model_probs["Logistic Regression (Baseline)"]
        raw_reg_min = float(bundle["delay_regressor"].predict(X_query)[0])
        is_delayed_pred = primary_prob >= decision_threshold
        expected_delay_min = max(16.0, raw_reg_min) if is_delayed_pred else max(0.0, raw_reg_min)

        # Business Intelligence Operational Cost & Connection Calculations (FAA/A4A $101.18/block-min)
        est_block_cost = round(expected_delay_min * 101.18) if is_delayed_pred else round(primary_prob * 12.0 * 101.18 * 0.25)
        connecting_pax_at_risk = int(round(175 * 0.84 * 0.28 * min(1.0, primary_prob * 1.15))) if is_delayed_pred else 0
        proactive_savings = round(est_block_cost * 0.38) if is_delayed_pred else 0
        tree_vs_linear_gap = (primary_prob - lr_prob) * 100.0

        with c_right:
            st.subheader("3. Real-Time ML Verdict & Business Intelligence Impact")
            if is_delayed_pred:
                st.markdown(
                    f"""
                    <div class="verdict-delayed">
                        <div style="font-size:0.82rem; text-transform:uppercase; font-weight:700; letter-spacing:0.04em;">
                            🚨 FAA Class 1 Alert — Departure Delay ≥ 15 Minutes Predicted
                        </div>
                        <div style="font-size:1.85rem; font-weight:800; margin:0.2rem 0;">
                            ⚠️ DELAY PROBABILITY: {primary_prob * 100:.1f}% &nbsp;|&nbsp; Est. Delay: +{expected_delay_min:.0f} min
                        </div>
                        <div style="font-size:0.9rem;">
                            Corridor <b>{origin_iata} → {dest_iata}</b> ({ctx['distance_mi']:.0f} mi) • Carrier <b>{carrier_code}</b> • Slot <b>{dep_time_val.strftime('%H:%M')}</b> ≥ Cutoff ({decision_threshold:.2f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-ontime">
                        <div style="font-size:0.82rem; text-transform:uppercase; font-weight:700; letter-spacing:0.04em;">
                            ✅ FAA Class 0 — Scheduled On-Time Departure (&lt; 15 Min)
                        </div>
                        <div style="font-size:1.85rem; font-weight:800; margin:0.2rem 0;">
                            🟢 ON-TIME EXPECTED &nbsp;|&nbsp; Delay Risk: {primary_prob * 100:.1f}% (Est. +{expected_delay_min:.0f} min)
                        </div>
                        <div style="font-size:0.9rem;">
                            Corridor <b>{origin_iata} → {dest_iata}</b> ({ctx['distance_mi']:.0f} mi) • Carrier <b>{carrier_code}</b> • Slot <b>{dep_time_val.strftime('%H:%M')}</b> &lt; Cutoff ({decision_threshold:.2f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # BI Operational Metrics Strip
            bi1, bi2, bi3, bi4 = st.columns(4)
            bi1.metric("Est. Delay Cost (A4A)", f"${est_block_cost:,}", "$101/min block burn")
            bi2.metric("Pax Misconnect Risk", f"{connecting_pax_at_risk} pax", "<45m hub layovers")
            bi3.metric("T-2h Dispatch Savings", f"${proactive_savings:,}", "38% early mitigation")
            bi4.metric("XGBoost vs Linear", f"{tree_vs_linear_gap:+.1f} pp", "Non-linear lift")

            g_col, m_col = st.columns([1.0, 1.15])
            with g_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(primary_prob * 100.0, 1),
                    number={"suffix": "%", "font": {"size": 33}},
                    title={"text": "Tuned XGBoost Risk Gauge", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#dc2626" if is_delayed_pred else "#059669"},
                        "steps": [
                            {"range": [0, decision_threshold * 100], "color": "#d1fae5"},
                            {"range": [decision_threshold * 100, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#0f172a", "width": 4},
                            "thickness": 0.82,
                            "value": decision_threshold * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(height=235, margin=dict(l=18, r=18, t=38, b=8))
                st.plotly_chart(fig_gauge, width="stretch")

            with m_col:
                comp_df = pd.DataFrame([
                    {"Model": k.replace(" Classifier", "").replace(" (Baseline)", ""), "Probability (%)": round(v * 100, 1)}
                    for k, v in model_probs.items()
                ])
                fig_comp = px.bar(
                    comp_df,
                    x="Probability (%)",
                    y="Model",
                    orientation="h",
                    text="Probability (%)",
                    range_x=[0, 108],
                    color="Probability (%)",
                    color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                )
                fig_comp.add_vline(x=decision_threshold * 100, line_dash="dash", line_color="#0f172a")
                fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_comp.update_layout(
                    title="4-Model Ensemble Consensus",
                    height=235,
                    margin=dict(l=8, r=28, t=38, b=8),
                    coloraxis_showscale=False,
                    yaxis_title="",
                )
                st.plotly_chart(fig_comp, width="stretch")

            # Actionable Operational Drivers
            drivers = []
            if wind_kt > 25.0:
                drivers.append(f"🌬️ **Crosswind Cliff ({wind_kt:.1f} kt > 25 kt)**: Active runway crosswind restriction at {origin_iata}.")
            if vis_mi < 3.0:
                drivers.append(f"🌫️ **IFR Low Visibility ({vis_mi:.2f} mi < 3.0 mi)**: Mandatory expanded ATC arrival/departure spacing.")
            if temp_f <= 33.5 and (precip_in >= 0.03 or (temp_f - eff_dew) < 2.8):
                drivers.append(f"❄️ **Freezing De-icing Regime ({temp_f:.0f}°F)**: Glycol pad queue required prior to taxi-out.")
            if dep_hour >= 16:
                drivers.append(f"🌇 **Evening Ripple Wave ({dep_time_val.strftime('%H:%M')})**: High downstream inbound tail-lag exposure.")
            if ctx["congestion_ratio"] > 1.05:
                drivers.append(f"🛫 **Elevated Hub Bank ({ctx['congestion_ratio']:.2f}x capacity, {ctx['roll_3h_vol']:.0f} rolling 3h deps)**: Departure queue buildup.")
            if not drivers:
                drivers.append("🟢 **Nominal VFR & Runway Profile**: Weather and departure wave load well within nominal hub capacity.")

            for d_msg in drivers:
                st.markdown(f"- {d_msg}")

        # ---------------------------------------------------------------------
        # INTERACTIVE DEEP-DIVE SUBTABS: XAI WATERFALL, 24H OPTIMIZER, WIND CURVE & MAP
        # ---------------------------------------------------------------------
        st.divider()
        deep_tabs = st.tabs([
            "🔍 Explainable AI (XAI) Risk Attribution",
            "⏰ 24-Hour 'Best Time to Fly' Optimizer",
            "🌬️ Non-Linear Wind Sensitivity (XGBoost vs Linear)",
            "🗺️ Corridor Map & 24h Cyclical Clock",
        ])

        with deep_tabs[0]:
            wf_df = compute_xai_waterfall(
                bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val,
                float(temp_f), eff_dew, float(wind_kt), float(vis_mi), float(precip_in), float(pressure_mb),
                traffic_mult, float(prev_tail_delay), hub_t2h_rate
            )
            fig_wf = go.Figure(go.Waterfall(
                name="Risk Attribution",
                orientation="v",
                measure=wf_df["Measure"].tolist(),
                x=wf_df["Factor"].tolist(),
                y=wf_df["Delta"].tolist(),
                text=[f"{v:+.1f}%" if m == "relative" else f"{v:.1f}%" for v, m in zip(wf_df["Delta"], wf_df["Measure"])],
                textposition="outside",
                connector={"line": {"color": "#64748b"}},
                increasing={"marker": {"color": "#ef4444"}},
                decreasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#1e3a8a"}},
            ))
            fig_wf.add_hline(y=decision_threshold * 100, line_dash="dash", line_color="#dc2626", annotation_text=f"Decision Cutoff ({decision_threshold*100:.0f}%)")
            fig_wf.update_layout(
                title=f"Telescoping Probability Decomposition: Why is {carrier_code} ({origin_iata} → {dest_iata}) at {primary_prob*100:.1f}% Risk?",
                yaxis_title="Delay Probability (%)",
                yaxis_range=[0, max(105.0, float(wf_df["Delta"].max()) + 18.0)],
                height=370,
                margin=dict(l=20, r=20, t=48, b=20),
            )
            st.plotly_chart(fig_wf, width="stretch")

        with deep_tabs[1]:
            # Sweep all 24 departure hours for this exact flight + weather
            hr_frames = []
            for h_cand in range(24):
                Xh, _ = build_single_feature_vector(
                    bundle, origin_iata, dest_iata, carrier_code, fl_date, dt_time(h_cand, int(dep_min)),
                    float(temp_f), eff_dew, float(wind_kt), float(vis_mi), float(precip_in), float(pressure_mb), traffic_mult
                )
                hr_frames.append(Xh)
            X_24h = pd.concat(hr_frames, ignore_index=True)
            probs_24h = bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(X_24h)[:, 1] * 100.0
            best_hr = int(np.argmin(probs_24h))
            worst_hr = int(np.argmax(probs_24h))
            risk_drop = primary_prob * 100.0 - float(probs_24h[best_hr])

            opt_c1, opt_c2, opt_c3 = st.columns(3)
            opt_c1.metric("Safest Departure Slot", f"{best_hr:02d}:{int(dep_min):02d}", f"{probs_24h[best_hr]:.1f}% Risk")
            opt_c2.metric("Peak Cascading Risk Slot", f"{worst_hr:02d}:{int(dep_min):02d}", f"{probs_24h[worst_hr]:.1f}% Risk", delta_color="inverse")
            opt_c3.metric("Potential Risk Reduction", f"-{max(0.0, risk_drop):.1f} pp", f"If shifted from {dep_time_val.strftime('%H:%M')} → {best_hr:02d}:{int(dep_min):02d}")

            fig_24h = go.Figure()
            fig_24h.add_trace(go.Scatter(
                x=[f"{h:02d}:{int(dep_min):02d}" for h in range(24)],
                y=np.round(probs_24h, 1),
                mode="lines+markers",
                fill="tozeroy",
                line=dict(color="#2563eb", width=3),
                name="XGBoost Risk (%)",
            ))
            fig_24h.add_trace(go.Scatter(
                x=[f"{int(dep_hour):02d}:{int(dep_min):02d}"],
                y=[round(primary_prob * 100.0, 1)],
                mode="markers",
                marker=dict(size=15, color="#dc2626", symbol="diamond"),
                name="Selected Slot",
            ))
            fig_24h.add_hline(y=decision_threshold * 100, line_dash="dash", line_color="#dc2626", annotation_text="Delay Threshold")
            fig_24h.update_layout(
                title=f"24-Hour 'What-If' Schedule Sweep ({origin_iata} → {dest_iata} under Active Weather)",
                xaxis_title="Scheduled Departure Slot (24-Hour Cycle)",
                yaxis_title="Predicted Delay Probability (%)",
                yaxis_range=[0, 102],
                height=330,
                margin=dict(l=20, r=20, t=42, b=20),
            )
            st.plotly_chart(fig_24h, width="stretch")

        with deep_tabs[2]:
            wind_grid = np.arange(0.0, 46.0, 1.5)
            w_frames = [
                build_single_feature_vector(
                    bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val,
                    float(temp_f), eff_dew, float(w_val), float(vis_mi), float(precip_in), float(pressure_mb), traffic_mult
                )[0]
                for w_val in wind_grid
            ]
            X_w_sweep = pd.concat(w_frames, ignore_index=True)
            xgb_w_curve = bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(X_w_sweep)[:, 1] * 100.0
            rf_w_curve = bundle["classifiers"]["Random Forest (100 Trees)"].predict_proba(X_w_sweep)[:, 1] * 100.0
            lr_w_curve = bundle["classifiers"]["Logistic Regression (Baseline)"].predict_proba(X_w_sweep)[:, 1] * 100.0

            fig_ws = go.Figure()
            fig_ws.add_trace(go.Scatter(x=wind_grid, y=xgb_w_curve, name="Tuned XGBoost (Captures >24 kt Step)", line=dict(color="#dc2626", width=3.2)))
            fig_ws.add_trace(go.Scatter(x=wind_grid, y=rf_w_curve, name="Random Forest (100 Trees)", line=dict(color="#0284c7", width=2.4)))
            fig_ws.add_trace(go.Scatter(x=wind_grid, y=lr_w_curve, name="Logistic Regression (Linear Approximation)", line=dict(color="#64748b", width=2.4, dash="dot")))
            fig_ws.add_vline(x=25.0, line_dash="dash", line_color="#f59e0b", annotation_text="FAA 25 kt Crosswind Limit")
            fig_ws.add_vline(x=float(wind_kt), line_color="#0f172a", annotation_text=f"Current={wind_kt:.1f} kt")
            fig_ws.update_layout(
                title="Live Sensitivity Proof: Why Tree Ensembles Outperform Linear Models on Surface Weather",
                xaxis_title="Origin Surface Wind Speed (knots)",
                yaxis_title="Predicted Delay Probability (%)",
                yaxis_range=[0, 103],
                height=350,
                margin=dict(l=20, r=20, t=42, b=20),
            )
            st.plotly_chart(fig_ws, width="stretch")

        with deep_tabs[3]:
            map_col, clock_col = st.columns([1.45, 1.0], gap="large")
            with map_col:
                ap_df_all = pd.DataFrame(AIRPORTS_REGISTRY)
                route_color = "#dc2626" if is_delayed_pred else "#059669"
                fig_map = go.Figure()
                fig_map.add_trace(go.Scattergeo(
                    lon=ap_df_all["LON"], lat=ap_df_all["LAT"],
                    text=ap_df_all["IATA"] + " — " + ap_df_all["CITY"],
                    mode="markers+text", textposition="top center",
                    marker=dict(size=8, color="#334155", opacity=0.72),
                    name="20 U.S. Hubs",
                ))
                fig_map.add_trace(go.Scattergeo(
                    lon=[ctx["orig_ap"]["LON"], ctx["dest_ap"]["LON"]],
                    lat=[ctx["orig_ap"]["LAT"], ctx["dest_ap"]["LAT"]],
                    mode="lines+markers",
                    line=dict(width=4.5, color=route_color),
                    marker=dict(size=13, color=route_color),
                    name=f"{origin_iata} → {dest_iata} ({ctx['distance_mi']:.0f} mi)",
                ))
                fig_map.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#f1f5f9", subunitcolor="#cbd5e1")
                fig_map.update_layout(title=f"Active Corridor: {origin_iata} ({ctx['orig_ap']['CITY']}) → {dest_iata} ({ctx['dest_ap']['CITY']})", height=340, margin=dict(l=0, r=0, t=38, b=0))
                st.plotly_chart(fig_map, width="stretch")

            with clock_col:
                theta_grid = np.linspace(0, 2 * np.pi, 200)
                sin_val = float(X_query["dep_hour_sin"].iloc[0])
                cos_val = float(X_query["dep_hour_cos"].iloc[0])
                fig_clock = go.Figure()
                fig_clock.add_trace(go.Scatter(x=np.cos(theta_grid), y=np.sin(theta_grid), mode="lines", line=dict(color="#94a3b8", dash="dot"), name="24h Unit Circle"))
                for hr_mark in [0, 6, 12, 18]:
                    ang = 2 * math.pi * (hr_mark / 24.0)
                    fig_clock.add_trace(go.Scatter(
                        x=[math.cos(ang)], y=[math.sin(ang)], mode="markers+text",
                        text=[f"{hr_mark:02d}:00"], textposition="top center",
                        marker=dict(size=7, color="#475569"), showlegend=False,
                    ))
                fig_clock.add_trace(go.Scatter(
                    x=[0, cos_val], y=[0, sin_val], mode="lines+markers",
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=13, color="#dc2626" if is_delayed_pred else "#2563eb"),
                    name=f"{dep_time_val.strftime('%H:%M')} (cos={cos_val:.2f}, sin={sin_val:.2f})",
                ))
                fig_clock.update_layout(
                    title="Cyclical Timestamp Encoding (Vipul Gupta Module)",
                    xaxis=dict(range=[-1.35, 1.35], title="cos(2π·h/24)"),
                    yaxis=dict(range=[-1.35, 1.35], title="sin(2π·h/24)", scaleanchor="x", scaleratio=1),
                    height=340, margin=dict(l=20, r=20, t=38, b=20), legend=dict(orientation="h", y=-0.18),
                )
                st.plotly_chart(fig_clock, width="stretch")

    # =========================================================================
    # TAB 2: HUB WAVE RADAR MAP & CUSTOM BATCH CSV SCANNER
    # =========================================================================
    with tabs[1]:
        st.subheader("📡 AOCC National Hub Departure Wave Radar & Batch Schedule Scanner")
        if viva_mode:
            st.markdown(
                '<div class="viva-tip">🎙️ <b>Demo Spotlight:</b> Dispatchers rarely check one flight at a time — here the trained XGBoost pipeline scores an entire 19-spoke departure bank simultaneously, mapping network ripple risk across the U.S.</div>',
                unsafe_allow_html=True,
            )

        b1, b2, b3, b4 = st.columns(4)
        hub_iata = b1.selectbox("Origin Hub", [a["IATA"] for a in AIRPORTS_REGISTRY], index=1, key="board_hub")
        board_wx_name = b2.selectbox("Hub Surface Weather", list(WEATHER_PRESETS.keys()), index=1, key="board_wx")
        wave_start_hr = b3.slider("Departure Bank Hour", 5, 21, 18, key="board_hr")
        board_traffic = b4.selectbox("Runway Load Factor", [0.85, 1.0, 1.25, 1.50], index=2, format_func=lambda x: f"{x:.2f}x Hub Load")

        bwx = WEATHER_PRESETS[board_wx_name]
        ap_lookup = {a["IATA"]: a for a in AIRPORTS_REGISTRY}
        hub_ap = ap_lookup[hub_iata]
        rng_board = np.random.default_rng(int(ord(hub_iata[0]) * 100 + wave_start_hr))
        other_airports = [a["IATA"] for a in AIRPORTS_REGISTRY if a["IATA"] != hub_iata][:24]
        carriers_list = [c["CARRIER_CODE"] for c in AIRLINES_REGISTRY]

        board_rows = []
        feature_frames = []
        for i, d_iata in enumerate(other_airports):
            c_code = carriers_list[int(rng_board.integers(0, len(carriers_list)))]
            slot_hr = min(23, wave_start_hr + (i // 7))
            slot_min = int((i * 5) % 60)
            fl_no = f"{c_code}{1140 + i * 39}"
            X_b, ctx_b = build_single_feature_vector(
                bundle, hub_iata, d_iata, c_code, date(2025, 3, 22), dt_time(slot_hr, slot_min),
                bwx["temp"], bwx["dew"], bwx["wind"], bwx["vis"], bwx["precip"], bwx["press"], board_traffic
            )
            feature_frames.append(X_b)
            board_rows.append({
                "Flight": fl_no,
                "Carrier": c_code,
                "Origin": hub_iata,
                "Dest": d_iata,
                "Dest City": ap_lookup[d_iata]["CITY"],
                "Sched Dep": f"{slot_hr:02d}:{slot_min:02d}",
                "Distance (mi)": int(ctx_b["distance_mi"]),
                "dest_lat": ap_lookup[d_iata]["LAT"],
                "dest_lon": ap_lookup[d_iata]["LON"],
            })

        X_batch = pd.concat(feature_frames, ignore_index=True)
        batch_probs = bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(X_batch)[:, 1]
        batch_mins = bundle["delay_regressor"].predict(X_batch)

        board_df = pd.DataFrame(board_rows)
        board_df["Delay Prob (%)"] = np.round(batch_probs * 100.0, 1)
        board_df["Est. Delay (min)"] = np.where(
            batch_probs >= decision_threshold,
            np.maximum(16, np.round(batch_mins, 0)).astype(int),
            np.maximum(0, np.round(batch_mins, 0)).astype(int),
        )
        board_df["Est. Cost ($)"] = np.where(
            batch_probs >= decision_threshold,
            (board_df["Est. Delay (min)"] * 101.18).round(0).astype(int),
            0,
        )
        board_df["Dispatch Status"] = np.where(
            batch_probs >= decision_threshold, "🔴 DELAY RISK (≥15m)", "🟢 ON-TIME"
        )

        flagged_cnt = int((batch_probs >= decision_threshold).sum())
        total_wave_cost = int(board_df["Est. Cost ($)"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Outbound Spoke Flights", f"{len(board_df)} Flights", f"Origin Hub: {hub_iata}")
        m2.metric("Flagged High-Risk Flights", f"{flagged_cnt} / {len(board_df)}", f"Cutoff: {decision_threshold:.2f}")
        m3.metric("Mean Bank Delay Risk", f"{batch_probs.mean() * 100:.1f}%", board_wx_name)
        m4.metric("Wave Financial Exposure", f"${total_wave_cost:,}", "Direct Airline Block Cost")

        # Hub-and-Spoke Radar Map
        fig_radar = go.Figure()
        for _, r_spoke in board_df.iterrows():
            delayed_spoke = r_spoke["Delay Prob (%)"] >= (decision_threshold * 100.0)
            fig_radar.add_trace(go.Scattergeo(
                lon=[hub_ap["LON"], r_spoke["dest_lon"]],
                lat=[hub_ap["LAT"], r_spoke["dest_lat"]],
                mode="lines",
                line=dict(width=2.8 if delayed_spoke else 1.6, color="#ef4444" if delayed_spoke else "#10b981"),
                opacity=0.85 if delayed_spoke else 0.45,
                hoverinfo="text",
                text=f"{r_spoke['Flight']} ({hub_iata}→{r_spoke['Dest']}): {r_spoke['Delay Prob (%)']}%",
                showlegend=False,
            ))
        fig_radar.add_trace(go.Scattergeo(
            lon=board_df["dest_lon"],
            lat=board_df["dest_lat"],
            mode="markers+text",
            text=board_df["Dest"],
            textposition="top center",
            marker=dict(
                size=9,
                color=np.where(board_df["Delay Prob (%)"] >= decision_threshold * 100, "#dc2626", "#059669"),
            ),
            name="Spoke Destinations",
        ))
        fig_radar.add_trace(go.Scattergeo(
            lon=[hub_ap["LON"]], lat=[hub_ap["LAT"]],
            mode="markers+text", text=[f"HUB: {hub_iata}"], textposition="bottom center",
            marker=dict(size=16, color="#1d4ed8", symbol="star"),
            name=f"Origin Hub ({hub_iata})",
        ))
        fig_radar.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#f8fafc", subunitcolor="#cbd5e1")
        fig_radar.update_layout(
            title=f"Live National Outbound Spoke Risk Radar from {hub_iata} ({hub_ap['CITY']}) — Red = Flagged Delay ≥15m | Green = On-Time",
            height=390, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_radar, width="stretch")

        disp_cols = ["Flight", "Carrier", "Origin", "Dest", "Dest City", "Sched Dep", "Distance (mi)", "Delay Prob (%)", "Est. Delay (min)", "Est. Cost ($)", "Dispatch Status"]
        board_sorted = board_df[disp_cols].sort_values("Delay Prob (%)", ascending=False).reset_index(drop=True)
        st.dataframe(board_sorted, width="stretch", hide_index=True)

        dl_c1, dl_c2 = st.columns([1, 2])
        dl_c1.download_button(
            "📥 Export Scored Hub Wave CSV",
            data=board_sorted.to_csv(index=False).encode("utf-8"),
            file_name=f"aocc_hub_wave_{hub_iata}.csv",
            mime="text/csv",
            width="stretch",
        )

        st.divider()
        st.markdown("#### 📂 Custom Schedule CSV Batch Scorer *(Upload Your Own Flights)*")
        sample_template = pd.DataFrame([
            {"ORIGIN": "ORD", "DEST": "JFK", "CARRIER": "UA", "DEP_HOUR": 18, "DEP_MIN": 30, "TEMP_F": 31.0, "DEW_F": 29.0, "WIND_KT": 28.0, "VIS_MI": 1.5, "PRECIP_IN": 0.18},
            {"ORIGIN": "ATL", "DEST": "LAX", "CARRIER": "DL", "DEP_HOUR": 8, "DEP_MIN": 0, "TEMP_F": 66.0, "DEW_F": 48.0, "WIND_KT": 7.0, "VIS_MI": 10.0, "PRECIP_IN": 0.00},
            {"ORIGIN": "EWR", "DEST": "MCO", "CARRIER": "B6", "DEP_HOUR": 19, "DEP_MIN": 45, "TEMP_F": 58.0, "DEW_F": 54.0, "WIND_KT": 26.5, "VIS_MI": 2.5, "PRECIP_IN": 0.12},
        ])
        up_c1, up_c2 = st.columns([1, 2.2])
        up_c1.download_button(
            "⬇️ Download Sample Input Template CSV",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="sample_flights_to_score.csv",
            mime="text/csv",
            width="stretch",
        )
        uploaded_csv = up_c2.file_uploader("Upload CSV with ORIGIN, DEST, CARRIER, DEP_HOUR, DEP_MIN, TEMP_F, DEW_F, WIND_KT, VIS_MI, PRECIP_IN", type=["csv"])
        if uploaded_csv is not None:
            user_df = pd.read_csv(uploaded_csv)
            u_frames = []
            valid_airports = {a["IATA"] for a in AIRPORTS_REGISTRY}
            for _, urow in user_df.iterrows():
                o_i = str(urow.get("ORIGIN", "ORD")).strip().upper()
                d_i = str(urow.get("DEST", "JFK")).strip().upper()
                o_i = o_i if o_i in valid_airports else "ORD"
                d_i = d_i if (d_i in valid_airports and d_i != o_i) else ("JFK" if o_i != "JFK" else "ATL")
                Xu, _ = build_single_feature_vector(
                    bundle, o_i, d_i, str(urow.get("CARRIER", "UA")).strip().upper(),
                    date(2025, 3, 22), dt_time(int(urow.get("DEP_HOUR", 12)) % 24, int(urow.get("DEP_MIN", 0)) % 60),
                    float(urow.get("TEMP_F", 62.0)), float(urow.get("DEW_F", 45.0)),
                    float(urow.get("WIND_KT", 10.0)), float(urow.get("VIS_MI", 10.0)),
                    float(urow.get("PRECIP_IN", 0.0)), 1015.0, 1.0
                )
                u_frames.append(Xu)
            Xu_all = pd.concat(u_frames, ignore_index=True)
            u_probs = bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(Xu_all)[:, 1]
            user_df["Predicted_Delay_Prob_%"] = np.round(u_probs * 100.0, 1)
            user_df["FAA_Class1_Verdict"] = np.where(u_probs >= decision_threshold, "DELAYED (>=15m)", "ON-TIME (<15m)")
            st.success(f"Scored {len(user_df)} custom uploaded flight rows!")
            st.dataframe(user_df, width="stretch", hide_index=True)

    # =========================================================================
    # TAB 3: MODEL BENCHMARKS & INTERACTIVE THRESHOLD LAB
    # =========================================================================
    with tabs[2]:
        st.subheader("📊 Untouched 20% Chronological Test Split — Model Architecture Benchmarks")
        if viva_mode:
            st.markdown(
                '<div class="viva-tip">🎙️ <b>Demo Spotlight:</b> Chronological 80/20 splitting ensures zero future weather leakage. Drag the sidebar Threshold slider to watch the Confusion Matrix and False-Negative connection misses update live!</div>',
                unsafe_allow_html=True,
            )
        bench_df = pd.DataFrame(art["eval"]["benchmarks"])
        st.dataframe(bench_df, width="stretch", hide_index=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            fig_roc = go.Figure()
            for m_name, cdata in art["eval"]["curves"].items():
                fig_roc.add_trace(go.Scatter(x=cdata["fpr"], y=cdata["tpr"], mode="lines", name=f"{m_name} (AUC={cdata['roc_auc']:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random (0.500)"))
            fig_roc.update_layout(title="ROC-AUC Curves (Untouched Future Test Split)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=380)
            st.plotly_chart(fig_roc, width="stretch")

        with rc2:
            fig_pr = go.Figure()
            for m_name, cdata in art["eval"]["curves"].items():
                fig_pr.add_trace(go.Scatter(x=cdata["pr_recall"], y=cdata["pr_precision"], mode="lines", name=f"{m_name} (AP={cdata['pr_auc']:.3f})"))
            fig_pr.update_layout(title="Precision-Recall Curves (80/20 Imbalanced Test Split)", xaxis_title="Recall", yaxis_title="Precision", height=380)
            st.plotly_chart(fig_pr, width="stretch")

        st.divider()
        st.subheader("🎚️ Interactive Operational Threshold Simulator (Abhijeet Rana Module)")
        th_col1, th_col2 = st.columns([1.15, 1.0], gap="large")
        with th_col1:
            th_df = pd.DataFrame(art["eval"]["threshold_tuning"])
            fig_th = go.Figure()
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["recall"], name="Recall (Catch Rate)", line=dict(color="#dc2626", width=3)))
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["precision"], name="Precision", line=dict(color="#2563eb", width=3)))
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f1"], name="F1-Score", line=dict(color="#059669", width=2.5)))
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f2_ops"], name="F2-Score (Ops Priority)", line=dict(color="#7c3aed", dash="dot", width=2.5)))
            fig_th.add_vline(x=decision_threshold, line_dash="dash", line_color="#0f172a", annotation_text=f"Active={decision_threshold:.2f}")
            fig_th.update_layout(title="Precision / Recall / F1 / F2 vs. Decision Threshold", xaxis_title="Probability Threshold", yaxis_title="Metric Score", height=360)
            st.plotly_chart(fig_th, width="stretch")

        with th_col2:
            y_t = art["y_true_sample"]
            y_p = art["y_prob_sample"]
            pred_sim = (y_p >= decision_threshold).astype(int)
            tn = int(((y_t == 0) & (pred_sim == 0)).sum())
            fp = int(((y_t == 0) & (pred_sim == 1)).sum())
            fn = int(((y_t == 1) & (pred_sim == 0)).sum())
            tp = int(((y_t == 1) & (pred_sim == 1)).sum())

            fig_cm = px.imshow(
                [[tn, fp], [fn, tp]],
                x=["Pred On-Time (0)", "Pred Delayed (1)"],
                y=["Actual On-Time (0)", "Actual Delayed (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Live Confusion Matrix @ Threshold = {decision_threshold:.2f} (n={len(y_t):,} test sample)",
            )
            fig_cm.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, width="stretch")

        st.subheader("🏆 Tuned XGBoost Feature Importance & Target Correlation Rankings")
        feat_rank_df = pd.DataFrame(art["eval"]["feature_rankings"])
        fig_fi = px.bar(
            feat_rank_df.head(14).iloc[::-1],
            x="xgb_importance", y="display_name", orientation="h",
            color="target_correlation", color_continuous_scale="RdBu_r",
            labels={"xgb_importance": "XGBoost Relative Gain Importance", "display_name": "Engineered Feature", "target_correlation": "Corr w/ Delay"},
            title="Top 14 Predictive Features (Weather Thresholds + Rolling Hub Volume + Cyclical Time)",
        )
        fig_fi.update_layout(height=420)
        st.plotly_chart(fig_fi, width="stretch")

    # =========================================================================
    # TAB 4: SMOTE & EMPIRICAL WEATHER PHYSICS
    # =========================================================================
    with tabs[3]:
        st.subheader("⚖️ Addressing the 80/20 Class Skew — Strict Training-Fold SMOTE (Krishna Sharma Module)")
        if viva_mode:
            st.markdown(
                '<div class="viva-tip">🎙️ <b>Demo Spotlight:</b> Per Belcastro et al. (2016), SMOTE is applied strictly to the 80% chronological training split (+42,749 synthetic delayed samples), while the 20% test split remains 100% untouched.</div>',
                unsafe_allow_html=True,
            )
        sm = art["smote"]
        sm_df = pd.DataFrame([
            {"Partition": "1. Train (Before SMOTE)", "Class": "On-Time (<15m)", "Flights": sm["train_before_smote"]["on_time_class_0"]},
            {"Partition": "1. Train (Before SMOTE)", "Class": "Delayed (≥15m)", "Flights": sm["train_before_smote"]["delayed_class_1"]},
            {"Partition": "2. Train (After SMOTE)", "Class": "On-Time (<15m)", "Flights": sm["train_after_smote"]["on_time_class_0"]},
            {"Partition": "2. Train (After SMOTE)", "Class": "Delayed (≥15m)", "Flights": sm["train_after_smote"]["delayed_class_1"]},
            {"Partition": "3. Chronological Test (Untouched)", "Class": "On-Time (<15m)", "Flights": sm["test_untouched"]["on_time_class_0"]},
            {"Partition": "3. Chronological Test (Untouched)", "Class": "Delayed (≥15m)", "Flights": sm["test_untouched"]["delayed_class_1"]},
        ])
        fig_smote = px.bar(
            sm_df, x="Partition", y="Flights", color="Class", barmode="group", text_auto=",d",
            color_discrete_map={"On-Time (<15m)": "#0ea5e9", "Delayed (≥15m)": "#ef4444"},
            title=f"Class Balance Progression (+{sm['train_after_smote']['synthetic_samples_added']:,} Synthetic Delayed Vectors in Training Partition Only)",
        )
        fig_smote.update_layout(height=360)
        st.plotly_chart(fig_smote, width="stretch")

        st.divider()
        st.subheader("🌩️ Empirical Non-Linear Weather & Ripple Patterns Across 88,292 Flights")
        fdf = art["featured_df"]
        a1, a2 = st.columns(2)
        with a1:
            wind_bins = pd.cut(fdf["WIND_SPEED_KT"], bins=[0, 8, 15, 22, 28, 50], labels=["0–8 kt", "8–15 kt", "15–22 kt", "22–28 kt", ">28 kt (Gale)"])
            wind_grp = fdf.groupby(wind_bins, observed=False)["DEP_DEL15"].mean().reset_index()
            wind_grp["Delay Rate (%)"] = np.round(wind_grp["DEP_DEL15"] * 100.0, 1)
            fig_w = px.bar(
                wind_grp, x="WIND_SPEED_KT", y="Delay Rate (%)", text="Delay Rate (%)",
                color="Delay Rate (%)", color_continuous_scale="YlOrRd",
                title="Empirical Wind Threshold Step-Function (Choi et al., 2016)",
                labels={"WIND_SPEED_KT": "Origin Surface Wind Speed Bucket"},
            )
            fig_w.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_w.update_layout(height=355, yaxis_range=[0, max(wind_grp["Delay Rate (%)"]) * 1.22], coloraxis_showscale=False)
            st.plotly_chart(fig_w, width="stretch")

        with a2:
            hr_bucket = fdf["dep_hour_float"].astype(int)
            hr_grp = fdf.groupby(hr_bucket)["DEP_DEL15"].mean().reset_index()
            hr_grp["Delay Rate (%)"] = np.round(hr_grp["DEP_DEL15"] * 100.0, 1)
            fig_h = px.area(
                hr_grp, x="dep_hour_float", y="Delay Rate (%)", markers=True,
                title="Empirical 24-Hour Downstream Ripple Wave (Rebollo & Balakrishnan, 2014)",
                labels={"dep_hour_float": "Scheduled Departure Hour (0–23)"},
            )
            fig_h.update_traces(line_color="#2563eb")
            fig_h.update_layout(height=355)
            st.plotly_chart(fig_h, width="stretch")

    # =========================================================================
    # TAB 5: INTERACTIVE SQL WAREHOUSE PLAYGROUND, FUNNEL & PYSPARK PLAN
    # =========================================================================
    with tabs[4]:
        st.subheader("🗄️ Big Data Warehouse Funnel, Live SQL Lab & PySpark Execution Plan (Parth Module)")
        if viva_mode:
            st.markdown(
                '<div class="viva-tip">🎙️ <b>Demo Spotlight:</b> Run live relational SQL queries joining the 90,000-row BTS flight warehouse with the 43,200-row NOAA weather table right inside the browser!</div>',
                unsafe_allow_html=True,
            )

        # Data Pipeline Record Funnel
        raw_n = art["clean_join"]["raw_rows"]
        after_canc = raw_n - art["clean_join"]["cancelled_dropped"]
        after_div = after_canc - art["clean_join"]["diverted_dropped"]
        clean_n = art["clean_join"]["joined_clean_rows"]
        train_raw_n = art["smote"].get("full_train_partition", art["smote"]["train_before_smote"])["total"]
        train_smote_n = art["smote"]["train_after_smote"]["total"]

        fig_funnel = go.Figure(go.Funnel(
            y=[
                "1. Raw Ingested BTS Partitions (3 CSVs)",
                "2. Active Flights (Filtered Cancelled == 0)",
                "3. Non-Diverted Domestic Flights",
                "4. PySpark Clean & NOAA Weather-Joined",
                "5. Chronological 80% Training Partition",
                "6. SMOTE-Balanced Training Vectors (50/50)",
            ],
            x=[raw_n, after_canc, after_div, clean_n, train_raw_n, train_smote_n],
            textinfo="value+percent initial",
            marker={"color": ["#1e3a8a", "#1d4ed8", "#0284c7", "#0d9488", "#059669", "#7c3aed"]},
        ))
        fig_funnel.update_layout(title="7-Stage Big Data Record Volume & Cleaning Funnel", height=330, margin=dict(l=20, r=20, t=42, b=15))
        st.plotly_chart(fig_funnel, width="stretch")

        st.markdown("#### ⚡ Live Relational SQL Warehouse Playground (`bts_flights` ⋈ `noaa_weather_hourly` ⋈ `airports_metadata`)")
        selected_q_name = st.selectbox("Select Analytical SQL Query Preset", list(PRESET_SQL_QUERIES.keys()))
        sql_text = st.text_area(
            "Editable Read-Only SQL Query",
            value=PRESET_SQL_QUERIES[selected_q_name],
            height=165,
            key=f"sql_editor_{selected_q_name}",
        )

        if st.button("▶️ Execute SQL Query on Indexed Warehouse", type="primary"):
            pass  # Always render active query cleanly below

        stripped_sql = sql_text.strip().upper()
        if not stripped_sql.startswith("SELECT"):
            st.error("Read-only mode: only SELECT analytical queries are permitted in the live demo.")
        else:
            try:
                sql_res_df = execute_cached_sql(sql_text)
                sq1, sq2 = st.columns([1.15, 1.0], gap="large")
                with sq1:
                    st.dataframe(sql_res_df, width="stretch", hide_index=True)
                with sq2:
                    if len(sql_res_df.columns) >= 2 and "delay_rate_pct" in sql_res_df.columns:
                        x_col = sql_res_df.columns[0]
                        fig_sql = px.bar(
                            sql_res_df, x=x_col, y="delay_rate_pct", text="delay_rate_pct",
                            color="delay_rate_pct", color_continuous_scale="YlOrRd",
                            title=f"Live SQL Result: Delay Rate (%) by {x_col}",
                        )
                        fig_sql.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        fig_sql.update_layout(height=310, coloraxis_showscale=False)
                        st.plotly_chart(fig_sql, width="stretch")
            except Exception as sql_err:
                st.error(f"SQL Query Error: {sql_err}")

        st.divider()
        with st.expander("📜 View PySpark SQL & Catalyst Physical Execution Plan (`outputs/plans/pyspark_execution_plan.txt`)", expanded=False):
            st.code(art["plan_text"], language="text")

        with st.expander("🐘 View Distributed Partitioned PostgreSQL DDL (`sql/schema_postgres.sql`)", expanded=False):
            st.code(art["sql_ddl"], language="sql")

        with st.expander("🖼️ View Exported Static Publication Plots (`outputs/plots/`)", expanded=False):
            p1, p2 = st.columns(2)
            p1.image(str(PLOTS_DIR / "confusion_matrices.png"), caption="Confusion Matrices Across Candidate Classifiers")
            p2.image(str(PLOTS_DIR / "roc_pr_curves.png"), caption="ROC-AUC & Precision-Recall Curves")
            p3, p4 = st.columns(2)
            p3.image(str(PLOTS_DIR / "threshold_tuning.png"), caption="Operational Decision Threshold Tuning")
            p4.image(str(PLOTS_DIR / "feature_importance.png"), caption="Tuned XGBoost Feature Importance Rankings")


if __name__ == "__main__":
    main()
