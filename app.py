"""
Stage 7 — Minimalist SKY / MOVE Flight Delay Intelligence Studio (app.py)
Course: Big Data Analytics and Business Intelligence (CBCA 293) — Bennett University
"""

from __future__ import annotations
import json
from datetime import date, time as dt_time
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import (
    AIRPORTS_REGISTRY,
    AIRLINES_REGISTRY,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from feature_engineering import encode_cyclical_time
from generate_datasets import haversine_miles


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


@st.cache_resource
def load_model_bundle() -> dict:
    bundle = joblib.load(MODELS_DIR / "flight_delay_models.joblib")
    for clf in bundle["classifiers"].values():
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = 1
    if hasattr(bundle["delay_regressor"], "n_jobs"):
        bundle["delay_regressor"].n_jobs = 1
    return bundle


@st.cache_data
def load_summary_artifacts() -> dict:
    with open(OUTPUTS_DIR / "ingestion_audit.json", "r", encoding="utf-8") as f:
        ing = json.load(f)
    with open(OUTPUTS_DIR / "clean_join_summary.json", "r", encoding="utf-8") as f:
        cln = json.load(f)
    with open(OUTPUTS_DIR / "evaluation_report.json", "r", encoding="utf-8") as f:
        evl = json.load(f)
    return {"ingestion": ing, "clean_join": cln, "eval": evl}


WX_SIMPLE = {
    "Clear Sky": {"temp": 66.0, "dew": 46.0, "wind": 7.0, "vis": 10.0, "precip": 0.00, "press": 1018.0},
    "Windy (31 kt)": {"temp": 52.0, "dew": 38.0, "wind": 31.0, "vis": 8.5, "precip": 0.02, "press": 1005.0},
    "Thunderstorm": {"temp": 76.0, "dew": 72.0, "wind": 24.0, "vis": 2.0, "precip": 0.35, "press": 1004.0},
    "Winter Snow": {"temp": 25.0, "dew": 23.0, "wind": 27.0, "vis": 0.8, "precip": 0.30, "press": 1003.0},
    "Dense Fog": {"temp": 52.0, "dew": 51.0, "wind": 6.0, "vis": 0.5, "precip": 0.01, "press": 1016.0},
}


def main() -> None:
    st.set_page_config(page_title="SKY / MOVE — Flight Delay Intelligence", page_icon="✈️", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1a16; color: #f3f7f4; }
        section[data-testid="stSidebar"] { background-color: #0b1512; border-right: 1px solid rgba(255,255,255,0.09); }
        .eyebrow { font-family: monospace; font-size: 0.75rem; letter-spacing: 0.14em; color: #d4ff32; text-transform: uppercase; }
        .hero-h1 { font-size: 3.6rem; font-weight: 800; line-height: 1.02; letter-spacing: -0.03em; margin: 0.5rem 0 1rem; }
        .hero-lime { color: #d4ff32; display: block; }
        .studio-card { background: #12211c; border: 1px solid rgba(255,255,255,0.09); border-radius: 12px; padding: 1.6rem; }
        .online-pill { background: rgba(45,168,112,0.2); border: 1px solid rgba(86,227,159,0.45); border-radius: 8px; padding: 0.65rem 0.9rem; font-size: 0.82rem; color: #d8f8e5; margin-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    bundle = load_model_bundle()
    art = load_summary_artifacts()

    with st.sidebar:
        st.markdown("### `S` &nbsp; SKY / MOVE\n<span style='font-size:0.72rem;color:#8fa39b;letter-spacing:0.12em;'>AVIATION INTELLIGENCE</span>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("WORKSPACE", ["Prediction studio", "How it works", "Model intelligence"], label_visibility="visible")
        st.markdown("<div class='online-pill'>● &nbsp; Prediction engine online</div>", unsafe_allow_html=True)
        st.caption("BUILT FOR SKIES IN MOTION.\nCBCA 293 • BENNETT UNIVERSITY")

    if page == "Prediction studio":
        h1, h2 = st.columns([1.15, 0.85], gap="large")
        with h1:
            st.markdown(
                """
                <div class="eyebrow">— YOUR FLIGHT. BETTER TIMED.</div>
                <div class="hero-h1">Less waiting.<span class="hero-lime">More moving.</span></div>
                <p style="color:#8fa39b;font-size:1rem;max-width:460px;">
                    Turn route schedule and surface weather into smarter departure expectations. Powered by 7.08 million U.S. flight logs.
                </p>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f"""
                <div class="studio-card" style="text-align:left;">
                    <div class="eyebrow">U.S. DOMESTIC NETWORK</div>
                    <div style="font-size:2.1rem;font-weight:800;color:#d4ff32;margin:0.4rem 0;">{art['ingestion']['total_flights_ingested']:,}</div>
                    <div style="color:#8fa39b;font-size:0.88rem;">Real 2024 BTS Flights + {art['ingestion']['weather_rows_ingested']:,} NOAA Hourly METAR Logs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("<div class='eyebrow'>THE PREDICTION STUDIO ↗</div><h2 style='font-weight:800;'>Make your next move count.</h2>", unsafe_allow_html=True)

        ap_opts = {f"{a['IATA']} — {a['CITY']}": a["IATA"] for a in AIRPORTS_REGISTRY}
        al_opts = {f"{c['CARRIER_CODE']} — {c['AIRLINE_NAME']}": c["CARRIER_CODE"] for c in AIRLINES_REGISTRY}

        col_in, col_out = st.columns([1.05, 0.95], gap="large")
        with col_in:
            f1, f2 = st.columns(2)
            orig_lbl = f1.selectbox("Origin Airport", list(ap_opts.keys()), index=0)
            dest_lbl = f2.selectbox("Destination Airport", list(ap_opts.keys()), index=6)
            origin = ap_opts[orig_lbl]
            dest = ap_opts[dest_lbl] if ap_opts[dest_lbl] != origin else "ORD"

            f3, f4 = st.columns(2)
            car_lbl = f3.selectbox("Airline", list(al_opts.keys()), index=1)
            dep_t = f4.time_input("Scheduled Departure", value=dt_time(7, 15))
            wx_choice = st.segmented_control("Origin Surface Weather", list(WX_SIMPLE.keys()), default="Clear Sky") or "Clear Sky"
            wx = WX_SIMPLE[wx_choice]

            with st.expander("+ Fine-tune sensors & inbound aircraft delay"):
                wind_kt = st.slider("Wind Speed (kt)", 0, 50, int(wx["wind"]))
                vis_mi = st.slider("Visibility (mi)", 0.5, 10.0, float(wx["vis"]), 0.5)
                prev_d = st.slider("Inbound Aircraft Delay (min)", -10, 150, 15 if dep_t.hour >= 17 else 0)

        X_q, ctx = build_single_feature_vector(
            bundle, origin, dest, al_opts[car_lbl], date(2024, 7, 18), dep_t,
            wx["temp"], wx["dew"], wind_kt, vis_mi, wx["precip"], wx["press"],
            prev_aircraft_delay=float(prev_d),
        )
        prob = float(bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(X_q)[0, 1])
        est_m = float(bundle["delay_regressor"].predict(X_q)[0])
        delayed = prob >= float(bundle.get("default_threshold", 0.50))

        with col_out:
            status_color = "#ff6b57" if delayed else "#d4ff32"
            verdict_txt = "DELAY EXPECTED (≥15M)" if delayed else "ON SCHEDULE"
            offset_txt = f"+{max(16, round(est_m))} min" if delayed else "On time"
            st.markdown(
                f"""
                <div class="studio-card">
                    <div class="eyebrow" style="color:{status_color};">{verdict_txt} &nbsp;•&nbsp; {origin} → {dest} ({ctx['distance_mi']:.0f} MI)</div>
                    <div style="display:flex;gap:2.5rem;margin:1.2rem 0;">
                        <div>
                            <div style="font-size:0.75rem;color:#8fa39b;text-transform:uppercase;">Delay Risk</div>
                            <div style="font-size:3.1rem;font-weight:800;color:{status_color};">{prob*100:.0f}%</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem;color:#8fa39b;text-transform:uppercase;">Expected Status</div>
                            <div style="font-size:3.1rem;font-weight:800;color:{status_color};">{offset_txt}</div>
                        </div>
                    </div>
                    <div style="color:#8fa39b;font-size:0.9rem;">
                        Weather: <b>{wx_choice}</b> ({wind_kt} kt wind, {vis_mi} mi vis) &nbsp;|&nbsp; Inbound lag: <b>{prev_d:+d} min</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "How it works":
        st.markdown("<div class='eyebrow'>— ARCHITECTURE</div><h2 style='font-weight:800;'>Seven million flights, distilled into clarity.</h2>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("U.S. DOT BTS Logs", f"{art['ingestion']['total_flights_ingested']:,}")
        k2.metric("NOAA ASOS Logs", f"{art['ingestion']['weather_rows_ingested']:,}")
        k3.metric("Cleaned & Joined", f"{art['clean_join']['joined_clean_rows']:,}")
        k4.metric("Untouched Test Set", f"{art['eval']['test_rows_evaluated']:,}")

    else:
        st.markdown("<div class='eyebrow'>— VALIDATION</div><h2 style='font-weight:800;'>Benchmarked on 1.13M untouched flights.</h2>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(art["eval"]["benchmarks"]), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
