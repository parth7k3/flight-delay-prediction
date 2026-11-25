"""
Stage 7 — Interactive Streamlit Live Demo Dashboard (app.py)
Project: Predicting Flight Delays: A Big Data and Machine Learning Approach
Course: Big Data Analytics and Business Intelligence (CBCA 293) — Bennett University
Module Owner: Abhijeet Rana (S24CSEU1419) | Full Team: Parth, Vipul Gupta, Krishna Sharma, Abhijeet Rana
"""

from __future__ import annotations
import json
import math
from datetime import date, time as dt_time
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
    MODELS_DIR,
    OUTPUTS_DIR,
    PLANS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    SQL_DIR,
)
from feature_engineering import encode_cyclical_time
from generate_datasets import haversine_miles

st.set_page_config(
    page_title="Flight Delay Intelligence | Big Data & ML Demo",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise aviation styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0ea5e9 100%);
        color: #f8fafc;
        padding: 1.35rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .hero-title { font-size: 1.75rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
    .hero-sub { font-size: 0.95rem; opacity: 0.9; margin-top: 0.3rem; }
    .verdict-ontime {
        background: linear-gradient(135deg, #064e3b, #059669);
        color: white; padding: 1.2rem 1.4rem; border-radius: 12px;
        border-left: 6px solid #34d399;
    }
    .verdict-delayed {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        color: white; padding: 1.2rem 1.4rem; border-radius: 12px;
        border-left: 6px solid #fca5a5;
    }
    .ops-box {
        background: #f8fafc; border: 1px solid #cbd5e1;
        border-radius: 10px; padding: 0.9rem 1.1rem; color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle() -> dict:
    model_path = MODELS_DIR / "flight_delay_models.joblib"
    if not model_path.exists():
        from run_pipeline import main as run_full_pipeline
        run_full_pipeline()
    return joblib.load(model_path)


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

    # Load compact sample of featured flights for empirical weather/hourly analytics
    df_feat = pd.read_parquet(
        PROCESSED_DIR / "featured_flights.parquet",
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


def fetch_live_airport_weather(lat: float, lon: float) -> dict | None:
    """Fetches real-time surface aviation weather from Open-Meteo for the selected U.S. hub."""
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
) -> tuple[pd.DataFrame, dict]:
    ap_map = {a["IATA"]: a for a in AIRPORTS_REGISTRY}
    orig_ap = ap_map[origin_iata]
    dest_ap = ap_map[dest_iata]
    meta = bundle["feature_metadata"]

    dist = max(180.0, haversine_miles(orig_ap["LAT"], orig_ap["LON"], dest_ap["LAT"], dest_ap["LON"]))
    dep_hr_float = dep_time_val.hour + dep_time_val.minute / 60.0
    dow = fl_date.weekday()
    cyc = encode_cyclical_time(dep_hr_float, dow)

    base_vol = meta["hub_hourly_vol_lookup"].get(f"{origin_iata}_{dep_time_val.hour}", orig_ap["CAPACITY_HR"] * 0.35)
    roll_3h_vol = round(float(base_vol * traffic_mult), 1)
    expected_cap = max(orig_ap["CAPACITY_HR"] * 0.42, 1.0)
    cong_ratio = round(roll_3h_vol / expected_cap, 3)

    prior = meta["global_delay_prior"]
    row = {
        "DISTANCE": dist,
        "dep_hour_sin": float(cyc["dep_hour_sin"]),
        "dep_hour_cos": float(cyc["dep_hour_cos"]),
        "dow_sin": float(cyc["dow_sin"]),
        "dow_cos": float(cyc["dow_cos"]),
        "is_weekend": int(dow in (5, 6)),
        "TEMP_F": float(temp_f),
        "DEW_POINT_F": float(dew_f),
        "temp_dew_spread": round(float(temp_f - dew_f), 2),
        "WIND_SPEED_KT": float(wind_kt),
        "VISIBILITY_MI": float(vis_mi),
        "PRECIP_IN": float(precip_in),
        "PRESSURE_MB": float(pressure_mb),
        "origin_rolling_3h_vol": roll_3h_vol,
        "origin_congestion_ratio": cong_ratio,
        "carrier_hist_delay_rate": float(meta["carrier_hist_delay_rate"].get(carrier_code, prior)),
        "origin_hist_delay_rate": float(meta["origin_hist_delay_rate"].get(origin_iata, prior)),
        "dest_hist_delay_rate": float(meta["dest_hist_delay_rate"].get(dest_iata, prior)),
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


WEATHER_PRESETS = {
    "Clear VFR Day (Nominal)": {"temp": 64.0, "dew": 45.0, "wind": 8.5, "vis": 10.0, "precip": 0.00, "press": 1018.0},
    "High Crosswinds (>25 kt Runway Alert)": {"temp": 52.0, "dew": 38.0, "wind": 31.0, "vis": 8.5, "precip": 0.02, "press": 1006.0},
    "Summer Convective Thunderstorm": {"temp": 79.0, "dew": 74.5, "wind": 29.5, "vis": 1.6, "precip": 0.46, "press": 1002.0},
    "Winter Blizzard / Freezing De-icing": {"temp": 25.0, "dew": 23.0, "wind": 27.0, "vis": 0.8, "precip": 0.31, "press": 1004.0},
    "Dense Coastal Morning Fog (IFR)": {"temp": 53.0, "dew": 51.8, "wind": 5.5, "vis": 0.5, "precip": 0.01, "press": 1016.0},
}


def main() -> None:
    bundle = load_model_bundle()
    art = load_pipeline_artifacts()

    xgb_bench = next(
        b for b in art["eval"]["benchmarks"] if "XGBoost" in b["Model Architecture"]
    )

    # Hero Header
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">✈️ Predicting Flight Delays — Big Data & Machine Learning Console</div>
            <div class="hero-sub">
                U.S. BTS Flight Logs + NOAA Surface Weather + PySpark ETL + Training-Fold SMOTE + Tuned XGBoost
                &nbsp;|&nbsp; <b>CBCA 293 — Bennett University</b> (Parth, Vipul Gupta, Krishna Sharma, Abhijeet Rana)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Pipeline KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Raw BTS Flight Records", f"{art['ingestion']['total_flights_ingested']:,}", f"{art['ingestion']['partitions_loaded']} Monthly Partitions")
    k2.metric("Cleaned & Weather-Joined", f"{art['clean_join']['joined_clean_rows']:,}", f"{art['clean_join']['delay_rate_pct']}% Delayed (>=15m)")
    k3.metric("Primary XGBoost ROC-AUC", f"{xgb_bench['ROC-AUC']:.3f}", "Untouched 20% Test")
    k4.metric("Delay Detection Recall", f"{xgb_bench['Recall'] * 100:.1f}%", f"F1: {xgb_bench['F1-Score']:.3f}")
    k5.metric("Delay Regressor MAE", f"{art['eval']['secondary_regression_baseline']['mae_minutes']:.1f} min", "Secondary Baseline")

    # Sidebar Operational Threshold & Quick Info
    with st.sidebar:
        st.header("🎛️ Dispatch Settings")
        decision_threshold = st.slider(
            "Classification Decision Threshold",
            min_value=0.20,
            max_value=0.80,
            value=float(bundle.get("default_threshold", 0.45)),
            step=0.02,
            help="Lower threshold boosts Recall (catching more potential delays to protect tight passenger connections); higher threshold boosts Precision.",
        )
        opt_f2 = art["eval"]["optimal_f2_ops_threshold"]["threshold"]
        st.caption(
            f"FAA Delay Definition: **Departure Delay ≥ 15 min (Class 1)**.\n\n"
            f"Recommended Recall-Prioritized Threshold (F2 Max): **`{opt_f2:.2f}`**."
        )
        st.divider()
        st.subheader("👥 Team Deliverables")
        st.markdown(
            "- **Parth** (`S24CSEU2470`): Ingestion, PostgreSQL/SQLite Storage & PySpark Cleaning\n"
            "- **Vipul Gupta** (`S24CSEU2526`): Cyclical $\\sin/\\cos$ Encodings & Hub Congestion\n"
            "- **Krishna Sharma** (`S24CSEU2756`): Training-Fold SMOTE & Ensemble Training\n"
            "- **Abhijeet Rana** (`S24CSEU1419`): ROC/PR Evaluation, Threshold Tuning & Streamlit UI"
        )

    tabs = st.tabs([
        "🛫 Live Flight Delay Predictor",
        "📋 Live Hub Departure Board",
        "📊 Model Benchmarks & Threshold Lab",
        "⚖️ SMOTE & Weather Threshold Analytics",
        "🗄️ Big Data Pipeline & PySpark Plan",
    ])

    # =========================================================================
    # TAB 1: LIVE FLIGHT DELAY PREDICTOR
    # =========================================================================
    with tabs[0]:
        ap_options = {f"{a['IATA']} — {a['CITY']} ({a['NAME']})": a["IATA"] for a in AIRPORTS_REGISTRY}
        ap_labels = list(ap_options.keys())
        al_options = {f"{c['CARRIER_CODE']} — {c['AIRLINE_NAME']}": c["CARRIER_CODE"] for c in AIRLINES_REGISTRY}

        c_left, c_right = st.columns([1.08, 1.32], gap="large")

        with c_left:
            st.subheader("1. Flight Schedule & Route")
            r1c1, r1c2 = st.columns(2)
            orig_label = r1c1.selectbox("Origin Hub Airport", ap_labels, index=1)  # ORD default
            dest_label = r1c2.selectbox("Destination Airport", ap_labels, index=5)  # JFK default
            origin_iata = ap_options[orig_label]
            dest_iata = ap_options[dest_label]

            if origin_iata == dest_iata:
                st.warning("Origin and Destination airports must be distinct. Adjusting destination automatically.")
                dest_iata = "ATL" if origin_iata != "ATL" else "LAX"

            r2c1, r2c2 = st.columns(2)
            carrier_label = r2c1.selectbox("Operating Carrier", list(al_options.keys()), index=2)
            carrier_code = al_options[carrier_label]
            fl_date = r2c2.date_input("Scheduled Flight Date", value=date(2025, 3, 21))

            r3c1, r3c2 = st.columns(2)
            dep_hour = r3c1.slider("Scheduled Departure Hour (24h)", 0, 23, 18)
            dep_min = r3c2.selectbox("Departure Minute Slot", [0, 15, 30, 45], index=2)
            dep_time_val = dt_time(dep_hour, dep_min)

            traffic_label = st.select_slider(
                "Origin Hub Departure Bank Load",
                options=["Light Off-Peak (0.75x)", "Standard Schedule (1.0x)", "Busy Bank (1.25x)", "Peak Holiday Bank (1.50x)"],
                value="Standard Schedule (1.0x)",
            )
            traffic_mult = {"Light Off-Peak (0.75x)": 0.75, "Standard Schedule (1.0x)": 1.0, "Busy Bank (1.25x)": 1.25, "Peak Holiday Bank (1.50x)": 1.50}[traffic_label]

            st.subheader("2. Origin Surface Weather (NOAA / METAR)")
            wx_col1, wx_col2 = st.columns([1.6, 1.0])
            preset_name = wx_col1.selectbox("Weather Scenario Preset", list(WEATHER_PRESETS.keys()), index=0)

            # Session state management for live weather override
            if "live_wx" not in st.session_state or st.session_state.get("last_preset") != preset_name:
                st.session_state["live_wx"] = WEATHER_PRESETS[preset_name].copy()
                st.session_state["last_preset"] = preset_name
                st.session_state["wx_source"] = f"Preset: {preset_name}"

            orig_coords = next(a for a in AIRPORTS_REGISTRY if a["IATA"] == origin_iata)
            if wx_col2.button(f"🌐 Fetch Live {origin_iata} Weather", use_container_width=True):
                live_res = fetch_live_airport_weather(orig_coords["LAT"], orig_coords["LON"])
                if live_res:
                    st.session_state["live_wx"] = {
                        "temp": live_res["temp_f"],
                        "dew": live_res["dew_f"],
                        "wind": live_res["wind_kt"],
                        "vis": live_res["vis_mi"],
                        "precip": live_res["precip_in"],
                        "press": live_res["pressure_mb"],
                    }
                    st.session_state["wx_source"] = f"Live Open-Meteo Aviation Feed ({origin_iata})"
                    st.toast(f"Loaded live surface weather for {origin_iata}!", icon="🌤️")
                else:
                    st.info("Live weather API unreachable in current network mode — using interactive controls below.")

            st.caption(f"Active Weather Source: **{st.session_state.get('wx_source', preset_name)}**")
            wx_vals = st.session_state["live_wx"]

            w1, w2, w3 = st.columns(3)
            temp_f = w1.number_input("Temp (°F)", min_value=-20.0, max_value=120.0, value=float(wx_vals["temp"]), step=1.0)
            dew_f = w2.number_input("Dew Point (°F)", min_value=-30.0, max_value=95.0, value=float(min(wx_vals["dew"], temp_f)), step=1.0)
            wind_kt = w3.slider("Wind Speed (kt)", 0.0, 48.0, float(wx_vals["wind"]), 0.5)

            w4, w5, w6 = st.columns(3)
            vis_mi = w4.slider("Visibility (mi)", 0.25, 10.0, float(wx_vals["vis"]), 0.25)
            precip_in = w5.slider("Precip (in/hr)", 0.0, 1.5, float(wx_vals["precip"]), 0.02)
            pressure_mb = w6.number_input("Pressure (mb)", min_value=975.0, max_value=1045.0, value=float(wx_vals["press"]), step=1.0)

        # Build features & run live inference across all 4 classifiers + secondary regressor
        X_query, ctx = build_single_feature_vector(
            bundle, origin_iata, dest_iata, carrier_code, fl_date, dep_time_val,
            temp_f, dew_f, wind_kt, vis_mi, precip_in, pressure_mb, traffic_mult
        )

        model_probs = {
            m_name: float(clf.predict_proba(X_query)[0, 1])
            for m_name, clf in bundle["classifiers"].items()
        }
        primary_prob = model_probs["XGBoost Classifier (Tuned)"]
        est_delay_min = float(bundle["delay_regressor"].predict(X_query)[0])
        is_delayed_pred = primary_prob >= decision_threshold

        with c_right:
            st.subheader("3. Real-Time ML Prediction & Dispatch Verdict")
            if is_delayed_pred:
                st.markdown(
                    f"""
                    <div class="verdict-delayed">
                        <div style="font-size:0.85rem; text-transform:uppercase; font-weight:700;">FAA Class 1 Alert — Departure Delay ≥ 15 Minutes Likely</div>
                        <div style="font-size:1.85rem; font-weight:800; margin:0.2rem 0;">
                            ⚠️ DELAY RISK: {primary_prob * 100:.1f}% &nbsp;|&nbsp; Est. Delay: +{max(15.0, est_delay_min):.0f} min
                        </div>
                        <div style="font-size:0.92rem;">
                            Route <b>{origin_iata} → {dest_iata}</b> ({ctx['distance_mi']:.0f} mi) • Carrier <b>{carrier_code}</b> • Slot <b>{dep_time_val.strftime('%H:%M')}</b> exceeds threshold ({decision_threshold:.2f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                disp_min = est_delay_min if est_delay_min > 0 else 0.0
                st.markdown(
                    f"""
                    <div class="verdict-ontime">
                        <div style="font-size:0.85rem; text-transform:uppercase; font-weight:700;">FAA Class 0 — Scheduled On-Time Operations (&lt; 15 Min)</div>
                        <div style="font-size:1.85rem; font-weight:800; margin:0.2rem 0;">
                            ✅ ON-TIME EXPECTED &nbsp;|&nbsp; Delay Risk: {primary_prob * 100:.1f}% (Est. {disp_min:.0f} min)
                        </div>
                        <div style="font-size:0.92rem;">
                            Route <b>{origin_iata} → {dest_iata}</b> ({ctx['distance_mi']:.0f} mi) • Carrier <b>{carrier_code}</b> • Slot <b>{dep_time_val.strftime('%H:%M')}</b> below threshold ({decision_threshold:.2f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            g_col, m_col = st.columns([1.0, 1.15])
            with g_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(primary_prob * 100.0, 1),
                    number={"suffix": "%", "font": {"size": 34}},
                    title={"text": "Tuned XGBoost Delay Probability", "font": {"size": 14}},
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
                fig_gauge.update_layout(height=245, margin=dict(l=20, r=20, t=42, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

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
                    range_x=[0, 105],
                    color="Probability (%)",
                    color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                )
                fig_comp.add_vline(x=decision_threshold * 100, line_dash="dash", line_color="#0f172a")
                fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_comp.update_layout(
                    title="Multi-Model Consensus Comparison",
                    height=245,
                    margin=dict(l=10, r=25, t=38, b=10),
                    coloraxis_showscale=False,
                    yaxis_title="",
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            # Operational Risk Factors & Dispatch Playbook
            drivers = []
            if wind_kt > 25.0:
                drivers.append(f"🌬️ **High Crosswind Threshold Triggered ({wind_kt:.1f} kt > 25 kt)**: FAA runway configuration restrictions likely.")
            if vis_mi < 3.0:
                drivers.append(f"🌫️ **IFR Low Visibility ({vis_mi:.2f} mi < 3.0 mi)**: Increased ATC spacing between arrivals/departures.")
            if temp_f <= 33.5 and precip_in >= 0.04:
                drivers.append(f"❄️ **Active De-icing Conditions ({temp_f:.0f}°F + {precip_in:.2f} in/hr)**: Mandatory glycol pad queue before takeoff.")
            if dep_hour >= 16:
                drivers.append(f"🌇 **Evening Ripple Window ({dep_time_val.strftime('%H:%M')})**: Elevated inbound aircraft turnaround cascade risk.")
            if ctx["congestion_ratio"] > 1.10:
                drivers.append(f"🛫 **High Hub Runway Load ({ctx['congestion_ratio']:.2f}x baseline, {ctx['roll_3h_vol']:.0f} rolling 3h deps)**: Taxiway queue buildup.")
            if not drivers:
                drivers.append("🟢 **Nominal Surface & Schedule Profile**: VFR weather conditions and manageable runway departure load.")

            st.markdown("**Active Operational Drivers & AOCC Playbook:**")
            for d_msg in drivers:
                st.markdown(f"- {d_msg}")

            if is_delayed_pred:
                st.info(
                    "**Recommended Dispatch Action (T-2 Hours):** Flag inbound tail assignment for turnaround buffer, "
                    "pre-notify passengers with <45 min downstream hub connections, and hold adjacent gate assignment."
                )

        st.divider()
        map_col, clock_col = st.columns([1.45, 1.0], gap="large")
        with map_col:
            ap_df_all = pd.DataFrame(AIRPORTS_REGISTRY)
            route_color = "#dc2626" if is_delayed_pred else "#059669"
            fig_map = go.Figure()
            fig_map.add_trace(go.Scattergeo(
                lon=ap_df_all["LON"],
                lat=ap_df_all["LAT"],
                text=ap_df_all["IATA"] + " — " + ap_df_all["CITY"],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=8, color="#334155", opacity=0.7),
                name="U.S. Hubs",
            ))
            fig_map.add_trace(go.Scattergeo(
                lon=[ctx["orig_ap"]["LON"], ctx["dest_ap"]["LON"]],
                lat=[ctx["orig_ap"]["LAT"], ctx["dest_ap"]["LAT"]],
                mode="lines+markers",
                line=dict(width=4, color=route_color),
                marker=dict(size=13, color=route_color),
                name=f"{origin_iata} → {dest_iata} ({ctx['distance_mi']:.0f} mi)",
            ))
            fig_map.update_geos(
                scope="usa",
                projection_type="albers usa",
                showland=True,
                landcolor="#f1f5f9",
                subunitcolor="#cbd5e1",
            )
            fig_map.update_layout(
                title=f"Continental U.S. Flight Corridor: {origin_iata} ({ctx['orig_ap']['CITY']}) → {dest_iata} ({ctx['dest_ap']['CITY']})",
                height=340,
                margin=dict(l=0, r=0, t=38, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with clock_col:
            theta_grid = np.linspace(0, 2 * np.pi, 200)
            sin_val = float(X_query["dep_hour_sin"].iloc[0])
            cos_val = float(X_query["dep_hour_cos"].iloc[0])
            fig_clock = go.Figure()
            fig_clock.add_trace(go.Scatter(
                x=np.cos(theta_grid), y=np.sin(theta_grid),
                mode="lines", line=dict(color="#94a3b8", dash="dot"), name="24h Unit Circle"
            ))
            for hr_mark in [0, 6, 12, 18]:
                ang = 2 * math.pi * (hr_mark / 24.0)
                fig_clock.add_trace(go.Scatter(
                    x=[math.cos(ang)], y=[math.sin(ang)],
                    mode="markers+text",
                    text=[f"{hr_mark:02d}:00"],
                    textposition="top center",
                    marker=dict(size=7, color="#475569"),
                    showlegend=False,
                ))
            fig_clock.add_trace(go.Scatter(
                x=[0, cos_val], y=[0, sin_val],
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=13, color="#dc2626" if is_delayed_pred else "#2563eb"),
                name=f"Slot {dep_time_val.strftime('%H:%M')} (cos={cos_val:.2f}, sin={sin_val:.2f})",
            ))
            fig_clock.update_layout(
                title="24-Hour Cyclical Time Encoding (sin/cos)",
                xaxis=dict(range=[-1.35, 1.35], title="cos(2π·h/24)", zeroline=True),
                yaxis=dict(range=[-1.35, 1.35], title="sin(2π·h/24)", zeroline=True, scaleanchor="x", scaleratio=1),
                height=340,
                margin=dict(l=20, r=20, t=38, b=20),
                legend=dict(orientation="h", y=-0.18),
            )
            st.plotly_chart(fig_clock, use_container_width=True)

    # =========================================================================
    # TAB 2: LIVE HUB DEPARTURE BOARD (BATCH SCANNER)
    # =========================================================================
    with tabs[1]:
        st.subheader("📋 Airline Operations Control Center (AOCC) — Live Hub Departure Wave Scanner")
        st.caption("Simulate an upcoming 20-flight departure bank at any U.S. hub under chosen weather conditions to prioritize gate & connection interventions.")

        b1, b2, b3, b4 = st.columns(4)
        hub_iata = b1.selectbox("Select Origin Hub", [a["IATA"] for a in AIRPORTS_REGISTRY], index=1, key="board_hub")
        board_wx_name = b2.selectbox("Hub Surface Weather", list(WEATHER_PRESETS.keys()), index=1, key="board_wx")
        wave_start_hr = b3.slider("Departure Wave Start Hour", 5, 21, 17, key="board_hr")
        board_traffic = b4.selectbox("Runway Load Factor", [1.0, 1.25, 1.45], index=1, format_func=lambda x: f"{x:.2f}x Capacity")

        bwx = WEATHER_PRESETS[board_wx_name]
        rng_board = np.random.default_rng(int( ord(hub_iata[0]) * 100 + wave_start_hr ))
        other_airports = [a["IATA"] for a in AIRPORTS_REGISTRY if a["IATA"] != hub_iata]
        carriers_list = [c["CARRIER_CODE"] for c in AIRLINES_REGISTRY]

        board_rows = []
        feature_frames = []
        for i in range(20):
            d_iata = other_airports[i % len(other_airports)]
            c_code = carriers_list[int(rng_board.integers(0, len(carriers_list)))]
            slot_hr = min(23, wave_start_hr + (i // 7))
            slot_min = int((i * 7) % 60)
            fl_no = f"{c_code}{1100 + i * 37}"
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
                "Sched Dep": f"{slot_hr:02d}:{slot_min:02d}",
                "Distance (mi)": int(ctx_b["distance_mi"]),
                "Wind (kt)": bwx["wind"],
                "Vis (mi)": bwx["vis"],
            })

        X_batch = pd.concat(feature_frames, ignore_index=True)
        batch_probs = bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(X_batch)[:, 1]
        batch_mins = bundle["delay_regressor"].predict(X_batch)

        board_df = pd.DataFrame(board_rows)
        board_df["Delay Prob (%)"] = np.round(batch_probs * 100.0, 1)
        board_df["Est. Delay (min)"] = np.maximum(0, np.round(batch_mins, 0)).astype(int)
        board_df["Dispatch Status"] = np.where(
            batch_probs >= decision_threshold, "🔴 DELAY RISK (>=15m)", "🟢 ON-TIME"
        )
        board_df = board_df.sort_values("Delay Prob (%)", ascending=False).reset_index(drop=True)

        flagged_cnt = int((batch_probs >= decision_threshold).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Flights Scanned in Wave", f"{len(board_df)}", f"Origin Hub: {hub_iata}")
        m2.metric("High-Risk Flights Flagged", f"{flagged_cnt} / {len(board_df)}", f"Threshold: {decision_threshold:.2f}")
        m3.metric("Mean Wave Delay Probability", f"{batch_probs.mean() * 100:.1f}%", board_wx_name)

        st.dataframe(board_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Export Scored Departure Wave CSV",
            data=board_df.to_csv(index=False).encode("utf-8"),
            file_name=f"aocc_departure_wave_{hub_iata}.csv",
            mime="text/csv",
        )

    # =========================================================================
    # TAB 3: MODEL BENCHMARKS & INTERACTIVE THRESHOLD LAB
    # =========================================================================
    with tabs[2]:
        st.subheader("📊 Untouched 20% Chronological Test Split — Model Architecture Benchmarks")
        st.markdown(
            "Candidate models were trained on the **SMOTE-balanced 80% chronological training split** and evaluated on "
            "the **untouched 20% future test split** (preserving realistic ~80/20 operational class ratios with zero temporal leakage)."
        )
        bench_df = pd.DataFrame(art["eval"]["benchmarks"])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            fig_roc = go.Figure()
            for m_name, cdata in art["eval"]["curves"].items():
                fig_roc.add_trace(go.Scatter(
                    x=cdata["fpr"], y=cdata["tpr"], mode="lines",
                    name=f"{m_name} (AUC={cdata['roc_auc']:.3f})",
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random (0.500)"))
            fig_roc.update_layout(title="ROC-AUC Curves (Untouched Test Split)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=390)
            st.plotly_chart(fig_roc, use_container_width=True)

        with rc2:
            fig_pr = go.Figure()
            for m_name, cdata in art["eval"]["curves"].items():
                fig_pr.add_trace(go.Scatter(
                    x=cdata["pr_recall"], y=cdata["pr_precision"], mode="lines",
                    name=f"{m_name} (AP={cdata['pr_auc']:.3f})",
                ))
            fig_pr.update_layout(title="Precision-Recall Curves (80/20 Skewed Test Split)", xaxis_title="Recall", yaxis_title="Precision", height=390)
            st.plotly_chart(fig_pr, use_container_width=True)

        st.divider()
        st.subheader("🎚️ Interactive Operational Threshold Simulator (Tuned XGBoost)")
        th_col1, th_col2 = st.columns([1.15, 1.0], gap="large")
        with th_col1:
            th_df = pd.DataFrame(art["eval"]["threshold_tuning"])
            fig_th = go.Figure()
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["recall"], name="Recall (Catch Rate)", line=dict(color="#dc2626", width=3)))
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["precision"], name="Precision", line=dict(color="#2563eb", width=3)))
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f1"], name="F1-Score", line=dict(color="#059669", width=2.5)))
            fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f2_ops"], name="F2-Score (Ops Priority)", line=dict(color="#7c3aed", dash="dot", width=2.5)))
            fig_th.add_vline(x=decision_threshold, line_dash="dash", line_color="#0f172a", annotation_text=f"Active={decision_threshold:.2f}")
            fig_th.update_layout(title="Precision / Recall / F1 / F2 vs. Decision Threshold", xaxis_title="Probability Threshold", yaxis_title="Score", height=360)
            st.plotly_chart(fig_th, use_container_width=True)

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
                title=f"Live Confusion Matrix @ Threshold = {decision_threshold:.2f} (n={len(y_t):,} sample)",
            )
            fig_cm.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("🏆 Tuned XGBoost Feature Importance & Target Correlation Rankings")
        feat_rank_df = pd.DataFrame(art["eval"]["feature_rankings"])
        fig_fi = px.bar(
            feat_rank_df.head(14).iloc[::-1],
            x="xgb_importance",
            y="display_name",
            orientation="h",
            color="target_correlation",
            color_continuous_scale="RdBu_r",
            labels={"xgb_importance": "XGBoost Relative Gain Importance", "display_name": "Engineered Feature", "target_correlation": "Corr w/ Delay"},
            title="Top 14 Predictive Features (Weather Thresholds + Rolling Hub Volume + Cyclical Time)",
        )
        fig_fi.update_layout(height=430)
        st.plotly_chart(fig_fi, use_container_width=True)

    # =========================================================================
    # TAB 4: SMOTE & NON-LINEAR WEATHER ANALYTICS
    # =========================================================================
    with tabs[3]:
        st.subheader("⚖️ Addressing the 80/20 Class Skew — Strict Training-Fold SMOTE")
        sm = art["smote"]
        sm_df = pd.DataFrame([
            {"Partition": "1. Train (Before SMOTE)", "Class": "On-Time (<15m)", "Flights": sm["train_before_smote"]["on_time_class_0"]},
            {"Partition": "1. Train (Before SMOTE)", "Class": "Delayed (>=15m)", "Flights": sm["train_before_smote"]["delayed_class_1"]},
            {"Partition": "2. Train (After SMOTE)", "Class": "On-Time (<15m)", "Flights": sm["train_after_smote"]["on_time_class_0"]},
            {"Partition": "2. Train (After SMOTE)", "Class": "Delayed (>=15m)", "Flights": sm["train_after_smote"]["delayed_class_1"]},
            {"Partition": "3. Chronological Test (Untouched)", "Class": "On-Time (<15m)", "Flights": sm["test_untouched"]["on_time_class_0"]},
            {"Partition": "3. Chronological Test (Untouched)", "Class": "Delayed (>=15m)", "Flights": sm["test_untouched"]["delayed_class_1"]},
        ])
        fig_smote = px.bar(
            sm_df, x="Partition", y="Flights", color="Class", barmode="group", text_auto=",d",
            color_discrete_map={"On-Time (<15m)": "#0ea5e9", "Delayed (>=15m)": "#ef4444"},
            title=f"Class Distribution Across Pipeline Splits (+{sm['train_after_smote']['synthetic_samples_added']:,} Synthetic Delayed Vectors in Train Only)",
        )
        fig_smote.update_layout(height=370)
        st.plotly_chart(fig_smote, use_container_width=True)

        st.divider()
        st.subheader("🌩️ Empirical Non-Linear Weather & Ripple Effects (Why Tree Ensembles Beat Linear Models)")
        fdf = art["featured_df"]
        a1, a2 = st.columns(2)
        with a1:
            wind_bins = pd.cut(fdf["WIND_SPEED_KT"], bins=[0, 8, 15, 22, 28, 50], labels=["0–8 kt", "8–15 kt", "15–22 kt", "22–28 kt", ">28 kt (Gale)"])
            wind_grp = fdf.groupby(wind_bins, observed=False)["DEP_DEL15"].mean().reset_index()
            wind_grp["Delay Rate (%)"] = np.round(wind_grp["DEP_DEL15"] * 100.0, 1)
            fig_w = px.bar(
                wind_grp, x="WIND_SPEED_KT", y="Delay Rate (%)", text="Delay Rate (%)",
                color="Delay Rate (%)", color_continuous_scale="YlOrRd",
                title="Sharp Non-Linear Wind Threshold Effect (Choi et al., 2016)",
                labels={"WIND_SPEED_KT": "Origin Surface Wind Speed Bucket"},
            )
            fig_w.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_w.update_layout(height=360, yaxis_range=[0, max(wind_grp["Delay Rate (%)"]) * 1.2], coloraxis_showscale=False)
            st.plotly_chart(fig_w, use_container_width=True)

        with a2:
            hr_bucket = fdf["dep_hour_float"].astype(int)
            hr_grp = fdf.groupby(hr_bucket)["DEP_DEL15"].mean().reset_index()
            hr_grp["Delay Rate (%)"] = np.round(hr_grp["DEP_DEL15"] * 100.0, 1)
            fig_h = px.area(
                hr_grp, x="dep_hour_float", y="Delay Rate (%)", markers=True,
                title="Downstream Ripple Effect Across 24-Hour Departure Cycle (Rebollo & Balakrishnan, 2014)",
                labels={"dep_hour_float": "Scheduled Departure Hour (0–23)"},
            )
            fig_h.update_traces(line_color="#2563eb")
            fig_h.update_layout(height=360)
            st.plotly_chart(fig_h, use_container_width=True)

    # =========================================================================
    # TAB 5: BIG DATA ARCHITECTURE, WAREHOUSE & PYSPARK PLAN
    # =========================================================================
    with tabs[4]:
        st.subheader("🗄️ End-to-End 7-Stage Big Data Architecture & Artifacts")
        c_ing, c_cln = st.columns(2)
        with c_ing:
            st.markdown("#### Stage 1: Database Ingestion Audit (`ingestion.py`)")
            st.json(art["ingestion"])
        with c_cln:
            st.markdown("#### Stage 2: Cleaning & Weather Join Telemetry (`spark_clean_join.py`)")
            st.json(art["clean_join"])

        with st.expander("📜 View PySpark SQL & Catalyst Physical Execution Plan (`outputs/plans/pyspark_execution_plan.txt`)", expanded=True):
            st.code(art["plan_text"], language="text")

        with st.expander("🐘 View Distributed Partitioned PostgreSQL DDL (`sql/schema_postgres.sql`)"):
            st.code(art["sql_ddl"], language="sql")

        st.subheader("🖼️ Exported Static Evaluation Artifacts (`outputs/plots/`)")
        p1, p2 = st.columns(2)
        p1.image(str(PLOTS_DIR / "confusion_matrices.png"), caption="Confusion Matrices Across Candidate Classifiers")
        p2.image(str(PLOTS_DIR / "roc_pr_curves.png"), caption="ROC-AUC & Precision-Recall Curves")
        p3, p4 = st.columns(2)
        p3.image(str(PLOTS_DIR / "threshold_tuning.png"), caption="Operational Decision Threshold Tuning")
        p4.image(str(PLOTS_DIR / "feature_importance.png"), caption="Tuned XGBoost Feature Importance Rankings")


if __name__ == "__main__":
    main()
