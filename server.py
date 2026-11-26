"""
Lightweight FastAPI/Starlette Live Web Server (server.py)
Serves the minimalist SKY / MOVE Flight Delay Intelligence Studio UI + JSON ML API
backed by the 7.08M-row U.S. DOT BTS + 518K-row NOAA ASOS Tuned XGBoost pipeline.
"""

from __future__ import annotations
import json
from datetime import date, datetime, time as dt_time
import joblib
import numpy as np
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from config import (
    AIRPORTS_REGISTRY,
    AIRLINES_REGISTRY,
    BASE_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from app import build_single_feature_vector, fetch_live_airport_weather

WEB_HTML_PATH = BASE_DIR / "web" / "index.html"

WEATHER_MODES = {
    "clear": {"temp": 66.0, "dew": 46.0, "wind": 7.0, "vis": 10.0, "precip": 0.00, "press": 1018.0, "label": "Clear VFR Sky"},
    "wind": {"temp": 52.0, "dew": 38.0, "wind": 31.0, "vis": 8.5, "precip": 0.02, "press": 1005.0, "label": "High Crosswinds (31 kt)"},
    "rain": {"temp": 76.0, "dew": 72.0, "wind": 24.0, "vis": 2.0, "precip": 0.35, "press": 1004.0, "label": "Thunderstorm / Heavy Rain"},
    "snow": {"temp": 25.0, "dew": 23.0, "wind": 27.0, "vis": 0.8, "precip": 0.30, "press": 1003.0, "label": "Snow / Freezing De-icing"},
    "fog": {"temp": 52.0, "dew": 51.0, "wind": 6.0, "vis": 0.5, "precip": 0.01, "press": 1016.0, "label": "Dense Marine Fog (0.5 mi)"},
}

_BUNDLE = None
_ARTIFACTS = None
_ROUTE_PROFILES = None


def _calibrate_prob(p_smote: np.ndarray | float, prior: float = 0.2134) -> np.ndarray | float:
    """Maps 50/50 SMOTE-balanced posterior back to calibrated empirical probability scale via Bayes' rule."""
    p = np.clip(p_smote, 0.001, 0.999)
    num = p * prior
    den = num + (1.0 - p) * (1.0 - prior)
    return num / den


def _get_state() -> tuple[dict, dict, dict]:
    global _BUNDLE, _ARTIFACTS, _ROUTE_PROFILES
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODELS_DIR / "flight_delay_models.joblib")
        for clf in _BUNDLE["classifiers"].values():
            if hasattr(clf, "n_jobs"):
                clf.n_jobs = 1
        if hasattr(_BUNDLE["delay_regressor"], "n_jobs"):
            _BUNDLE["delay_regressor"].n_jobs = 1

    if _ARTIFACTS is None:
        with open(OUTPUTS_DIR / "ingestion_audit.json", "r", encoding="utf-8") as f:
            ing = json.load(f)
        with open(OUTPUTS_DIR / "clean_join_summary.json", "r", encoding="utf-8") as f:
            cln = json.load(f)
        with open(OUTPUTS_DIR / "smote_summary.json", "r", encoding="utf-8") as f:
            smt = json.load(f)
        with open(OUTPUTS_DIR / "evaluation_report.json", "r", encoding="utf-8") as f:
            evl = json.load(f)
        _ARTIFACTS = {"ingestion": ing, "clean_join": cln, "smote": smt, "eval": evl}

    if _ROUTE_PROFILES is None:
        rp_path = OUTPUTS_DIR / "route_slot_profiles.json"
        if not rp_path.exists():
            from build_route_profiles import build_route_profiles
            _ROUTE_PROFILES = build_route_profiles()
        else:
            with open(rp_path, "r", encoding="utf-8") as f:
                _ROUTE_PROFILES = json.load(f)

    return _BUNDLE, _ARTIFACTS, _ROUTE_PROFILES


async def index_page(request: Request) -> HTMLResponse:
    return HTMLResponse(WEB_HTML_PATH.read_text(encoding="utf-8"))


async def api_meta(request: Request) -> JSONResponse:
    bundle, art, _ = _get_state()
    return JSONResponse({
        "airports": AIRPORTS_REGISTRY,
        "airlines": AIRLINES_REGISTRY,
        "default_threshold": 0.32,
        "pipeline": {
            "raw_flights": art["ingestion"]["total_flights_ingested"],
            "partitions": art["ingestion"]["partitions_loaded"],
            "raw_mb": art["ingestion"].get("raw_disk_footprint_mb", 652.6),
            "weather_rows": art["ingestion"]["weather_rows_ingested"],
            "clean_joined": art["clean_join"]["joined_clean_rows"],
            "cancelled_dropped": art["clean_join"]["cancelled_dropped"],
            "diverted_dropped": art["clean_join"]["diverted_dropped"],
            "delay_rate_pct": art["clean_join"]["delay_rate_pct"],
            "train_rows": art["smote"].get("full_train_partition", art["smote"]["train_before_smote"])["total"],
            "smote_rows": art["smote"]["train_after_smote"]["total"],
            "test_rows": art["eval"].get("test_rows_evaluated", art["smote"]["test_untouched"]["total"]),
            "mae_minutes": art["eval"]["secondary_regression_baseline"]["mae_minutes"],
        },
        "benchmarks": art["eval"]["benchmarks"],
        "curves": art["eval"]["curves"],
        "threshold_tuning": art["eval"]["threshold_tuning"],
        "top_features": art["eval"]["feature_rankings"][:10],
    })


async def api_live_weather(request: Request) -> JSONResponse:
    iata = request.query_params.get("iata", "ORD").upper()
    ap = next((a for a in AIRPORTS_REGISTRY if a["IATA"] == iata), AIRPORTS_REGISTRY[0])
    wx = fetch_live_airport_weather(ap["LAT"], ap["LON"])
    if wx is None:
        return JSONResponse({"ok": False, "iata": iata, "fallback": WEATHER_MODES["clear"]})
    return JSONResponse({"ok": True, "iata": iata, "weather": wx})


async def api_predict(request: Request) -> JSONResponse:
    bundle, _, route_profiles = _get_state()
    body = await request.json()

    origin = str(body.get("origin", "ORD")).upper()
    dest = str(body.get("dest", "JFK")).upper()
    if origin == dest:
        dest = "LAX" if origin != "LAX" else "ATL"
    carrier = str(body.get("carrier", "UA")).upper()

    date_str = str(body.get("fl_date", "2024-07-18"))
    try:
        fl_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        fl_date = date(2024, 7, 18)

    time_str = str(body.get("dep_time", "17:30"))
    try:
        hh, mm = [int(x) for x in time_str.split(":")[:2]]
        dep_time_val = dt_time(hh % 24, mm % 60)
    except Exception:
        dep_time_val = dt_time(17, 30)

    wx_mode = str(body.get("weather_mode", "clear"))
    preset = WEATHER_MODES.get(wx_mode, WEATHER_MODES["clear"])

    temp_f = float(body.get("temp_f", preset["temp"]))
    dew_f = min(temp_f, float(body.get("dew_f", preset["dew"])))
    wind_kt = float(body.get("wind_kt", preset["wind"]))
    vis_mi = float(body.get("vis_mi", preset["vis"]))
    precip_in = float(body.get("precip_in", preset["precip"]))
    pressure_mb = float(body.get("pressure_mb", preset["press"]))
    traffic_mult = float(body.get("traffic_mult", 1.0))

    prior = float(bundle["feature_metadata"].get("global_delay_prior", 0.2134))
    cal_cutoff = float(_calibrate_prob(float(bundle.get("default_threshold", 0.52)), prior))

    prev_delay_in = body.get("prev_delay", None)
    prev_delay = float(prev_delay_in) if prev_delay_in is not None else None

    X_q, ctx = build_single_feature_vector(
        bundle, origin, dest, carrier, fl_date, dep_time_val,
        temp_f, dew_f, wind_kt, vis_mi, precip_in, pressure_mb,
        traffic_mult=traffic_mult, prev_aircraft_delay=prev_delay,
    )

    raw_xgb_p = float(bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(X_q)[0, 1])
    cal_xgb_p = float(_calibrate_prob(raw_xgb_p, prior))

    # Lookup real 2024 BTS operating schedule for (ORIGIN, DEST, CARRIER) -> fallback (ORIGIN, DEST) -> fallback ORIGIN
    odc_key = f"{origin}_{dest}_{carrier}"
    od_key = f"{origin}_{dest}"
    if odc_key in route_profiles["odc"]:
        sched_slots = route_profiles["odc"][odc_key]
        schedule_scope = f"{carrier} {origin}→{dest} timetable"
    elif od_key in route_profiles["od"]:
        sched_slots = route_profiles["od"][od_key]
        schedule_scope = f"{origin}→{dest} nonstop timetable"
    else:
        sched_slots = route_profiles["origin"].get(origin, {"7": [15, 11.0], "12": [30, 18.0], "18": [0, 26.0]})
        schedule_scope = f"{origin} hub timetable"

    # Blend active slot's calibrated XGBoost risk with empirical route-hour baseline if available
    cur_slot_info = sched_slots.get(str(dep_time_val.hour))
    wx_stress_boost = (1.0 if (wind_kt >= 22 or vis_mi < 4.0 or precip_in >= 0.05 or (prev_delay or 0) >= 15) else 0.0)
    if cur_slot_info and wx_stress_boost == 0.0:
        primary_pct = round(0.55 * (cal_xgb_p * 100.0) + 0.45 * float(cur_slot_info[1]), 1)
    else:
        primary_pct = round(cal_xgb_p * 100.0, 1)

    raw_min = float(bundle["delay_regressor"].predict(X_q)[0])
    is_delayed = (primary_pct / 100.0) >= cal_cutoff
    est_delay_min = int(round(max(16.0, raw_min))) if is_delayed else int(round(max(0.0, raw_min)))

    # Compute 24-hour route-specific curve across actual operating flight hours
    hr_frames = []
    for h in range(24):
        s_info = sched_slots.get(str(h))
        s_min = int(s_info[0]) if s_info else dep_time_val.minute
        # Realistic diurnal inbound aircraft lag progression across the day
        typ_prev = 0.0 if h < 10 else (6.0 if h < 15 else 14.0)
        if wx_stress_boost > 0 and prev_delay is not None and h >= 12:
            typ_prev = max(typ_prev, prev_delay * 0.65)
        hr_frames.append(
            build_single_feature_vector(
                bundle, origin, dest, carrier, fl_date, dt_time(h, s_min),
                temp_f, dew_f, wind_kt, vis_mi, precip_in, pressure_mb,
                traffic_mult=traffic_mult, prev_aircraft_delay=typ_prev,
            )[0]
        )

    raw_24h = bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(pd.concat(hr_frames, ignore_index=True))[:, 1]
    cal_24h = (_calibrate_prob(raw_24h.astype(float), prior) * 100.0)

    hourly_curve = []
    active_hours = []
    candidate_slots = []
    for h in range(24):
        s_info = sched_slots.get(str(h))
        if s_info is not None:
            s_min, emp_pct = int(s_info[0]), float(s_info[1])
            # Blend live XGBoost weather/carrier response with actual BTS route-hour empirical delay rate
            w_xgb = 0.72 if wx_stress_boost > 0 else 0.45
            blended = round(w_xgb * float(cal_24h[h]) + (1.0 - w_xgb) * emp_pct, 1)
            hourly_curve.append(blended)
            active_hours.append(True)
            candidate_slots.append((blended, h, s_min))
        else:
            hourly_curve.append(round(float(cal_24h[h]), 1))
            active_hours.append(False)

    if candidate_slots:
        best_risk, best_hr, best_min = min(candidate_slots, key=lambda x: x[0])
    else:
        best_hr = int(np.argmin(cal_24h))
        best_min = dep_time_val.minute
        best_risk = round(float(cal_24h[best_hr]), 1)

    reasons = []
    if wind_kt >= 25.0:
        reasons.append(f"Surface winds ({wind_kt:.0f} kt) exceed FAA 25 kt crosswind threshold")
    if vis_mi < 3.0:
        reasons.append(f"Low IFR visibility ({vis_mi:.1f} mi) increases ATC runway spacing")
    if temp_f <= 33.0 and precip_in > 0.01:
        reasons.append(f"Freezing precipitation ({temp_f:.0f}°F) requires de-icing pad queue")
    if (prev_delay or 0) >= 15.0:
        reasons.append(f"Inbound aircraft carrying +{prev_delay:.0f} min prior-leg turnaround delay")
    if dep_time_val.hour >= 17:
        reasons.append(f"Evening departure bank ({dep_time_val.strftime('%H:%M')}) faces cumulative network ripple")
    if not reasons:
        reasons.append(f"Clear surface weather and smooth {origin} departure bank")

    block_min = int(round(ctx["distance_mi"] / 7.5 + 35.0))
    return JSONResponse({
        "origin": ctx["orig_ap"],
        "dest": ctx["dest_ap"],
        "carrier": carrier,
        "distance_mi": int(ctx["distance_mi"]),
        "block_time_str": f"{block_min // 60}h {block_min % 60:02d}m",
        "risk_pct": primary_pct,
        "threshold_pct": round(cal_cutoff * 100.0, 1),
        "is_delayed": is_delayed,
        "verdict": "DELAY EXPECTED" if is_delayed else "ON SCHEDULE",
        "est_delay_min": est_delay_min,
        "hourly_curve": hourly_curve,
        "active_hours": active_hours,
        "schedule_scope": schedule_scope,
        "best_slot": f"{best_hr:02d}:{best_min:02d}",
        "best_slot_risk_pct": best_risk,
        "reasons": reasons[:3],
    })


app = Starlette(
    routes=[
        Route("/", index_page),
        Route("/app", index_page),
        Route("/app/", index_page),
        Route("/api/meta", api_meta),
        Route("/api/weather", api_live_weather),
        Route("/api/predict", api_predict, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")
