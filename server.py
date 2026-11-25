"""
Lightweight FastAPI/Starlette Live Web Server (server.py)
Serves the minimalist SKY / MOVE Flight Delay Intelligence Studio UI + JSON ML API
backed by the 7.08M-row U.S. DOT BTS + 518K-row NOAA ASOS Tuned XGBoost pipeline.
"""

from __future__ import annotations
import json
from datetime import date, datetime, time as dt_time
from pathlib import Path
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


def _get_state() -> tuple[dict, dict]:
    global _BUNDLE, _ARTIFACTS
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
    return _BUNDLE, _ARTIFACTS


async def index_page(request: Request) -> HTMLResponse:
    return HTMLResponse(WEB_HTML_PATH.read_text(encoding="utf-8"))


async def api_meta(request: Request) -> JSONResponse:
    bundle, art = _get_state()
    return JSONResponse({
        "airports": AIRPORTS_REGISTRY,
        "airlines": AIRLINES_REGISTRY,
        "default_threshold": float(bundle.get("default_threshold", 0.50)),
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
    bundle, _ = _get_state()
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
    threshold = float(body.get("threshold", bundle.get("default_threshold", 0.50)))

    prev_delay_in = body.get("prev_delay", None)
    prev_delay = float(prev_delay_in) if prev_delay_in is not None else None

    X_q, ctx = build_single_feature_vector(
        bundle, origin, dest, carrier, fl_date, dep_time_val,
        temp_f, dew_f, wind_kt, vis_mi, precip_in, pressure_mb,
        traffic_mult=traffic_mult, prev_aircraft_delay=prev_delay,
    )

    model_probs = {
        name: round(float(clf.predict_proba(X_q)[0, 1]) * 100.0, 1)
        for name, clf in bundle["classifiers"].items()
    }
    primary_pct = model_probs["XGBoost Classifier (Tuned)"]
    raw_min = float(bundle["delay_regressor"].predict(X_q)[0])
    is_delayed = (primary_pct / 100.0) >= threshold
    est_delay_min = int(round(max(16.0, raw_min))) if is_delayed else int(round(max(0.0, raw_min)))

    # 24-hour slot sweep to suggest optimal departure window
    hr_frames = [
        build_single_feature_vector(
            bundle, origin, dest, carrier, fl_date, dt_time(h, dep_time_val.minute),
            temp_f, dew_f, wind_kt, vis_mi, precip_in, pressure_mb,
            traffic_mult=traffic_mult, prev_aircraft_delay=(0.0 if h < 9 else prev_delay),
        )[0]
        for h in range(24)
    ]
    probs_24h = np.round(
        bundle["classifiers"]["XGBoost Classifier (Tuned)"].predict_proba(pd.concat(hr_frames, ignore_index=True))[:, 1] * 100.0,
        1,
    ).tolist()
    best_hr = int(np.argmin(probs_24h))

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
        reasons.append("Nominal surface weather and clear runway departure slot")

    block_min = int(round(ctx["distance_mi"] / 7.5 + 35.0))
    return JSONResponse({
        "origin": ctx["orig_ap"],
        "dest": ctx["dest_ap"],
        "carrier": carrier,
        "distance_mi": int(ctx["distance_mi"]),
        "block_time_str": f"{block_min // 60}h {block_min % 60:02d}m",
        "risk_pct": primary_pct,
        "threshold_pct": round(threshold * 100.0, 0),
        "is_delayed": is_delayed,
        "verdict": "DELAY EXPECTED" if is_delayed else "ON SCHEDULE",
        "est_delay_min": est_delay_min,
        "model_probs": model_probs,
        "hourly_curve": probs_24h,
        "best_slot": f"{best_hr:02d}:{dep_time_val.minute:02d}",
        "best_slot_risk_pct": probs_24h[best_hr],
        "reasons": reasons[:3],
        "applied_weather": {
            "temp_f": round(temp_f),
            "dew_f": round(dew_f),
            "wind_kt": round(wind_kt),
            "vis_mi": round(vis_mi, 1),
            "precip_in": round(precip_in, 2),
        },
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
