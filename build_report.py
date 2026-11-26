"""
Generates the complete, print-ready academic project report in both Markdown
(Flight_Delay_Prediction_Final_Report.md) and typeset PDF
(Flight_Delay_Prediction_Final_Report.pdf) with embedded evaluation plots.
"""

from __future__ import annotations
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section
import fitz

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

CSS = """
@page {
    size: A4;
    margin: 22mm 20mm 22mm 20mm;
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.2pt;
    line-height: 1.55;
    color: #1e293b;
}
h1 {
    color: #0f294a;
    font-size: 20pt;
    border-bottom: 2px solid #0f294a;
    padding-bottom: 5px;
    margin-top: 8px;
    margin-bottom: 12px;
}
h2 {
    color: #1e3a5f;
    font-size: 13.5pt;
    margin-top: 16px;
    margin-bottom: 7px;
}
h3 {
    color: #334155;
    font-size: 11.2pt;
    margin-top: 12px;
    margin-bottom: 5px;
}
p {
    margin-top: 0;
    margin-bottom: 9px;
    text-align: justify;
}
ul, ol {
    margin-top: 4px;
    margin-bottom: 10px;
    padding-left: 20px;
}
li {
    margin-bottom: 4px;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 14px 0;
    font-size: 9.2pt;
}
th {
    color: #0f294a;
    font-weight: bold;
    text-align: left;
    padding: 6.5px 8px;
    border-top: 2px solid #0f294a;
    border-bottom: 1.5px solid #0f294a;
    border-left: 1px solid #cbd5e1;
    border-right: 1px solid #cbd5e1;
}
td {
    padding: 6px 8px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
}
code {
    font-family: Courier, monospace;
    font-size: 9pt;
    color: #0f294a;
}
blockquote {
    border-left: 4px solid #1d4ed8;
    padding: 6px 12px;
    margin: 10px 0 12px 0;
    color: #1e3a8a;
    font-size: 9.6pt;
}
img {
    width: 92%;
    height: auto;
    display: block;
    margin: 8px auto 4px auto;
}
.caption {
    text-align: center;
    font-size: 8.8pt;
    color: #475569;
    font-style: italic;
    margin-bottom: 14px;
}
.cover {
    text-align: center;
    padding-top: 70px;
}
.cover-title {
    font-size: 27pt;
    font-weight: bold;
    color: #0f294a;
    margin-bottom: 10px;
    line-height: 1.2;
}
.cover-sub {
    font-size: 14pt;
    color: #334155;
    margin-bottom: 44px;
}
.cover-meta {
    font-size: 11.2pt;
    color: #1e293b;
    line-height: 1.9;
    margin-bottom: 36px;
}
.cover-box {
    border: 1px solid #cbd5e1;
    background-color: #f8fafc;
    padding: 16px 22px;
    margin: 24px auto;
    width: 82%;
    text-align: left;
    font-size: 10pt;
}
"""

SEC_COVER = """
<div class="cover">
  <div class="cover-title">Predicting Flight Delays</div>
  <div class="cover-sub">A Big Data and Machine Learning Approach on 7.08 Million U.S. Flights</div>

  <div class="cover-meta">
    <b>Course:</b> Big Data Analytics and Business Intelligence (CBCA 293)<br/>
    <b>Program:</b> B.Tech Computer Science and Engineering<br/>
    <b>Institution:</b> Bennett University (The Times Group), Greater Noida<br/>
    <b>Supervisor:</b> Associate Professor, School of Computer Science Engineering and Technology (SCSET)
  </div>

  <div class="cover-box">
    <b>Submitted by:</b><br/>
    1. <b>Parth</b> (S24CSEU2470) — Data Acquisition, Out-of-Core Warehouse &amp; Cleaning Pipeline<br/>
    2. <b>Vipul Gupta</b> (S24CSEU2526) — Cyclical Temporal Geometry, Hub Congestion &amp; Timetable Profiles<br/>
    3. <b>Krishna Sharma</b> (S24CSEU2756) — Chronological Split, Sensor-Aligned SMOTE &amp; Ensemble Training<br/>
    4. <b>Abhijeet Rana</b> (S24CSEU1419) — 1.13M-Flight Evaluation, Threshold Tuning &amp; SKY/MOVE Web Studio
  </div>

  <div style="margin-top:34px; font-size:9.8pt; color:#475569;">
    <b>Repository:</b> https://github.com/parth7k3/flight-delay-prediction<br/>
    <b>Academic Year:</b> 2025–2026
  </div>
</div>
"""

SEC_DECLARATION = """
# Declaration & Individual Team Contributions

## Honesty Declaration and Plagiarism Statement
We declare that this project report and accompanying codebase represent our original academic work carried out for the course **Big Data Analytics and Business Intelligence (CBCA 293)** at the School of Computer Science Engineering and Technology (SCSET), Bennett University. All external public datasets (U.S. DOT Bureau of Transportation Statistics and NOAA/NWS ASOS surface weather archives), literature references, and open-source software libraries have been cited explicitly.

**Turnitin similarity score:** `[   ]%` *(to be verified against the university threshold upon final submission).*

## Authors & Enrollment Registry

| No. | Team Member | Enrollment Number | Primary Modules Owned |
| :--- | :--- | :--- | :--- |
| 1 | **Parth** | `S24CSEU2470` | `download_real_bigdata.py`, `ingestion.py`, `spark_clean_join.py`, `sql/schema_postgres.sql` |
| 2 | **Vipul Gupta** | `S24CSEU2526` | `feature_engineering.py`, `build_route_profiles.py`, `config.py` (Feature Schema) |
| 3 | **Krishna Sharma** | `S24CSEU2756` | `balance_classes.py`, `train_models.py` |
| 4 | **Abhijeet Rana** | `S24CSEU1419` | `evaluate.py`, `server.py`, `web/index.html`, `app.py` |

## Detailed Technical Contributions

| Team Member | Technical Responsibilities & Verified Deliverables |
| :--- | :--- |
| **Parth** (`S24CSEU2470`) | Built parallel bulk acquisition for all 12 monthly 2024 U.S. DOT BTS partitions (**7,079,061** flights) and 59-hub NOAA ASOS hourly weather (**518,256** observations with UTC-to-local IANA timezone conversion). Engineered out-of-core DuckDB columnar ingestion with composite indexes, PostgreSQL range-partitioned DDL, cancellation/diversion filtering (`96,315` cancelled + `17,499` diverted dropped), 3-hour storm persistence windows, and ZSTD Parquet export. |
| **Vipul Gupta** (`S24CSEU2526`) | Designed the out-of-core SQL window feature engineering pipeline across **5,654,432** weather-aligned flights: 24-hour and weekday sine/cosine circular time encodings, rolling 3-hour origin hub departure volume and capacity-normalized congestion ratios, dew-point fog spread, zero-leakage `T - 2h` inbound aircraft turnaround lag by `TAIL_NUM`, `T - 2h` airport disruption rates, and `5,340` empirical airline-route timetables. |
| **Krishna Sharma** (`S24CSEU2756`) | Implemented the strict 80/20 chronological split (`4,523,544` train vs. `1,130,888` untouched future test) to eliminate temporal leakage. Engineered training-only SMOTE oversampling (`372,426` balanced vectors) with geometric unit-circle projection and ASOS sensor-grid quantization. Ran 3-fold fold-internal SMOTE cross-validation and trained Logistic Regression, Random Forest (100 Trees), Histogram Gradient Boosting, Tuned Histogram XGBoost, and the secondary XGBoost delay-minutes regressor. |
| **Abhijeet Rana** (`S24CSEU1419`) | Built the 1.13-million-flight untouched evaluation pipeline computing Accuracy, Precision, Recall, F1, F2, ROC-AUC, PR-AUC, and operational threshold sweeps; exported all five publication-grade plots; and built the minimalist **SKY / MOVE** live Prediction Studio (`server.py` + `web/index.html` + simplified `app.py`) featuring Bayes prior calibration, live Open-Meteo METAR weather fetching, and route-specific lowest-risk flight slot recommendations. |
"""

SEC_CH1 = """
# Chapter 1: Problem Statement & Literature Review

## 1.1 Problem Motivation and Operational Realities
Commercial aviation operates on tightly coupled hub-and-spoke networks with narrow turnaround buffers. When a single departure is delayed at a major origin hub such as Atlanta (`ATL`), Chicago O'Hare (`ORD`), or Dallas/Fort Worth (`DFW`), disruptions propagate rapidly across downstream legs: inbound aircraft arrive late for their next departure, flight crews approach FAA duty-time limits, gates back up, and passengers miss tight hub connections. According to Airlines for America (A4A) industry benchmarks, aircraft block delays cost U.S. carriers over **$101 per minute** in extra fuel burn, crew overtime, gate fees, and passenger rebooking.

Predicting departure delay risk **two hours prior to scheduled pushback (`T - 2h`)** gives Airline Operations Control Centers (AOCC) actionable lead time to swap tail assignments, hold adjacent gates, allocate de-icing pads, and proactively rebook at-risk connecting passengers before disruptions compound.

## 1.2 Problem Formulation
Given a scheduled commercial flight (origin hub, destination airport, operating airline carrier, route distance, scheduled block time, and departure time slot), origin surface weather observations, and strictly causal `T - 2h` airport/aircraft turnaround telemetry:

- **Primary Model (Binary Classification):** Predict `DEP_DEL15`, where **Class 0** indicates an on-time departure (`DEP_DELAY < 15` minutes) and **Class 1** indicates an official FAA delay (`DEP_DELAY >= 15` minutes).
- **Secondary Baseline (Continuous Regression):** Predict continuous departure delay length in minutes (`DEP_DELAY`) to quantify operational severity.

Three compounding technical hurdles define this task:
1. **Multi-Million-Row Data Volume:** Full-year 2024 U.S. DOT BTS logs contain **7,079,061** domestic records (`652.6 MB` of raw monthly CSV partitions), requiring out-of-core columnar processing rather than naive in-memory Pandas loading.
2. **Heterogeneous Spatio-Temporal Alignment:** BTS scheduled departures (`CRSDepTime`) are recorded in **local airport wall-clock time**, whereas NOAA/NWS ASOS METAR logs are archived in **UTC** at `:51`–`:54` minute offsets. Joining both without timezone normalization silently misaligns storms by 5 to 8 hours.
3. **Operational 80/20 Class Imbalance:** Across our cleaned 2024 hub dataset, only **21.34%** of flights experience a departure delay of 15 minutes or more. A naive classifier predicting "on-time" for every flight achieves ~79% accuracy while achieving **0.00 recall** on actual disruptions.

## 1.3 Review of Existing Aviation ML Literature
- **Queuing & Econometric Baselines (Hansen et al., 2002):** Early aviation studies modeled runway delays using aggregate queuing theory. While informative for long-term airport expansion, aggregate queues cannot score individual flight risk under rapidly shifting local weather.
- **Non-Linear Weather Thresholds (Choi et al., 2016):** Evaluated Decision Trees, Random Forests, and Neural Networks on BTS flights joined with surface weather. Tree ensembles consistently outperformed linear baselines because surface weather exhibits sharp physical thresholds—a 7-knot breeze has negligible impact, whereas crosswinds exceeding **25 knots** or IFR visibility dropping below **3 statute miles** trigger immediate runway restrictions.
- **Downstream Network Ripple Effects (Rebollo & Balakrishnan, 2014):** Demonstrated that a substantial fraction of departure delays stem not from local weather alone, but from late inbound aircraft arriving from previous legs and cumulative hub departure congestion across afternoon/evening banks.
- **Class Imbalance & Leakage Control (Chawla et al., 2002; Belcastro et al., 2016; Sternberg et al., 2017):** Established that raw accuracy is misleading on skewed aviation datasets and proved that synthetic oversampling (`SMOTE`) must be confined strictly inside training folds—never before the train/test split—to prevent interpolated points from leaking into evaluation.
"""

SEC_CH2 = """
# Chapter 2: System Architecture & Integrated Datasets

## 2.1 Three-Tier Authentic Empirical Datasets
Unlike toy or synthetic benchmarks, our pipeline ingests **full-year 2024 U.S. government aviation and meteorological archives** across three integrated tiers:

| Tier | Dataset Source | Volume & Coverage | Core Variables |
| :--- | :--- | :--- | :--- |
| **Primary** | U.S. DOT Bureau of Transportation Statistics (`transtats.bts.gov/PREZIP`) | **7,079,061** flights across 12 monthly CSV partitions (`2024-01` to `2024-12`, `652.6 MB`) | `FL_DATE`, `OP_UNIQUE_CARRIER`, `TAIL_NUM`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DEP_DELAY`, `DEP_DEL15`, `CANCELLED`, `DIVERTED`, `CRS_ELAPSED_TIME`, `DISTANCE` |
| **Secondary** | NOAA / NWS ASOS Hourly Surface Weather (`mesonet.agron.iastate.edu`) | **518,256** hourly local-aligned METAR observations across **59 major U.S. hubs** | `STATION_IATA`, `OBS_TIMESTAMP`, `TEMP_F`, `DEW_POINT_F`, `WIND_SPEED_KT`, `VISIBILITY_MI`, `PRECIP_IN`, `PRESSURE_MB`, `WEATHER_CONDITION` |
| **Tertiary** | U.S. Airport & Reporting Carrier Metadata Registries | **60 major U.S. commercial hubs** & **15 official BTS reporting airlines** | IATA codes, full names, city/state, GPS coordinates (`LAT`, `LON`), IANA timezone (`TZ`), hourly runway capacity, fleet operating model |

## 2.2 End-to-End 7-Stage Out-of-Core Architecture
To process over seven million rows on standard hardware without out-of-memory crashes, every data-intensive stage executes out-of-core using **DuckDB** columnar vectorized SQL, **PyArrow**, and **ZSTD-compressed Parquet**, paired with a PostgreSQL range-partitioned schema (`sql/schema_postgres.sql`) and PySpark-compatible SQL transformation DAG:

1. **Stage 1 — Ingestion & Indexed Warehouse (`ingestion.py`):** Streams all 12 monthly BTS CSV partitions (`7,079,061` rows) and `518,256` NOAA ASOS observations into `aviation_warehouse.duckdb` in **`8.38s`**, verifying schema integrity and building composite indexes on `(ORIGIN, CRS_DEP_TIMESTAMP)` and `(STATION_IATA, OBS_TIMESTAMP)`.
2. **Stage 2 — Cleaning, Storm Windows & Weather Join (`spark_clean_join.py`):** Filters out **`96,315`** cancelled flights (`CANCELLED == 1`) and **`17,499`** diverted flights (`DIVERTED == 1`), verifies zero NULLs on critical keys, computes 3-hour storm persistence windows (`wind_max_3h`, `precip_sum_3h`, `vis_min_3h`), joins **`5,654,432`** active hub flights to local hourly weather in **`3.77s`**, and exports the physical execution plan (`outputs/plans/pyspark_execution_plan.txt`).
3. **Stage 3 — Cyclical Geometry & Ripple Features (`feature_engineering.py`):** Engineers **26 features** out-of-core across all `5,654,432` flights in **`5.95s`**, including 24-hour and weekday circular sine/cosine encodings, rolling 3-hour origin departure load (`origin_rolling_3h_vol`), capacity-normalized congestion ratio, dew-point fog spread, zero-leakage `T - 2h` inbound aircraft `TAIL_NUM` turnaround lag, and `T - 2h` hub delay rates.
4. **Stage 4 — Chronological Split & Sensor-Aligned SMOTE (`balance_classes.py`):** Partitions strictly by timestamp at **`2024-10-19 22:15:00`** (`4,523,544` Jan–Oct training rows vs. `1,130,888` untouched Oct–Dec test rows). Applies training-only SMOTE (`239,768` stratified training sample → `372,426` 50/50 balanced vectors) followed by **unit-circle re-projection** and **ASOS sensor-grid quantization**.
5. **Stage 5 — Scalable Ensemble Training (`train_models.py`):** Runs 3-fold fold-internal SMOTE cross-validation and trains Logistic Regression, Random Forest (100 Trees), Histogram Gradient Boosting (`HistGradientBoostingClassifier`), Tuned Histogram XGBoost (`tree_method='hist'`), and a secondary XGBoost continuous delay regressor.
6. **Stage 6 — 1.13M-Flight Evaluation & Threshold Tuning (`evaluate.py`):** Scores every classifier on **all `1,130,888` untouched future test flights**, sweeps operational decision thresholds (F1 vs. recall-prioritized F2), and exports publication plots.
7. **Stage 7 — Live Prediction Studio (`server.py`, `web/index.html`, `app.py`):** Serves the minimalist **SKY / MOVE** web studio with Bayes prior calibration, live Open-Meteo METAR weather fetching, and route-specific lowest-risk flight slot recommendations derived from **5,340** real 2024 airline-route timetables.
"""

SEC_CH3 = """
# Chapter 3: Implementation Highlights & Key Engineering Decisions

## 3.1 Modular Codebase Structure
The repository (`https://github.com/parth7k3/flight-delay-prediction`) is organized into discrete, single-responsibility modules orchestrated end-to-end by `run_pipeline.py`:

| Module | Operational Role | Runtime (7.08M Scale) |
| :--- | :--- | :---: |
| `download_real_bigdata.py` | Parallel U.S. DOT BTS (12 zips) & NOAA ASOS (59 hubs) bulk acquisition | ~325s (one-time) |
| `ingestion.py` | Out-of-core DuckDB warehouse ingestion, NULL audit & composite indexing | **8.38s** |
| `spark_clean_join.py` | Cancellation/diversion filtering, 3h storm windows & local-hour weather join | **3.77s** |
| `feature_engineering.py` | Out-of-core SQL window features, cyclical sin/cos & leak-free 80% priors | **5.95s** |
| `build_route_profiles.py` | Extracts 5,340 real airline-route schedules & 2,423 nonstop corridor timetables | **2.10s** |
| `balance_classes.py` | Chronological 80/20 split & training-fold geometric/sensor-aligned SMOTE | **3.49s** |
| `train_models.py` | Fold-safe 3-fold CV & training of 4 classifiers + secondary delay regressor | **22.1s** |
| `evaluate.py` | Evaluation on 1,130,888 untouched test flights, threshold sweep & plot export | **18.4s** |
| `server.py` & `web/index.html` | Minimalist SKY / MOVE Starlette web studio, REST API & Live METAR client | `< 20ms` / query |

## 3.2 Solving SMOTE Geometric & Sensor-Grid Artifacts
During empirical validation on real 2024 NOAA ASOS data, we uncovered a critical real-world pitfall of naive SMOTE on tree ensembles:
- **Off-Circle Chord Interpolation:** Linear SMOTE interpolation between two points on the 24-hour unit circle `(dep_hour_sin, dep_hour_cos)` or 7-day circle `(dow_sin, dow_cos)` produces synthetic vectors **inside** the circle (`sin^2 + cos^2 < 1`), which deep boosted trees exploit as a shortcut to distinguish synthetic Class 1 rows from real rows.
- **Sub-Unit Instrument Precision:** Real NOAA ASOS wind speeds (`WIND_SPEED_KT`) are reported in whole knots and visibility in half-mile increments, whereas naive SMOTE generates fractional values (e.g., `7.43 kt`) that never occur in real METAR reports.

**Engineering Fix (`balance_classes.py`):** Immediately after SMOTE resampling, we project interpolated circular coordinates back onto the unit circle via `arctan2(sin, cos)` and snap synthetic weather/schedule measurements onto the exact physical ASOS sensor grid. This eliminated synthetic overfitting and lifted untouched test Accuracy and ROC-AUC cleanly across every tree ensemble.

## 3.3 Preventing Temporal & Target Leakage
- **Strict Chronological Partitioning:** Training ends at `2024-10-19 22:15:00` (`4,523,544` flights); evaluation covers strictly unseen future operations from `2024-10-19` through `2024-12-31` (`1,130,888` flights).
- **Causal `T - 2h` Lag Windows:** Inbound aircraft delay (`prev_aircraft_delay_t2h`) only reads a `TAIL_NUM`'s previous leg if it departed **at least 115 minutes earlier**, and hub disruption rates (`origin_t2h_delay_rate`) aggregate strictly over `ROWS BETWEEN 5 PRECEDING AND 2 PRECEDING` (2 to 5 hours prior)—guaranteeing zero future information leakage at the 2-hour operational prediction horizon.

![Class Balance Progression](outputs/plots/class_balance_smote.png)
<div class="caption">Figure 3.1: Full 80% chronological training split (4.52M flights), training-only SMOTE balanced fold (372K vectors), and completely untouched 20% future test split (1.13M flights).</div>
"""

SEC_CH4 = """
# Chapter 4: Experimental Results & Model Benchmarks

## 4.1 Untouched 1.13-Million-Flight Chronological Test Benchmarks
All candidate models were trained on the sensor-aligned SMOTE training fold (`372,426` vectors) and evaluated on **all `1,130,888` untouched Oct–Dec 2024 flights** (`17.49%` empirical delay rate). Decision thresholds were calibrated strictly on a held-out validation slice from the training period:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | 3-Fold CV AUC | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 81.23% | 0.465 | 0.489 | 0.477 | 0.782 | 0.751 | 0.501 |
| **Random Forest (100 Trees)** | 82.43% | 0.497 | 0.461 | 0.478 | 0.787 | 0.759 | 0.521 |
| **Gradient Boosting (HistGBDT)** | 83.22% | 0.524 | 0.452 | **0.485** | **0.790** | 0.763 | 0.532 |
| **XGBoost Classifier (Tuned)** | **83.38%** | **0.529** | 0.446* | 0.484 | 0.788 | **0.764** | **0.534** |

- **Secondary Continuous Delay Regressor (`XGBRegressor`):** Achieved **MAE = `14.76` minutes** and **RMSE = `31.27` minutes** across all `1,130,888` test flights.
- **Monotonic Ensemble Gain:** Accuracy, Precision, ROC-AUC, and PR-AUC improve monotonically from Logistic Regression (`0.751` AUC, `0.501` PR-AUC) → Random Forest (`0.759` AUC) → Histogram Gradient Boosting (`0.763` AUC) → Tuned XGBoost (`0.764` AUC, **`0.534` PR-AUC**—more than **3.05×** the `0.175` positive class baseline).

![ROC and Precision-Recall Curves](outputs/plots/roc_pr_curves.png)
<div class="caption">Figure 4.1: Receiver Operating Characteristic (ROC) and Precision-Recall curves evaluated across 1,130,888 untouched late-2024 U.S. domestic flights.</div>
"""

SEC_CH4_PART2 = """
## 4.2 Confusion Matrices & Operational Threshold Tuning

![Confusion Matrices](outputs/plots/confusion_matrices.png)
<div class="caption">Figure 4.2: Confusion matrices across all four candidate architectures on the 1,130,888-flight chronological test partition.</div>

In commercial airline dispatch, **False Negatives** (failing to flag a real delay, leading to missed hub connections) are significantly costlier than **False Positives** (proactively checking an on-time flight). Sweeping the probability threshold on Tuned XGBoost demonstrates this operational lever directly:
- At the balanced F1-optimal cutoff (`0.525`), XGBoost achieves `82.6%` accuracy, `0.502` precision, and `0.458` recall.
- Lowering the cutoff to the recall-prioritized **F2-optimal cutoff (`0.275`)** increases delay detection **Recall to `78.1%`** (catching `154,527` of `197,767` delayed flights across the test quarter).
"""

SEC_CH4_PART3 = """
## 4.3 Threshold Trade-Off & Feature Importance Rankings

![Operational Threshold Tuning](outputs/plots/threshold_tuning.png)
<div class="caption">Figure 4.3: Precision, Recall, F1, and recall-weighted F2 operational score across classification probability thresholds.</div>

![Feature Importance Rankings](outputs/plots/feature_importance.png)
<div class="caption">Figure 4.4: Tuned XGBoost relative gain feature importance rankings confirming that inbound aircraft turnaround lag, cyclical departure slots, T-2h hub cascade rates, and 3-hour storm persistence dominate predictive power.</div>
"""

SEC_CH5 = """
# Chapter 5: Live Demo, Limitations & Future Work

## 5.1 Live Serving Interface — SKY / MOVE Prediction Studio
To bridge offline Big Data modeling with real-time operational decision-making, we packaged the trained ensemble bundle (`flight_delay_models.joblib`, `12.3 MB`) and real 2024 route timetables (`route_slot_profiles.json`) into a lightweight Starlette/Uvicorn web studio (`server.py` + `web/index.html`) alongside a matching simplified Streamlit interface (`app.py`):

- **Three Focused Workspaces:** *Prediction Studio* (live flight queries), *How It Works* (7.08M-row pipeline architecture & live counters), and *Model Intelligence* (1.13M-flight benchmarks & top predictive signals).
- **Bayes Prior Calibration:** Converts 50/50 SMOTE tree posteriors back onto the true empirical probability scale (`prior = 0.2134`) so displayed delay risks match real-world aviation rates.
- **Live METAR Integration:** A one-click **Live METAR** action fetches real-time surface weather (temperature, dew point, wind knots, visibility, precipitation, pressure) at the selected origin airport's GPS coordinates via Open-Meteo and runs instant inference.
- **Route-Specific Lowest-Risk Slot Optimizer:** Evaluates only **actual scheduled commercial operating hours** on the selected `(ORIGIN, DEST, CARRIER)` corridor from 2024 BTS timetables (`5,340` airline-routes), dimming unscheduled hours and recommending the lowest-risk real departure time.

## 5.2 Technical Limitations Observed
1. **Unscheduled ATC Ground Stops:** Sudden high-altitude airspace closures, security incidents, or short-notice Ground Delay Programs (GDPs) triggered less than 2 hours before departure cannot be anticipated from surface weather and schedule data alone.
2. **Seasonal Extrapolation in Single-Year Splits:** Because the 80/20 chronological split trains on Jan–Oct and tests on Oct–Dec, winter holiday peak patterns (Thanksgiving/Christmas) appear only in the test partition, slightly capping out-of-sample AUC compared to multi-year training (`2022–2024`).
3. **Regional Spoke Coverage:** Weather alignment covers the 59 busiest U.S. hubs (`~80%` of national commercial departures); smaller regional feeder airports outside those 59 ASOS stations require nearest-station spatial interpolation.

## 5.3 Directions for Future Work
- **Multi-Year Rolling Retraining (2021–2025):** Training across 25M+ flights spanning multiple full seasonal cycles will capture holiday and winter-storm regimes in both train and test partitions.
- **Graph Neural Networks (GNNs) over Tail Rotations:** Modeling the full U.S. hub-and-spoke network as a spatio-temporal graph where aircraft tail numbers (`TAIL_NUM`) propagate delay embeddings across multi-hop daily itineraries.
- **Live FAA SWIM & Maintenance Telemetry:** Integrating real-time FAA System Wide Information Management (SWIM) airspace congestion feeds for continuous `T - 60m` risk updates.

## References
1. **U.S. Department of Transportation — Bureau of Transportation Statistics (BTS).** *Reporting Carrier On-Time Performance Database (2024 Monthly Partitions).* `https://transtats.bts.gov/PREZIP/`
2. **NOAA / National Weather Service & Iowa Environmental Mesonet (IEM).** *ASOS/AWOS Automated Surface Observing System Hourly METAR Archive.* `https://mesonet.agron.iastate.edu/request/download.phtml`
3. **Choi, S., Kim, Y. J., Briceno, S., & Mavris, D. (2016).** *Prediction of Weather-Induced Airline Delays Based on Machine Learning Algorithms.* IEEE/AIAA 35th Digital Avionics Systems Conference (DASC).
4. **Rebollo, J. J., & Balakrishnan, H. (2014).** *Characterization and Prediction of Air Traffic Delays.* Transportation Research Part C: Emerging Technologies, 44, 231–241.
5. **Sternberg, A., Soares, J., Carvalho, D., & Ogasawara, E. (2017).** *A Review on Flight Delay Prediction.* arXiv:1703.06118.
6. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.
7. **Belcastro, L., Marozzo, F., Talia, D., & Trunfio, P. (2016).** *Using Scalable Data Mining for Predicting Flight Delays.* ACM Transactions on Intelligent Systems and Technology (TIST), 8(1), 1–20.
8. **Hansen, M. (2002).** *Micro-level Analysis of Airport Delay Externalities Using Deterministic Queuing Models.* Journal of Air Transport Management, 8(2), 73–87.
"""


def build_report() -> None:
    sections = [SEC_COVER, SEC_DECLARATION, SEC_CH1, SEC_CH2, SEC_CH3, SEC_CH4, SEC_CH4_PART2, SEC_CH4_PART3, SEC_CH5]

    # 1. Write full Markdown version
    md_path = BASE_DIR / "Flight_Delay_Prediction_Final_Report.md"
    md_text = "\n\n---\n\n".join(s.strip() for s in sections)
    md_path.write_text(md_text, encoding="utf-8")

    # 2. Compile print-ready PDF with embedded high-res plots
    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = "Predicting Flight Delays: A Big Data and Machine Learning Approach"
    pdf.meta["author"] = "Parth, Vipul Gupta, Krishna Sharma, Abhijeet Rana (Bennett University)"
    pdf.meta["subject"] = "CBCA 293 — Big Data Analytics and Business Intelligence"

    for idx, sec in enumerate(sections):
        pdf.add_section(Section(sec.strip(), toc=(idx > 0), root=str(BASE_DIR)), user_css=CSS)

    pdf_path = BASE_DIR / "Flight_Delay_Prediction_Final_Report.pdf"
    pdf.save(str(pdf_path))

    # Add subtle academic footer + page numbers (skip cover page 1)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    for page_idx in range(1, total_pages):
        page = doc[page_idx]
        rect = page.rect
        footer_y = rect.height - 28
        page.draw_line(fitz.Point(56, footer_y - 8), fitz.Point(rect.width - 56, footer_y - 8), color=(0.8, 0.84, 0.88), width=0.6)
        page.insert_text(
            fitz.Point(56, footer_y + 4),
            "Predicting Flight Delays: A Big Data and Machine Learning Approach (CBCA 293)",
            fontsize=8,
            color=(0.35, 0.42, 0.50),
        )
        page.insert_text(
            fitz.Point(rect.width - 96, footer_y + 4),
            f"Page {page_idx + 1} of {total_pages}",
            fontsize=8,
            color=(0.35, 0.42, 0.50),
        )
    final_tmp = BASE_DIR / "Flight_Delay_Prediction_Final_Report_numbered.pdf"
    doc.save(str(final_tmp), garbage=4, deflate=True)
    doc.close()
    final_tmp.replace(pdf_path)

    print(f"[build_report] Generated {pdf_path.name} ({total_pages} pages, {round(pdf_path.stat().st_size / 1024, 1)} KB) & {md_path.name}")


if __name__ == "__main__":
    build_report()
