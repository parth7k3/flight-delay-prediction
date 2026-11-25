# Predicting Flight Delays: A Big Data and Machine Learning Approach

**Course:** Big Data Analytics and Business Intelligence (`CBCA 293`)  
**Institution:** School of Computer Science Engineering and Technology (SCSET), Bennett University, Greater Noida  

### Authors & Modular Contributions
| No. | Name | Enrollment No. | Assigned Deliverables & Modules |
| :--- | :--- | :--- | :--- |
| 1 | **Parth** | `S24CSEU2470` | Raw dataset ingestion, PostgreSQL/SQLite indexed storage setup, PySpark data cleaning & NULL handling (`ingestion.py`, `spark_clean_join.py`, `sql/schema_postgres.sql`) |
| 2 | **Vipul Gupta** | `S24CSEU2526` | Cyclical 24-hour timestamp trigonometric encoding ($\sin/\cos$ transforms) & rolling airport departure congestion indicators (`feature_engineering.py`) |
| 3 | **Krishna Sharma** | `S24CSEU2756` | Chronological 80/20 partitioning, training-fold SMOTE class balancing, cross-validation, Random Forest, Gradient Boosting & Tuned XGBoost (`balance_classes.py`, `train_models.py`) |
| 4 | **Abhijeet Rana** | `S24CSEU1419` | ROC-AUC, Precision-Recall, Confusion Matrix evaluation pipeline, operational threshold tuning, and interactive Streamlit live demo (`evaluate.py`, `app.py`) |

---

## End-to-End 7-Stage Architecture

1. **`generate_datasets.py` & `ingestion.py` (Data Storage):** Generates/ingests monthly partitioned U.S. BTS flight records, hourly NOAA surface weather observations across 20 major continental U.S. hubs, and airport/airline lookup registries into PostgreSQL (`DATABASE_URL`) or indexed local SQLite storage (`data/db/aviation_warehouse.db`) with composite B-tree indexes on `(ORIGIN, CRS_DEP_TIMESTAMP)` and `(STATION_IATA, OBS_TIMESTAMP)`.
2. **`spark_clean_join.py` (Cleaning & Weather Alignment):** Drops malformed/NULL records, filters out cancelled (`CANCELLED == 1`) and diverted (`DIVERTED == 1`) flights, aligns scheduled departures to hourly NOAA weather readings, and exports the PySpark SQL & Catalyst query execution plan (`outputs/plans/pyspark_execution_plan.txt`).
3. **`feature_engineering.py` (Feature Extraction):** Encodes 24-hour circular departure geometry (`dep_hour_sin`, `dep_hour_cos`), day-of-week cyclical encodings, dew-point fog spread, rolling 3-hour origin hub departure volume (`origin_rolling_3h_vol`), runway congestion ratio, and leak-free chronological carrier/airport baselines.
4. **`balance_classes.py` (Chronological Split & SMOTE):** Performs a strict 80/20 chronological split to prevent temporal data leakage and applies SMOTE synthetic oversampling exclusively on the training partition while keeping the 20% test split untouched.
5. **`train_models.py` (Model Training):** Trains and cross-validates Logistic Regression, Random Forest (100 Trees), Gradient Boosting, and Tuned XGBoost classifiers (`DEP_DEL15` $\ge$ 15 min), plus a secondary continuous XGBoost Regressor estimating delay duration in minutes.
6. **`evaluate.py` (Evaluation & Threshold Tuning):** Computes ROC-AUC, PR-AUC, Precision, Recall, F1, and F2 scores on the untouched test split, sweeps operational classification thresholds, and exports plots to `outputs/plots/`.
7. **`app.py` (Interactive Live Serving & BI Console):** Launches the Streamlit Airline Operations Control Center (AOCC) & Business Intelligence demonstration suite featuring:
   - **1-Click Live Presentation Case Studies:** Five pre-configured aviation scenarios (*Nominal Morning VFR*, *>25 kt Crosswind Cliff*, *Winter Blizzard & De-icing*, *SFO Marine Fog*, and *Clear-Sky Evening Hub Cascade*) + **Guided Viva Walkthrough Mode**.
   - **Explainable AI (XAI) & What-If Labs:** Telescoping probability attribution waterfall, 24-hour *"Best Time to Fly"* schedule sweep optimizer, and live non-linear wind sensitivity comparison (*XGBoost vs. Random Forest vs. Logistic Regression*).
   - **Business Intelligence KPIs:** Real-time direct block-delay cost estimation (`$101.18/min` A4A benchmark), connecting passenger misconnect exposure, and T-2h proactive dispatch savings.
   - **National Hub-and-Spoke Radar & Custom Batch CSV Scorer:** Interactive 19-spoke U.S. risk map plus custom CSV schedule upload & scoring.
   - **Live Relational SQL Warehouse Playground:** Editable read-only SQL lab joining `bts_flights`, `noaa_weather_hourly`, `airports_metadata`, and `airlines_metadata` directly in the browser.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete 6-stage Big Data & ML pipeline
python run_pipeline.py

# 3. Launch the interactive Streamlit live dashboard
streamlit run app.py
```

To connect Stage 1 & Stage 2 to an external PostgreSQL cluster instead of the embedded warehouse:
```powershell
$env:DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/aviation"
python run_pipeline.py
```
