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
7. **`app.py` (Interactive Live Serving):** Launches the Streamlit Airline Operations Control Center (AOCC) dashboard with single-flight live delay prediction, live Open-Meteo airport weather fetching, 20-flight hub wave batch scanning, interactive threshold tuning, and pipeline inspection.

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
