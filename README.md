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

## End-to-End 7-Stage Big Data Architecture (~7.0M Real Flight Records)

1. **`download_real_bigdata.py` & `ingestion.py` (Multi-Million-Row Data Storage):** Acquires all 12 monthly partitions of **2024 U.S. DOT Bureau of Transportation Statistics (BTS) On-Time Performance Logs** (~7.0M real domestic flights), **Full-Year 2024 NOAA/NWS ASOS Hourly Surface Weather Observations** across 60 major U.S. hubs (~525K METAR readings aligned to local airport timezones), and airport/airline metadata registries. Streams partitions out-of-core into a columnar **DuckDB / PostgreSQL-compatible analytical warehouse** (`data/db/aviation_warehouse.duckdb`) indexed on `(ORIGIN, CRS_DEP_TIMESTAMP)` and `(STATION_IATA, OBS_TIMESTAMP)`.
2. **`spark_clean_join.py` (Out-of-Core Cleaning & Weather Alignment):** Drops malformed/NULL records, filters out cancelled (`CANCELLED == 1`) and diverted (`DIVERTED == 1`) flights, computes 3-hour rolling storm persistence (`wind_max_3h`, `precip_sum_3h`, `vis_min_3h`), joins flights to local hourly NOAA surface weather, writes ZSTD-compressed Parquet, and exports [pyspark_execution_plan.txt](outputs/plans/pyspark_execution_plan.txt).
3. **`feature_engineering.py` (Window & Cyclical Feature Extraction):** Executes out-of-core SQL window functions and trigonometric encodings (`dep_hour_sin`, `dep_hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`), dew-point fog spread, rolling 3-hour origin hub departure volume (`origin_rolling_3h_vol`), runway congestion ratio, and leak-free chronological 80% carrier/airport delay priors.
4. **`balance_classes.py` (Chronological 80/20 Split & Training-Fold SMOTE):** Enforces a strict chronological 80/20 split across the full multi-million-row timeline (`Jan–Oct 2024` Train vs. `Oct–Dec 2024` Untouched Test = **1M+ real test flights**) and applies `SMOTE` synthetic oversampling exclusively on the training partition.
5. **`train_models.py` (Scalable Ensemble Training):** Trains cross-validated Logistic Regression, Random Forest (100 Trees), Histogram Gradient Boosting, and Tuned Histogram XGBoost (`tree_method='hist'`) classifiers (`DEP_DEL15` $\ge$ 15 min), plus a secondary continuous XGBoost Delay Regressor (minutes).
6. **`evaluate.py` (1M+ Row Evaluation & Threshold Tuning):** Scores all classifiers across the **entire 1,000,000+ row untouched chronological test partition**, calibrates operational recall-prioritized thresholds on a held-out training validation slice, and exports evaluation plots to `outputs/plots/`.
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

# 2. Download full-year 2024 U.S. DOT BTS (~7M flights) + NOAA ASOS weather (~525K rows)
python download_real_bigdata.py

# 3. Run the complete 6-stage Out-of-Core Big Data & ML pipeline
python run_pipeline.py

# 4. Launch the interactive Streamlit live dashboard
streamlit run app.py
```

To connect Stage 1 & Stage 2 to an external PostgreSQL cluster instead of the embedded warehouse:
```powershell
$env:DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/aviation"
python run_pipeline.py
```
