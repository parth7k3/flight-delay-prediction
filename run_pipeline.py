"""
Master End-to-End Big Data & Machine Learning Pipeline Orchestrator (run_pipeline.py)
Executes Stages 1 through 6 sequentially and exports all database tables, trained models,
execution plans, and evaluation artifacts required by the Streamlit dashboard (app.py).
"""

from __future__ import annotations
import time
from ingestion import run_ingestion
from spark_clean_join import run_spark_clean_join
from feature_engineering import run_feature_engineering
from balance_classes import run_balance_classes
from train_models import run_train_models
from evaluate import run_evaluation


def main(force_regenerate: bool = False) -> None:
    t_start = time.perf_counter()
    print("=" * 78)
    print(" PREDICTING FLIGHT DELAYS: A BIG DATA & MACHINE LEARNING PIPELINE (CBCA 293)")
    print("=" * 78)

    print("\n[Stage 1/6] Raw CSV Partition Ingestion & Database Indexing...")
    run_ingestion(force_regenerate=force_regenerate)

    print("\n[Stage 2/6] PySpark Data Cleaning, Cancellation Filtering & NOAA Weather Join...")
    run_spark_clean_join()

    print("\n[Stage 3/6] Cyclical Time Encoding & Rolling Hub Congestion Engineering...")
    run_feature_engineering()

    print("\n[Stage 4/6] 80/20 Chronological Split & Training-Partition SMOTE Oversampling...")
    run_balance_classes()

    print("\n[Stage 5/6] Multi-Model Training & Fold-Internal Cross-Validation...")
    run_train_models()

    print("\n[Stage 6/6] Untouched Test Evaluation, Threshold Tuning & Plot Export...")
    run_evaluation()

    total_sec = round(time.perf_counter() - t_start, 2)
    print("\n" + "=" * 78)
    print(f" PIPELINE COMPLETE IN {total_sec}s — Ready to launch live demo: streamlit run app.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
