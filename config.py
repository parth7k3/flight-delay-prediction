"""
Central configuration and domain metadata for the Flight Delay Prediction Pipeline.
Course: Big Data Analytics and Business Intelligence (CBCA 293) — Bennett University
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLANS_DIR = OUTPUTS_DIR / "plans"
SQL_DIR = BASE_DIR / "sql"

for _d in [RAW_DIR, PROCESSED_DIR, DB_DIR, MODELS_DIR, PLOTS_DIR, PLANS_DIR, SQL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Supports PostgreSQL cluster URI via DATABASE_URL env var (e.g. postgresql+psycopg2://user:pass@host:5432/aviation)
# Defaults to local indexed SQLite analytical warehouse for zero-config standalone execution.
DEFAULT_SQLITE_URL = f"sqlite:///{(DB_DIR / 'aviation_warehouse.db').as_posix()}"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)

RANDOM_SEED = 42
DELAY_THRESHOLD_MIN = 15       # FAA official 15-minute departure delay threshold
CHRONOLOGICAL_TRAIN_RATIO = 0.80

# 20 Major Continental U.S. Commercial Hub Airports (Tertiary Registry Metadata)
AIRPORTS_REGISTRY = [
    {"IATA": "ATL", "NAME": "Hartsfield-Jackson Atlanta Intl", "CITY": "Atlanta", "STATE": "GA", "LAT": 33.6407, "LON": -84.4277, "CAPACITY_HR": 42, "CLIMATE": "humid_subtropical"},
    {"IATA": "ORD", "NAME": "Chicago O'Hare Intl", "CITY": "Chicago", "STATE": "IL", "LAT": 41.9742, "LON": -87.9073, "CAPACITY_HR": 38, "CLIMATE": "continental_winter"},
    {"IATA": "DFW", "NAME": "Dallas/Fort Worth Intl", "CITY": "Dallas", "STATE": "TX", "LAT": 32.8998, "LON": -97.0403, "CAPACITY_HR": 39, "CLIMATE": "convective_plains"},
    {"IATA": "DEN", "NAME": "Denver International", "CITY": "Denver", "STATE": "CO", "LAT": 39.8561, "LON": -104.6737, "CAPACITY_HR": 36, "CLIMATE": "alpine_winter"},
    {"IATA": "LAX", "NAME": "Los Angeles International", "CITY": "Los Angeles", "STATE": "CA", "LAT": 33.9416, "LON": -118.4085, "CAPACITY_HR": 35, "CLIMATE": "coastal_marine"},
    {"IATA": "JFK", "NAME": "John F. Kennedy Intl", "CITY": "New York", "STATE": "NY", "LAT": 40.6413, "LON": -73.7781, "CAPACITY_HR": 28, "CLIMATE": "northeast_coastal"},
    {"IATA": "EWR", "NAME": "Newark Liberty Intl", "CITY": "Newark", "STATE": "NJ", "LAT": 40.6895, "LON": -74.1745, "CAPACITY_HR": 26, "CLIMATE": "northeast_coastal"},
    {"IATA": "SFO", "NAME": "San Francisco Intl", "CITY": "San Francisco", "STATE": "CA", "LAT": 37.6213, "LON": -122.3790, "CAPACITY_HR": 27, "CLIMATE": "coastal_fog"},
    {"IATA": "SEA", "NAME": "Seattle-Tacoma Intl", "CITY": "Seattle", "STATE": "WA", "LAT": 47.4502, "LON": -122.3088, "CAPACITY_HR": 28, "CLIMATE": "pacific_rain"},
    {"IATA": "MIA", "NAME": "Miami International", "CITY": "Miami", "STATE": "FL", "LAT": 25.7959, "LON": -80.2870, "CAPACITY_HR": 30, "CLIMATE": "tropical"},
    {"IATA": "MCO", "NAME": "Orlando International", "CITY": "Orlando", "STATE": "FL", "LAT": 28.4312, "LON": -81.3081, "CAPACITY_HR": 30, "CLIMATE": "tropical"},
    {"IATA": "BOS", "NAME": "Boston Logan Intl", "CITY": "Boston", "STATE": "MA", "LAT": 42.3656, "LON": -71.0096, "CAPACITY_HR": 27, "CLIMATE": "northeast_coastal"},
    {"IATA": "CLT", "NAME": "Charlotte Douglas Intl", "CITY": "Charlotte", "STATE": "NC", "LAT": 35.2144, "LON": -80.9473, "CAPACITY_HR": 33, "CLIMATE": "humid_subtropical"},
    {"IATA": "LAS", "NAME": "Harry Reid International", "CITY": "Las Vegas", "STATE": "NV", "LAT": 36.0840, "LON": -115.1537, "CAPACITY_HR": 31, "CLIMATE": "arid_desert"},
    {"IATA": "PHX", "NAME": "Phoenix Sky Harbor Intl", "CITY": "Phoenix", "STATE": "AZ", "LAT": 33.4373, "LON": -112.0078, "CAPACITY_HR": 32, "CLIMATE": "arid_desert"},
    {"IATA": "IAH", "NAME": "George Bush Intercontinental", "CITY": "Houston", "STATE": "TX", "LAT": 29.9902, "LON": -95.3368, "CAPACITY_HR": 32, "CLIMATE": "humid_subtropical"},
    {"IATA": "MSP", "NAME": "Minneapolis-Saint Paul Intl", "CITY": "Minneapolis", "STATE": "MN", "LAT": 44.8848, "LON": -93.2223, "CAPACITY_HR": 29, "CLIMATE": "continental_winter"},
    {"IATA": "DTW", "NAME": "Detroit Metropolitan", "CITY": "Detroit", "STATE": "MI", "LAT": 42.2162, "LON": -83.3554, "CAPACITY_HR": 29, "CLIMATE": "continental_winter"},
    {"IATA": "LGA", "NAME": "LaGuardia Airport", "CITY": "New York", "STATE": "NY", "LAT": 40.7769, "LON": -73.8740, "CAPACITY_HR": 25, "CLIMATE": "northeast_coastal"},
    {"IATA": "SLC", "NAME": "Salt Lake City Intl", "CITY": "Salt Lake City", "STATE": "UT", "LAT": 40.7899, "LON": -111.9791, "CAPACITY_HR": 28, "CLIMATE": "alpine_winter"},
]

AIRLINES_REGISTRY = [
    {"CARRIER_CODE": "DL", "AIRLINE_NAME": "Delta Air Lines", "OTP_FACTOR": -0.22, "FLEET_TYPE": "Mainline Hub"},
    {"CARRIER_CODE": "AS", "AIRLINE_NAME": "Alaska Airlines", "OTP_FACTOR": -0.16, "FLEET_TYPE": "Mainline West"},
    {"CARRIER_CODE": "UA", "AIRLINE_NAME": "United Airlines", "OTP_FACTOR": -0.04, "FLEET_TYPE": "Mainline Hub"},
    {"CARRIER_CODE": "AA", "AIRLINE_NAME": "American Airlines", "OTP_FACTOR": 0.05, "FLEET_TYPE": "Mainline Hub"},
    {"CARRIER_CODE": "WN", "AIRLINE_NAME": "Southwest Airlines", "OTP_FACTOR": 0.08, "FLEET_TYPE": "Point-to-Point"},
    {"CARRIER_CODE": "B6", "AIRLINE_NAME": "JetBlue Airways", "OTP_FACTOR": 0.18, "FLEET_TYPE": "Northeast Focus"},
    {"CARRIER_CODE": "NK", "AIRLINE_NAME": "Spirit Airlines", "OTP_FACTOR": 0.22, "FLEET_TYPE": "Ultra Low-Cost"},
    {"CARRIER_CODE": "F9", "AIRLINE_NAME": "Frontier Airlines", "OTP_FACTOR": 0.26, "FLEET_TYPE": "Ultra Low-Cost"},
]

MODEL_FEATURE_COLUMNS = [
    "DISTANCE",
    "dep_hour_sin",
    "dep_hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "TEMP_F",
    "DEW_POINT_F",
    "temp_dew_spread",
    "WIND_SPEED_KT",
    "VISIBILITY_MI",
    "PRECIP_IN",
    "PRESSURE_MB",
    "origin_rolling_3h_vol",
    "origin_congestion_ratio",
    "carrier_hist_delay_rate",
    "origin_hist_delay_rate",
    "dest_hist_delay_rate",
]
