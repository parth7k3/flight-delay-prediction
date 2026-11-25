"""
Central configuration and real U.S. aviation metadata for the Big Data Flight Delay Pipeline.
Course: Big Data Analytics and Business Intelligence (CBCA 293) — Bennett University
Dataset Scale: Full-Year 2024 U.S. DOT BTS On-Time Performance (~7.0M flights, 12 monthly partitions)
               + Full-Year 2024 NOAA/NWS ASOS Hourly Surface Weather (~525K observations across 60 U.S. hubs)
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

DUCKDB_PATH = DB_DIR / "aviation_warehouse.duckdb"
DATABASE_URL = os.environ.get("DATABASE_URL", f"duckdb:///{DUCKDB_PATH.as_posix()}")

RANDOM_SEED = 42
DELAY_THRESHOLD_MIN = 15       # FAA official 15-minute departure delay threshold
CHRONOLOGICAL_TRAIN_RATIO = 0.80
SMOTE_MAX_TRAIN_SAMPLE = 300000  # Stratified chronological training sample for tractable k-NN SMOTE

# Top 60 U.S. Commercial Airports (covers ~88% of all U.S. domestic commercial departures)
AIRPORTS_REGISTRY = [
    {"IATA": "ATL", "NAME": "Hartsfield-Jackson Atlanta Intl", "CITY": "Atlanta", "STATE": "GA", "LAT": 33.6407, "LON": -84.4277, "TZ": "America/New_York", "CAPACITY_HR": 110},
    {"IATA": "DFW", "NAME": "Dallas/Fort Worth Intl", "CITY": "Dallas", "STATE": "TX", "LAT": 32.8998, "LON": -97.0403, "TZ": "America/Chicago", "CAPACITY_HR": 98},
    {"IATA": "DEN", "NAME": "Denver International", "CITY": "Denver", "STATE": "CO", "LAT": 39.8561, "LON": -104.6737, "TZ": "America/Denver", "CAPACITY_HR": 95},
    {"IATA": "ORD", "NAME": "Chicago O'Hare Intl", "CITY": "Chicago", "STATE": "IL", "LAT": 41.9742, "LON": -87.9073, "TZ": "America/Chicago", "CAPACITY_HR": 96},
    {"IATA": "LAX", "NAME": "Los Angeles International", "CITY": "Los Angeles", "STATE": "CA", "LAT": 33.9416, "LON": -118.4085, "TZ": "America/Los_Angeles", "CAPACITY_HR": 82},
    {"IATA": "CLT", "NAME": "Charlotte Douglas Intl", "CITY": "Charlotte", "STATE": "NC", "LAT": 35.2144, "LON": -80.9473, "TZ": "America/New_York", "CAPACITY_HR": 78},
    {"IATA": "MCO", "NAME": "Orlando International", "CITY": "Orlando", "STATE": "FL", "LAT": 28.4312, "LON": -81.3081, "TZ": "America/New_York", "CAPACITY_HR": 68},
    {"IATA": "LAS", "NAME": "Harry Reid International", "CITY": "Las Vegas", "STATE": "NV", "LAT": 36.0840, "LON": -115.1537, "TZ": "America/Los_Angeles", "CAPACITY_HR": 66},
    {"IATA": "PHX", "NAME": "Phoenix Sky Harbor Intl", "CITY": "Phoenix", "STATE": "AZ", "LAT": 33.4373, "LON": -112.0078, "TZ": "America/Phoenix", "CAPACITY_HR": 68},
    {"IATA": "MIA", "NAME": "Miami International", "CITY": "Miami", "STATE": "FL", "LAT": 25.7959, "LON": -80.2870, "TZ": "America/New_York", "CAPACITY_HR": 62},
    {"IATA": "SEA", "NAME": "Seattle-Tacoma Intl", "CITY": "Seattle", "STATE": "WA", "LAT": 47.4502, "LON": -122.3088, "TZ": "America/Los_Angeles", "CAPACITY_HR": 60},
    {"IATA": "IAH", "NAME": "George Bush Intercontinental", "CITY": "Houston", "STATE": "TX", "LAT": 29.9902, "LON": -95.3368, "TZ": "America/Chicago", "CAPACITY_HR": 66},
    {"IATA": "EWR", "NAME": "Newark Liberty Intl", "CITY": "Newark", "STATE": "NJ", "LAT": 40.6895, "LON": -74.1745, "TZ": "America/New_York", "CAPACITY_HR": 52},
    {"IATA": "SFO", "NAME": "San Francisco Intl", "CITY": "San Francisco", "STATE": "CA", "LAT": 37.6213, "LON": -122.3790, "TZ": "America/Los_Angeles", "CAPACITY_HR": 54},
    {"IATA": "BOS", "NAME": "Boston Logan Intl", "CITY": "Boston", "STATE": "MA", "LAT": 42.3656, "LON": -71.0096, "TZ": "America/New_York", "CAPACITY_HR": 56},
    {"IATA": "MSP", "NAME": "Minneapolis-Saint Paul Intl", "CITY": "Minneapolis", "STATE": "MN", "LAT": 44.8848, "LON": -93.2223, "TZ": "America/Chicago", "CAPACITY_HR": 58},
    {"IATA": "FLL", "NAME": "Fort Lauderdale-Hollywood Intl", "CITY": "Fort Lauderdale", "STATE": "FL", "LAT": 26.0726, "LON": -80.1527, "TZ": "America/New_York", "CAPACITY_HR": 48},
    {"IATA": "LGA", "NAME": "LaGuardia Airport", "CITY": "New York", "STATE": "NY", "LAT": 40.7769, "LON": -73.8740, "TZ": "America/New_York", "CAPACITY_HR": 54},
    {"IATA": "DTW", "NAME": "Detroit Metropolitan", "CITY": "Detroit", "STATE": "MI", "LAT": 42.2162, "LON": -83.3554, "TZ": "America/Detroit", "CAPACITY_HR": 58},
    {"IATA": "PHL", "NAME": "Philadelphia International", "CITY": "Philadelphia", "STATE": "PA", "LAT": 39.8744, "LON": -75.2424, "TZ": "America/New_York", "CAPACITY_HR": 52},
    {"IATA": "SLC", "NAME": "Salt Lake City Intl", "CITY": "Salt Lake City", "STATE": "UT", "LAT": 40.7899, "LON": -111.9791, "TZ": "America/Denver", "CAPACITY_HR": 52},
    {"IATA": "BWI", "NAME": "Baltimore/Washington Intl", "CITY": "Baltimore", "STATE": "MD", "LAT": 39.1774, "LON": -76.6684, "TZ": "America/New_York", "CAPACITY_HR": 48},
    {"IATA": "DCA", "NAME": "Ronald Reagan Washington National", "CITY": "Washington", "STATE": "VA", "LAT": 38.8512, "LON": -77.0402, "TZ": "America/New_York", "CAPACITY_HR": 50},
    {"IATA": "SAN", "NAME": "San Diego International", "CITY": "San Diego", "STATE": "CA", "LAT": 32.7338, "LON": -117.1933, "TZ": "America/Los_Angeles", "CAPACITY_HR": 42},
    {"IATA": "IAD", "NAME": "Washington Dulles Intl", "CITY": "Dulles", "STATE": "VA", "LAT": 38.9531, "LON": -77.4565, "TZ": "America/New_York", "CAPACITY_HR": 50},
    {"IATA": "TPA", "NAME": "Tampa International", "CITY": "Tampa", "STATE": "FL", "LAT": 27.9755, "LON": -82.5332, "TZ": "America/New_York", "CAPACITY_HR": 44},
    {"IATA": "BNA", "NAME": "Nashville International", "CITY": "Nashville", "STATE": "TN", "LAT": 36.1263, "LON": -86.6774, "TZ": "America/Chicago", "CAPACITY_HR": 46},
    {"IATA": "AUS", "NAME": "Austin-Bergstrom Intl", "CITY": "Austin", "STATE": "TX", "LAT": 30.1975, "LON": -97.6664, "TZ": "America/Chicago", "CAPACITY_HR": 44},
    {"IATA": "MDW", "NAME": "Chicago Midway Intl", "CITY": "Chicago", "STATE": "IL", "LAT": 41.7868, "LON": -87.7522, "TZ": "America/Chicago", "CAPACITY_HR": 42},
    {"IATA": "JFK", "NAME": "John F. Kennedy Intl", "CITY": "New York", "STATE": "NY", "LAT": 40.6413, "LON": -73.7781, "TZ": "America/New_York", "CAPACITY_HR": 56},
    {"IATA": "DAL", "NAME": "Dallas Love Field", "CITY": "Dallas", "STATE": "TX", "LAT": 32.8471, "LON": -96.8518, "TZ": "America/Chicago", "CAPACITY_HR": 38},
    {"IATA": "PDX", "NAME": "Portland International", "CITY": "Portland", "STATE": "OR", "LAT": 45.5898, "LON": -122.5951, "TZ": "America/Los_Angeles", "CAPACITY_HR": 38},
    {"IATA": "STL", "NAME": "St. Louis Lambert Intl", "CITY": "St. Louis", "STATE": "MO", "LAT": 38.7487, "LON": -90.3700, "TZ": "America/Chicago", "CAPACITY_HR": 38},
    {"IATA": "RDU", "NAME": "Raleigh-Durham Intl", "CITY": "Raleigh", "STATE": "NC", "LAT": 35.8801, "LON": -78.7880, "TZ": "America/New_York", "CAPACITY_HR": 36},
    {"IATA": "HOU", "NAME": "William P. Hobby Airport", "CITY": "Houston", "STATE": "TX", "LAT": 29.6454, "LON": -95.2789, "TZ": "America/Chicago", "CAPACITY_HR": 36},
    {"IATA": "SMF", "NAME": "Sacramento International", "CITY": "Sacramento", "STATE": "CA", "LAT": 38.6954, "LON": -121.5908, "TZ": "America/Los_Angeles", "CAPACITY_HR": 34},
    {"IATA": "MSY", "NAME": "Louis Armstrong New Orleans Intl", "CITY": "New Orleans", "STATE": "LA", "LAT": 29.9934, "LON": -90.2580, "TZ": "America/Chicago", "CAPACITY_HR": 34},
    {"IATA": "SJC", "NAME": "San Jose Mineta Intl", "CITY": "San Jose", "STATE": "CA", "LAT": 37.3639, "LON": -121.9289, "TZ": "America/Los_Angeles", "CAPACITY_HR": 34},
    {"IATA": "SNA", "NAME": "John Wayne Airport", "CITY": "Santa Ana", "STATE": "CA", "LAT": 33.6762, "LON": -117.8675, "TZ": "America/Los_Angeles", "CAPACITY_HR": 32},
    {"IATA": "MCI", "NAME": "Kansas City International", "CITY": "Kansas City", "STATE": "MO", "LAT": 39.2976, "LON": -94.7139, "TZ": "America/Chicago", "CAPACITY_HR": 34},
    {"IATA": "OAK", "NAME": "Oakland International", "CITY": "Oakland", "STATE": "CA", "LAT": 37.7126, "LON": -122.2197, "TZ": "America/Los_Angeles", "CAPACITY_HR": 32},
    {"IATA": "SAT", "NAME": "San Antonio International", "CITY": "San Antonio", "STATE": "TX", "LAT": 29.5337, "LON": -98.4698, "TZ": "America/Chicago", "CAPACITY_HR": 32},
    {"IATA": "RSW", "NAME": "Southwest Florida Intl", "CITY": "Fort Myers", "STATE": "FL", "LAT": 26.5362, "LON": -81.7552, "TZ": "America/New_York", "CAPACITY_HR": 32},
    {"IATA": "CLE", "NAME": "Cleveland Hopkins Intl", "CITY": "Cleveland", "STATE": "OH", "LAT": 41.4117, "LON": -81.8498, "TZ": "America/New_York", "CAPACITY_HR": 32},
    {"IATA": "IND", "NAME": "Indianapolis International", "CITY": "Indianapolis", "STATE": "IN", "LAT": 39.7173, "LON": -86.2944, "TZ": "America/Indiana/Indianapolis", "CAPACITY_HR": 32},
    {"IATA": "PIT", "NAME": "Pittsburgh International", "CITY": "Pittsburgh", "STATE": "PA", "LAT": 40.4915, "LON": -80.2329, "TZ": "America/New_York", "CAPACITY_HR": 32},
    {"IATA": "CVG", "NAME": "Cincinnati/Northern Kentucky Intl", "CITY": "Cincinnati", "STATE": "KY", "LAT": 39.0488, "LON": -84.6678, "TZ": "America/New_York", "CAPACITY_HR": 32},
    {"IATA": "CMH", "NAME": "John Glenn Columbus Intl", "CITY": "Columbus", "STATE": "OH", "LAT": 39.9980, "LON": -82.8919, "TZ": "America/New_York", "CAPACITY_HR": 32},
    {"IATA": "PBI", "NAME": "Palm Beach International", "CITY": "West Palm Beach", "STATE": "FL", "LAT": 26.6832, "LON": -80.0956, "TZ": "America/New_York", "CAPACITY_HR": 30},
    {"IATA": "JAX", "NAME": "Jacksonville International", "CITY": "Jacksonville", "STATE": "FL", "LAT": 30.4941, "LON": -81.6879, "TZ": "America/New_York", "CAPACITY_HR": 28},
    {"IATA": "ONT", "NAME": "Ontario International", "CITY": "Ontario", "STATE": "CA", "LAT": 34.0560, "LON": -117.6012, "TZ": "America/Los_Angeles", "CAPACITY_HR": 28},
    {"IATA": "BUR", "NAME": "Hollywood Burbank Airport", "CITY": "Burbank", "STATE": "CA", "LAT": 34.2007, "LON": -118.3585, "TZ": "America/Los_Angeles", "CAPACITY_HR": 26},
    {"IATA": "BDL", "NAME": "Bradley International", "CITY": "Hartford", "STATE": "CT", "LAT": 41.9389, "LON": -72.6832, "TZ": "America/New_York", "CAPACITY_HR": 28},
    {"IATA": "BUF", "NAME": "Buffalo Niagara Intl", "CITY": "Buffalo", "STATE": "NY", "LAT": 42.9405, "LON": -78.7322, "TZ": "America/New_York", "CAPACITY_HR": 26},
    {"IATA": "MKE", "NAME": "Milwaukee Mitchell Intl", "CITY": "Milwaukee", "STATE": "WI", "LAT": 42.9472, "LON": -87.8966, "TZ": "America/Chicago", "CAPACITY_HR": 28},
    {"IATA": "ABQ", "NAME": "Albuquerque International Sunport", "CITY": "Albuquerque", "STATE": "NM", "LAT": 35.0402, "LON": -106.6090, "TZ": "America/Denver", "CAPACITY_HR": 26},
    {"IATA": "OMA", "NAME": "Eppley Airfield", "CITY": "Omaha", "STATE": "NE", "LAT": 41.3032, "LON": -95.8941, "TZ": "America/Chicago", "CAPACITY_HR": 24},
    {"IATA": "MEM", "NAME": "Memphis International", "CITY": "Memphis", "STATE": "TN", "LAT": 35.0424, "LON": -89.9767, "TZ": "America/Chicago", "CAPACITY_HR": 28},
    {"IATA": "RIC", "NAME": "Richmond International", "CITY": "Richmond", "STATE": "VA", "LAT": 37.5052, "LON": -77.3197, "TZ": "America/New_York", "CAPACITY_HR": 24},
    {"IATA": "BOI", "NAME": "Boise Airport", "CITY": "Boise", "STATE": "ID", "LAT": 43.5644, "LON": -116.2228, "TZ": "America/Boise", "CAPACITY_HR": 26},
]

# Official U.S. DOT BTS Reporting Carriers (Full 2024 Fleet)
AIRLINES_REGISTRY = [
    {"CARRIER_CODE": "AA", "AIRLINE_NAME": "American Airlines", "FLEET_TYPE": "Mainline Network Hub"},
    {"CARRIER_CODE": "DL", "AIRLINE_NAME": "Delta Air Lines", "FLEET_TYPE": "Mainline Network Hub"},
    {"CARRIER_CODE": "UA", "AIRLINE_NAME": "United Airlines", "FLEET_TYPE": "Mainline Network Hub"},
    {"CARRIER_CODE": "WN", "AIRLINE_NAME": "Southwest Airlines", "FLEET_TYPE": "Point-to-Point National"},
    {"CARRIER_CODE": "AS", "AIRLINE_NAME": "Alaska Airlines", "FLEET_TYPE": "Mainline West Coast"},
    {"CARRIER_CODE": "B6", "AIRLINE_NAME": "JetBlue Airways", "FLEET_TYPE": "Northeast / East Focus"},
    {"CARRIER_CODE": "NK", "AIRLINE_NAME": "Spirit Airlines", "FLEET_TYPE": "Ultra Low-Cost"},
    {"CARRIER_CODE": "F9", "AIRLINE_NAME": "Frontier Airlines", "FLEET_TYPE": "Ultra Low-Cost"},
    {"CARRIER_CODE": "G4", "AIRLINE_NAME": "Allegiant Air", "FLEET_TYPE": "Leisure Low-Cost"},
    {"CARRIER_CODE": "HA", "AIRLINE_NAME": "Hawaiian Airlines", "FLEET_TYPE": "Pacific / Mainline"},
    {"CARRIER_CODE": "OO", "AIRLINE_NAME": "SkyWest Airlines", "FLEET_TYPE": "Regional Partner"},
    {"CARRIER_CODE": "YX", "AIRLINE_NAME": "Republic Airways", "FLEET_TYPE": "Regional Partner"},
    {"CARRIER_CODE": "MQ", "AIRLINE_NAME": "Envoy Air", "FLEET_TYPE": "Regional Partner"},
    {"CARRIER_CODE": "OH", "AIRLINE_NAME": "PSA Airlines", "FLEET_TYPE": "Regional Partner"},
    {"CARRIER_CODE": "9E", "AIRLINE_NAME": "Endeavor Air", "FLEET_TYPE": "Regional Partner"},
]

MODEL_FEATURE_COLUMNS = [
    "DISTANCE",
    "CRS_ELAPSED_TIME",
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
    "wind_max_3h",
    "precip_sum_3h",
    "vis_min_3h",
    "origin_rolling_3h_vol",
    "origin_congestion_ratio",
    "prev_aircraft_delay_t2h",
    "has_inbound_leg",
    "origin_t2h_delay_rate",
    "dest_t2h_delay_rate",
    "carrier_hist_delay_rate",
    "origin_hist_delay_rate",
    "dest_hist_delay_rate",
]
