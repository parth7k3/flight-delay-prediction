-- ============================================================================
-- Distributed / Partitioned PostgreSQL Storage Schema
-- Project: Predicting Flight Delays: A Big Data and Machine Learning Approach
-- Course: CBCA 293 (Bennett University) | Module Owner: Parth (S24CSEU2470)
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS aviation;

-- 1. Tertiary Tier: Airport & Airline Metadata Registries
CREATE TABLE IF NOT EXISTS aviation.airports_metadata (
    iata            CHAR(3) PRIMARY KEY,
    name            VARCHAR(120) NOT NULL,
    city            VARCHAR(80)  NOT NULL,
    state           CHAR(2)      NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    capacity_hr     INTEGER NOT NULL,
    climate         VARCHAR(40)
);

CREATE TABLE IF NOT EXISTS aviation.airlines_metadata (
    carrier_code    VARCHAR(4) PRIMARY KEY,
    airline_name    VARCHAR(100) NOT NULL,
    fleet_type      VARCHAR(50)
);

-- 2. Secondary Tier: NOAA Hourly Surface Weather Observations
CREATE TABLE IF NOT EXISTS aviation.noaa_weather_hourly (
    station_iata    CHAR(3) NOT NULL REFERENCES aviation.airports_metadata(iata),
    obs_timestamp   TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    temp_f          DOUBLE PRECISION,
    dew_point_f     DOUBLE PRECISION,
    wind_speed_kt   DOUBLE PRECISION,
    visibility_mi   DOUBLE PRECISION,
    precip_in       DOUBLE PRECISION,
    pressure_mb     DOUBLE PRECISION,
    weather_cond    VARCHAR(32),
    PRIMARY KEY (station_iata, obs_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_weather_station_ts
    ON aviation.noaa_weather_hourly (station_iata, obs_timestamp);

-- 3. Primary Tier: Partitioned U.S. BTS On-Time Performance Flight Logs
CREATE TABLE IF NOT EXISTS aviation.bts_flights (
    flight_id           BIGSERIAL,
    fl_date             DATE NOT NULL,
    op_unique_carrier   VARCHAR(4) NOT NULL,
    tail_num            VARCHAR(12),
    op_carrier_fl_num   INTEGER,
    origin              CHAR(3) NOT NULL,
    dest                CHAR(3) NOT NULL,
    crs_dep_time        INTEGER NOT NULL,
    crs_dep_timestamp   TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    dep_time            INTEGER,
    dep_delay           DOUBLE PRECISION,
    dep_del15           SMALLINT,
    cancelled           SMALLINT DEFAULT 0,
    cancellation_code   CHAR(1),
    diverted            SMALLINT DEFAULT 0,
    distance            DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (flight_id, fl_date)
) PARTITION BY RANGE (fl_date);

CREATE TABLE IF NOT EXISTS aviation.bts_flights_2025_01
    PARTITION OF aviation.bts_flights
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE IF NOT EXISTS aviation.bts_flights_2025_02
    PARTITION OF aviation.bts_flights
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE IF NOT EXISTS aviation.bts_flights_2025_03
    PARTITION OF aviation.bts_flights
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Composite index on airport IATA code and scheduled departure timestamp
CREATE INDEX IF NOT EXISTS idx_flights_origin_dep_ts
    ON aviation.bts_flights (origin, crs_dep_timestamp);

CREATE INDEX IF NOT EXISTS idx_flights_chronological
    ON aviation.bts_flights (crs_dep_timestamp);
