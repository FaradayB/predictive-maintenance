"""
=============================================================================
 db.py
 Vehicle Predictive Maintenance — PostgreSQL Database Layer
=============================================================================
 Handles all database operations:
   - Connection management (pooled, context-managed)
   - Schema creation + seeding from the test dataset Excel file
   - Sensor data queries by plate number (Track 1 + Track 2)
   - Writing ML predictions back to the database
   - Query logging for audit trail

 Tables:
   track1_technician   — 30-day telemetry records (320 rows from test set)
   track2_owner        — 12-hour daily readings  (120 rows from test set)
   query_log           — every prediction request made through the app

 Usage:
   from db import get_connection, get_track1_record, get_track2_record,
                  save_track1_prediction, save_track2_prediction,
                  seed_database, get_all_plates

 Required .env:
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=vehicle_maintenance
   DB_USER=vpm_user
   DB_PASSWORD=your_password_here
=============================================================================
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "vehicle_maintenance"),
    "user":     os.getenv("DB_USER",     "vpm_user"),
    "password": os.getenv("DB_PASSWORD", ""),
}

EXCEL_PATH = os.getenv("DATASET_PATH", "Vehicle_Sensor_TestSet.xlsx")


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """
    Context-managed PostgreSQL connection.
    Commits on success, rolls back on exception, always closes.

    Usage:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT ...")
                rows = cur.fetchall()
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        log.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SQL_CREATE_TRACK1 = """
CREATE TABLE IF NOT EXISTS track1_technician (
    test_id             VARCHAR(20) PRIMARY KEY,
    owner_name          VARCHAR(100) NOT NULL,
    plate_number        VARCHAR(20)  NOT NULL,
    car_model           VARCHAR(100) NOT NULL,
    car_year            SMALLINT     NOT NULL,
    day_in_window       SMALLINT     NOT NULL,

    -- Sensor readings
    o2_sensor_v         NUMERIC(6,4),
    maf_g_per_s         NUMERIC(6,3),
    throttle_pos_pct    NUMERIC(6,3),
    crank_rpm           SMALLINT,
    cam_advance_deg     NUMERIC(6,3),
    knock_count_30d     SMALLINT,
    coolant_temp_c      NUMERIC(7,3),
    oil_pressure_psi    NUMERIC(7,3),
    map_kpa             NUMERIC(7,3),
    egr_duty_pct        NUMERIC(6,3),
    battery_voltage_v   NUMERIC(6,3),
    fuel_temp_c         NUMERIC(6,3),

    -- Ground truth
    true_fault_class    SMALLINT     NOT NULL,
    true_fault_label    VARCHAR(50)  NOT NULL,

    -- ML predictions (filled by the app)
    predicted_class     SMALLINT,
    predicted_label     VARCHAR(50),
    predicted_at        TIMESTAMP
);
"""

SQL_CREATE_TRACK2 = """
CREATE TABLE IF NOT EXISTS track2_owner (
    test_id             VARCHAR(20) PRIMARY KEY,
    owner_name          VARCHAR(100) NOT NULL,
    plate_number        VARCHAR(20)  NOT NULL,
    car_model           VARCHAR(100) NOT NULL,
    car_year            SMALLINT     NOT NULL,
    hour_of_day         SMALLINT     NOT NULL,

    -- Sensor readings
    o2_sensor_v         NUMERIC(6,4),
    maf_g_per_s         NUMERIC(6,3),
    throttle_pos_pct    NUMERIC(6,3),
    coolant_temp_c      NUMERIC(7,3),
    oil_pressure_psi    NUMERIC(7,3),
    battery_voltage_v   NUMERIC(6,3),
    tpms_psi            NUMERIC(6,3),
    ambient_temp_c      NUMERIC(6,3),
    cabin_humidity_pct  NUMERIC(6,3),
    fuel_level_pct      NUMERIC(6,3),
    brake_pedal_events  SMALLINT,
    speed_kmh           NUMERIC(7,3),

    -- Ground truth
    true_risk_class     SMALLINT     NOT NULL,
    true_risk_label     VARCHAR(50)  NOT NULL,

    -- ML predictions (filled by the app)
    predicted_class     SMALLINT,
    predicted_label     VARCHAR(50),
    predicted_at        TIMESTAMP
);
"""

SQL_CREATE_QUERY_LOG = """
CREATE TABLE IF NOT EXISTS query_log (
    id                  SERIAL PRIMARY KEY,
    queried_at          TIMESTAMP DEFAULT NOW(),
    track               SMALLINT     NOT NULL,   -- 1 or 2
    plate_number        VARCHAR(20)  NOT NULL,
    test_id             VARCHAR(20),
    predicted_class     SMALLINT,
    predicted_label     VARCHAR(50),
    response_time_ms    INTEGER,
    context_chunks      SMALLINT,
    llm_called          BOOLEAN DEFAULT FALSE
);
"""

SQL_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_t1_plate ON track1_technician(plate_number);
CREATE INDEX IF NOT EXISTS idx_t2_plate ON track2_owner(plate_number);
CREATE INDEX IF NOT EXISTS idx_log_track ON query_log(track);
CREATE INDEX IF NOT EXISTS idx_log_queried_at ON query_log(queried_at);
"""


def create_schema() -> None:
    """Create all tables and indexes if they don't exist."""
    log.info("Creating database schema ...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_CREATE_TRACK1)
            cur.execute(SQL_CREATE_TRACK2)
            cur.execute(SQL_CREATE_QUERY_LOG)
            cur.execute(SQL_CREATE_INDEXES)
    log.info("Schema ready.")


# ─────────────────────────────────────────────────────────────────────────────
# SEEDING
# ─────────────────────────────────────────────────────────────────────────────

def seed_database(excel_path: str = EXCEL_PATH) -> None:
    """
    Load the test dataset Excel file and insert all rows into PostgreSQL.
    Skips rows that already exist (ON CONFLICT DO NOTHING).

    Args:
        excel_path: Path to Vehicle_Sensor_TestSet.xlsx
    """
    log.info(f"Seeding database from '{excel_path}' ...")

    xl = pd.ExcelFile(excel_path)
    t1 = xl.parse("TestA_Track1_Technician")
    t2 = xl.parse("TestB_Track2_Owner")

    with get_connection() as conn:
        with conn.cursor() as cur:

            # ── Track 1 ──────────────────────────────────────────────────────
            inserted_t1 = 0
            for _, row in t1.iterrows():
                cur.execute("""
                    INSERT INTO track1_technician (
                        test_id, owner_name, plate_number, car_model, car_year,
                        day_in_window,
                        o2_sensor_v, maf_g_per_s, throttle_pos_pct, crank_rpm,
                        cam_advance_deg, knock_count_30d, coolant_temp_c,
                        oil_pressure_psi, map_kpa, egr_duty_pct,
                        battery_voltage_v, fuel_temp_c,
                        true_fault_class, true_fault_label
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s
                    ) ON CONFLICT (test_id) DO NOTHING
                """, (
                    row["TEST ID"], row["OWNER NAME"], row["PLATE NUMBER"],
                    row["CAR MODEL"], int(row["CAR YEAR"]),
                    int(row["DAY IN WINDOW"]),
                    row["O2 SENSOR V"], row["MAF G PER S"],
                    row["THROTTLE POS PCT"], int(row["CRANK RPM"]),
                    row["CAM ADVANCE DEG"], int(row["KNOCK COUNT 30D"]),
                    row["COOLANT TEMP C"], row["OIL PRESSURE PSI"],
                    row["MAP KPA"], row["EGR DUTY PCT"],
                    row["BATTERY VOLTAGE V"], row["FUEL TEMP C"],
                    int(row["TRUE FAULT CLASS"]), row["TRUE FAULT LABEL"],
                ))
                inserted_t1 += cur.rowcount

            # ── Track 2 ──────────────────────────────────────────────────────
            inserted_t2 = 0
            for _, row in t2.iterrows():
                cur.execute("""
                    INSERT INTO track2_owner (
                        test_id, owner_name, plate_number, car_model, car_year,
                        hour_of_day,
                        o2_sensor_v, maf_g_per_s, throttle_pos_pct,
                        coolant_temp_c, oil_pressure_psi, battery_voltage_v,
                        tpms_psi, ambient_temp_c, cabin_humidity_pct,
                        fuel_level_pct, brake_pedal_events, speed_kmh,
                        true_risk_class, true_risk_label
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    ) ON CONFLICT (test_id) DO NOTHING
                """, (
                    row["TEST ID"], row["OWNER NAME"], row["PLATE NUMBER"],
                    row["CAR MODEL"], int(row["CAR YEAR"]),
                    int(row["HOUR OF DAY"]),
                    row["O2 SENSOR V"], row["MAF G PER S"],
                    row["THROTTLE POS PCT"],
                    row["COOLANT TEMP C"], row["OIL PRESSURE PSI"],
                    row["BATTERY VOLTAGE V"],
                    row["TPMS PSI"], row["AMBIENT TEMP C"],
                    row["CABIN HUMIDITY PCT"],
                    row["FUEL LEVEL PCT"], int(row["BRAKE PEDAL EVENTS"]),
                    row["SPEED KMH"],
                    int(row["TRUE RISK CLASS"]), row["TRUE RISK LABEL"],
                ))
                inserted_t2 += cur.rowcount

    log.info(f"Seeded: {inserted_t1} Track 1 rows, {inserted_t2} Track 2 rows.")


# ─────────────────────────────────────────────────────────────────────────────
# QUERIES — TRACK 1
# ─────────────────────────────────────────────────────────────────────────────

def get_all_plates(track: int = 1) -> List[str]:
    """Return sorted list of all unique plate numbers for a given track."""
    table = "track1_technician" if track == 1 else "track2_owner"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT plate_number FROM {table} ORDER BY plate_number"
            )
            return [row[0] for row in cur.fetchall()]


def get_track1_record(plate_number: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent Track 1 record for a given plate number.
    Returns the row with the highest day_in_window (most recent telemetry).

    Args:
        plate_number: Vehicle plate number (e.g. 'B 1234 ABC')

    Returns:
        Dict with all sensor readings and metadata, or None if not found.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM track1_technician
                WHERE plate_number = %s
                ORDER BY day_in_window DESC
                LIMIT 1
            """, (plate_number,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_track1_history(plate_number: str) -> List[Dict[str, Any]]:
    """
    Fetch all Track 1 records for a plate number, ordered by day.
    Used to show sensor trend over the 30-day window in the technician app.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM track1_technician
                WHERE plate_number = %s
                ORDER BY day_in_window ASC
            """, (plate_number,))
            return [dict(r) for r in cur.fetchall()]


def get_track1_avg_sensors(plate_number: str) -> Optional[Dict[str, Any]]:
    """
    Compute the 30-day average of all sensor readings for a given plate.
    This is the correct input for the ML classifier — it was trained on
    aggregated telemetry, not single-day snapshots.

    Also returns vehicle identity and fault ground truth from the most
    recent row (day 30) for display purposes.

    Args:
        plate_number: Vehicle plate number (e.g. 'B 1234 ABC')

    Returns:
        Dict with:
          - averaged sensor values (keyed as the ML model expects)
          - owner_name, car_model, car_year, plate_number (from day 30)
          - true_fault_class, true_fault_label (from day 30)
          - test_id (from day 30, used for saving predictions)
        or None if plate not found.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Fetch all 30 rows for averaging
            cur.execute("""
                SELECT
                    AVG(o2_sensor_v)        AS o2_sensor_v,
                    AVG(maf_g_per_s)        AS maf_g_per_s,
                    AVG(throttle_pos_pct)   AS throttle_pos_pct,
                    AVG(crank_rpm)          AS crank_rpm,
                    AVG(cam_advance_deg)    AS cam_advance_deg,
                    AVG(knock_count_30d)    AS knock_count_30d,
                    AVG(coolant_temp_c)     AS coolant_temp_c,
                    AVG(oil_pressure_psi)   AS oil_pressure_psi,
                    AVG(map_kpa)            AS map_kpa,
                    AVG(egr_duty_pct)       AS egr_duty_pct,
                    AVG(battery_voltage_v)  AS battery_voltage_v,
                    AVG(fuel_temp_c)        AS fuel_temp_c,
                    COUNT(*)                AS days_recorded
                FROM track1_technician
                WHERE plate_number = %s
            """, (plate_number,))
            avg_row = cur.fetchone()

            if not avg_row or avg_row['days_recorded'] == 0:
                return None

            # Fetch identity + ground truth from the latest day
            cur.execute("""
                SELECT test_id, owner_name, plate_number,
                       car_model, car_year,
                       true_fault_class, true_fault_label
                FROM track1_technician
                WHERE plate_number = %s
                ORDER BY day_in_window DESC
                LIMIT 1
            """, (plate_number,))
            meta = cur.fetchone()

            if not meta:
                return None

            return {
                # Identity
                'test_id':           meta['test_id'],
                'owner_name':        meta['owner_name'],
                'plate_number':      meta['plate_number'],
                'car_model':         meta['car_model'],
                'car_year':          meta['car_year'],
                # Ground truth
                'true_fault_class':  meta['true_fault_class'],
                'true_fault_label':  meta['true_fault_label'],
                # 30-day averaged sensors (float for ML input)
                'O2 SENSOR V':       float(avg_row['o2_sensor_v']),
                'MAF G PER S':       float(avg_row['maf_g_per_s']),
                'THROTTLE POS PCT':  float(avg_row['throttle_pos_pct']),
                'CRANK RPM':         float(avg_row['crank_rpm']),
                'CAM ADVANCE DEG':   float(avg_row['cam_advance_deg']),
                'KNOCK COUNT 30D':   float(avg_row['knock_count_30d']),
                'COOLANT TEMP C':    float(avg_row['coolant_temp_c']),
                'OIL PRESSURE PSI':  float(avg_row['oil_pressure_psi']),
                'MAP KPA':           float(avg_row['map_kpa']),
                'EGR DUTY PCT':      float(avg_row['egr_duty_pct']),
                'BATTERY VOLTAGE V': float(avg_row['battery_voltage_v']),
                'FUEL TEMP C':       float(avg_row['fuel_temp_c']),
                'days_recorded':     int(avg_row['days_recorded']),
            }


def save_track1_prediction(
    test_id: str,
    predicted_class: int,
    predicted_label: str,
) -> None:
    """Write ML prediction result back to track1_technician table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE track1_technician
                SET predicted_class  = %s,
                    predicted_label  = %s,
                    predicted_at     = NOW()
                WHERE test_id = %s
            """, (predicted_class, predicted_label, test_id))
    log.info(f"Saved T1 prediction: {test_id} → {predicted_class} ({predicted_label})")


# ─────────────────────────────────────────────────────────────────────────────
# QUERIES — TRACK 2
# ─────────────────────────────────────────────────────────────────────────────

def get_track2_record(plate_number: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent Track 2 record for a given plate number.
    Returns the row with the latest hour_of_day reading.

    Args:
        plate_number: Vehicle plate number (e.g. 'B 1234 ABC')

    Returns:
        Dict with all sensor readings and metadata, or None if not found.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM track2_owner
                WHERE plate_number = %s
                ORDER BY hour_of_day DESC
                LIMIT 1
            """, (plate_number,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_track2_history(plate_number: str) -> List[Dict[str, Any]]:
    """
    Fetch all Track 2 records for a plate, ordered by hour.
    Used to show the day's sensor readings in the owner app.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM track2_owner
                WHERE plate_number = %s
                ORDER BY hour_of_day ASC
            """, (plate_number,))
            return [dict(r) for r in cur.fetchall()]


def get_track2_avg_sensors(plate_number: str) -> Optional[Dict[str, Any]]:
    """
    Compute the daily average of all 7 two-hourly sensor readings
    (07:00–19:00) for a given plate.

    The ML risk classifier receives this averaged dict — consistent with
    how the system was designed: pattern detection across the day,
    not a single-moment snapshot.

    Returns:
        Dict with averaged sensor values, vehicle identity, and ground
        truth — or None if plate not found.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # 7-reading daily average
            cur.execute("""
                SELECT
                    AVG(o2_sensor_v)        AS o2_sensor_v,
                    AVG(maf_g_per_s)        AS maf_g_per_s,
                    AVG(throttle_pos_pct)   AS throttle_pos_pct,
                    AVG(coolant_temp_c)     AS coolant_temp_c,
                    AVG(oil_pressure_psi)   AS oil_pressure_psi,
                    AVG(battery_voltage_v)  AS battery_voltage_v,
                    AVG(tpms_psi)           AS tpms_psi,
                    AVG(ambient_temp_c)     AS ambient_temp_c,
                    AVG(cabin_humidity_pct) AS cabin_humidity_pct,
                    AVG(fuel_level_pct)     AS fuel_level_pct,
                    AVG(brake_pedal_events) AS brake_pedal_events,
                    AVG(speed_kmh)          AS speed_kmh,
                    COUNT(*)                AS readings_count
                FROM track2_owner
                WHERE plate_number = %s
            """, (plate_number,))
            avg_row = cur.fetchone()

            if not avg_row or avg_row['readings_count'] == 0:
                return None

            # Identity + ground truth from the latest hour
            cur.execute("""
                SELECT test_id, owner_name, plate_number,
                       car_model, car_year,
                       true_risk_class, true_risk_label
                FROM track2_owner
                WHERE plate_number = %s
                ORDER BY hour_of_day DESC
                LIMIT 1
            """, (plate_number,))
            meta = cur.fetchone()

            if not meta:
                return None

            return {
                # Identity
                'test_id':           meta['test_id'],
                'owner_name':        meta['owner_name'],
                'plate_number':      meta['plate_number'],
                'car_model':         meta['car_model'],
                'car_year':          meta['car_year'],
                # Ground truth
                'true_risk_class':   meta['true_risk_class'],
                'true_risk_label':   meta['true_risk_label'],
                # 7-reading daily averaged sensors (float for ML input)
                'O2 SENSOR V':        float(avg_row['o2_sensor_v']),
                'MAF G PER S':        float(avg_row['maf_g_per_s']),
                'THROTTLE POS PCT':   float(avg_row['throttle_pos_pct']),
                'COOLANT TEMP C':     float(avg_row['coolant_temp_c']),
                'OIL PRESSURE PSI':   float(avg_row['oil_pressure_psi']),
                'BATTERY VOLTAGE V':  float(avg_row['battery_voltage_v']),
                'TPMS PSI':           float(avg_row['tpms_psi']),
                'AMBIENT TEMP C':     float(avg_row['ambient_temp_c']),
                'CABIN HUMIDITY PCT': float(avg_row['cabin_humidity_pct']),
                'FUEL LEVEL PCT':     float(avg_row['fuel_level_pct']),
                'BRAKE PEDAL EVENTS': float(avg_row['brake_pedal_events']),
                'SPEED KMH':          float(avg_row['speed_kmh']),
                'readings_count':     int(avg_row['readings_count']),
                # Averaged values also as lowercase keys for sensor pill display
                'coolant_temp_c':     float(avg_row['coolant_temp_c']),
                'oil_pressure_psi':   float(avg_row['oil_pressure_psi']),
                'battery_voltage_v':  float(avg_row['battery_voltage_v']),
                'tpms_psi':           float(avg_row['tpms_psi']),
                'fuel_level_pct':     float(avg_row['fuel_level_pct']),
                'speed_kmh':          float(avg_row['speed_kmh']),
            }


def save_track2_prediction(
    test_id: str,
    predicted_class: int,
    predicted_label: str,
) -> None:
    """Write ML prediction result back to track2_owner table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE track2_owner
                SET predicted_class  = %s,
                    predicted_label  = %s,
                    predicted_at     = NOW()
                WHERE test_id = %s
            """, (predicted_class, predicted_label, test_id))
    log.info(f"Saved T2 prediction: {test_id} → {predicted_class} ({predicted_label})")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY LOG
# ─────────────────────────────────────────────────────────────────────────────

def log_query(
    track: int,
    plate_number: str,
    test_id: str,
    predicted_class: int,
    predicted_label: str,
    response_time_ms: int,
    context_chunks: int,
    llm_called: bool,
) -> None:
    """
    Insert a row into query_log after every prediction request.
    This is the audit trail — separate from Prometheus metrics.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_log (
                    track, plate_number, test_id,
                    predicted_class, predicted_label,
                    response_time_ms, context_chunks, llm_called
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                track, plate_number, test_id,
                predicted_class, predicted_label,
                response_time_ms, context_chunks, llm_called,
            ))


def get_query_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch the most recent query log entries. Used in monitoring sidebar."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM query_log
                ORDER BY queried_at DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR DICT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def row_to_track1_sensors(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a track1_technician DB row into the sensor dict
    expected by llm_chain.run_track1() and the ML classifier.
    """
    return {
        "O2 SENSOR V":       float(row["o2_sensor_v"]),
        "MAF G PER S":       float(row["maf_g_per_s"]),
        "THROTTLE POS PCT":  float(row["throttle_pos_pct"]),
        "CRANK RPM":         float(row["crank_rpm"]),
        "CAM ADVANCE DEG":   float(row["cam_advance_deg"]),
        "KNOCK COUNT 30D":   float(row["knock_count_30d"]),
        "COOLANT TEMP C":    float(row["coolant_temp_c"]),
        "OIL PRESSURE PSI":  float(row["oil_pressure_psi"]),
        "MAP KPA":           float(row["map_kpa"]),
        "EGR DUTY PCT":      float(row["egr_duty_pct"]),
        "BATTERY VOLTAGE V": float(row["battery_voltage_v"]),
        "FUEL TEMP C":       float(row["fuel_temp_c"]),
    }


def row_to_track2_sensors(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a track2_owner DB row into the sensor dict
    expected by llm_chain.run_track2() and the ML classifier.
    """
    return {
        "O2 SENSOR V":        float(row["o2_sensor_v"]),
        "MAF G PER S":        float(row["maf_g_per_s"]),
        "THROTTLE POS PCT":   float(row["throttle_pos_pct"]),
        "COOLANT TEMP C":     float(row["coolant_temp_c"]),
        "OIL PRESSURE PSI":   float(row["oil_pressure_psi"]),
        "BATTERY VOLTAGE V":  float(row["battery_voltage_v"]),
        "TPMS PSI":           float(row["tpms_psi"]),
        "AMBIENT TEMP C":     float(row["ambient_temp_c"]),
        "CABIN HUMIDITY PCT": float(row["cabin_humidity_pct"]),
        "FUEL LEVEL PCT":     float(row["fuel_level_pct"]),
        "BRAKE PEDAL EVENTS": float(row["brake_pedal_events"]),
        "SPEED KMH":          float(row["speed_kmh"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n" + "=" * 60)
    print("  db.py — Self Test")
    print("=" * 60)

    # Step 1: Create schema
    print("\n[1] Creating schema ...")
    create_schema()

    # Step 2: Seed from Excel
    print("\n[2] Seeding from Excel ...")
    seed_database()

    # Step 3: Query plates
    print("\n[3] Available plates (Track 1):")
    plates_t1 = get_all_plates(track=1)
    for p in plates_t1[:5]:
        print(f"   {p}")
    print(f"   ... ({len(plates_t1)} total)")

    # Step 4: Fetch a Track 1 record
    test_plate = plates_t1[0]
    print(f"\n[4] Track 1 record for plate: {test_plate}")
    rec = get_track1_record(test_plate)
    if rec:
        print(f"   Owner     : {rec['owner_name']}")
        print(f"   Model     : {rec['car_model']} {rec['car_year']}")
        print(f"   True Fault: {rec['true_fault_class']} — {rec['true_fault_label']}")
        sensors = row_to_track1_sensors(rec)
        print(f"   Sensors   : {len(sensors)} features extracted")

    # Step 5: Fetch a Track 2 record
    plates_t2 = get_all_plates(track=2)
    test_plate2 = plates_t2[0]
    print(f"\n[5] Track 2 record for plate: {test_plate2}")
    rec2 = get_track2_record(test_plate2)
    if rec2:
        print(f"   Owner     : {rec2['owner_name']}")
        print(f"   Model     : {rec2['car_model']} {rec2['car_year']}")
        print(f"   True Risk : {rec2['true_risk_class']} — {rec2['true_risk_label']}")
        sensors2 = row_to_track2_sensors(rec2)
        print(f"   Sensors   : {len(sensors2)} features extracted")

    # Step 6: Log a dummy query
    print("\n[6] Logging dummy query ...")
    if rec:
        log_query(
            track=1,
            plate_number=test_plate,
            test_id=rec["test_id"],
            predicted_class=0,
            predicted_label="Normal",
            response_time_ms=0,
            context_chunks=0,
            llm_called=False,
        )
        print("   Query logged.")

    # Step 7: Read query log
    print("\n[7] Recent query log:")
    logs = get_query_log(limit=3)
    for entry in logs:
        print(f"   {entry['queried_at']} | T{entry['track']} | "
              f"{entry['plate_number']} | {entry['predicted_label']}")

    print("\n" + "=" * 60)
    print("  Self-test complete.")
    print("=" * 60)
