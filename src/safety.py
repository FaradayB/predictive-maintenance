"""
=============================================================================
 src/safety.py
 Vehicle Predictive Maintenance — Input Validation & Output Guardrails
=============================================================================
 Two responsibilities:
 
   1. INPUT VALIDATION
      validate_plate()    — plate number format check
      validate_sensors()  — sensor values within physically possible ranges
 
   2. OUTPUT GUARDRAILS
      validate_brief()    — Track 1: ensure required sections are present
      validate_alert()    — Track 2: ensure Bahasa Indonesia, no sensor values
=============================================================================
"""
 
import re
import logging
from typing import Dict, Any, Tuple
 
log = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# PLATE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
 
# Indonesian plate: 1-2 letters, space, 1-4 digits, space, 1-3 letters
# Examples: B 1234 ABC, T 4567 BCD, AB 123 DE
PLATE_PATTERN = re.compile(
    r"^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$",
    re.IGNORECASE,
)
 
def validate_plate(plate: str) -> Tuple[bool, str]:
    """
    Validate Indonesian vehicle plate number format.
 
    Returns:
        (True, "") if valid
        (False, error_message) if invalid
    """
    if not plate or not plate.strip():
        return False, "Plate number cannot be empty."
 
    plate = plate.strip().upper()
 
    if not PLATE_PATTERN.match(plate):
        return False, (
            f"Invalid plate format: '{plate}'. "
            "Expected format: B 1234 ABC (region code, digits, suffix)."
        )
 
    return True, ""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SENSOR RANGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
 
# Physically possible ranges — much wider than SOP thresholds.
# These catch impossible values (e.g. negative RPM, 500°C coolant)
# before they hit the ML model or LLM.
 
TRACK1_PHYSICAL_RANGES = {
    "O2 SENSOR V":       (0.0,   1.2),
    "MAF G PER S":       (0.0,   30.0),
    "THROTTLE POS PCT":  (0.0,   100.0),
    "CRANK RPM":         (0.0,   8000.0),
    "CAM ADVANCE DEG":   (-10.0, 60.0),
    "KNOCK COUNT 30D":   (0.0,   200.0),
    "COOLANT TEMP C":    (-40.0, 200.0),
    "OIL PRESSURE PSI":  (0.0,   150.0),
    "MAP KPA":           (0.0,   250.0),
    "EGR DUTY PCT":      (0.0,   100.0),
    "BATTERY VOLTAGE V": (0.0,   20.0),
    "FUEL TEMP C":       (-40.0, 150.0),
}
 
TRACK2_PHYSICAL_RANGES = {
    "O2 SENSOR V":        (0.0,   1.2),
    "MAF G PER S":        (0.0,   30.0),
    "THROTTLE POS PCT":   (0.0,   100.0),
    "COOLANT TEMP C":     (-40.0, 200.0),
    "OIL PRESSURE PSI":   (0.0,   150.0),
    "BATTERY VOLTAGE V":  (0.0,   20.0),
    "TPMS PSI":           (0.0,   100.0),
    "AMBIENT TEMP C":     (-20.0, 70.0),
    "CABIN HUMIDITY PCT": (0.0,   100.0),
    "FUEL LEVEL PCT":     (0.0,   100.0),
    "BRAKE PEDAL EVENTS": (0.0,   500.0),
    "SPEED KMH":          (0.0,   300.0),
}
 
 
def validate_sensors(
    sensors: Dict[str, float],
    track: int,
) -> Tuple[bool, str]:
    """
    Validate that all sensor values are within physically possible ranges.
 
    Args:
        sensors: Dict of sensor_name -> value
        track:   1 or 2
 
    Returns:
        (True, "") if all values valid
        (False, error_message) listing out-of-range sensors
    """
    ranges = TRACK1_PHYSICAL_RANGES if track == 1 else TRACK2_PHYSICAL_RANGES
    errors = []
 
    for sensor, value in sensors.items():
        if sensor not in ranges:
            continue
        lo, hi = ranges[sensor]
        if not (lo <= float(value) <= hi):
            errors.append(
                f"{sensor}: {value} is outside physically possible "
                f"range [{lo}, {hi}]"
            )
 
    if errors:
        return False, "Sensor validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
 
    return True, ""
 
 
def validate_sensor_completeness(
    sensors: Dict[str, float],
    track: int,
) -> Tuple[bool, str]:
    """
    Check that all required sensors for the given track are present.
 
    Returns:
        (True, "") if all required sensors present
        (False, error_message) listing missing sensors
    """
    required = (
        list(TRACK1_PHYSICAL_RANGES.keys()) if track == 1
        else list(TRACK2_PHYSICAL_RANGES.keys())
    )
    missing = [s for s in required if s not in sensors]
 
    if missing:
        return False, f"Missing sensors for Track {track}: {missing}"
 
    return True, ""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAILS — TRACK 1
# ─────────────────────────────────────────────────────────────────────────────
 
REQUIRED_BRIEF_SECTIONS = [
    "SENSOR FINDINGS",
    "PROBABLE CAUSE",
    "INSPECTION CHECKLIST",
    "RECOMMENDED ACTION",
    "SOP REFERENCE",
]
 
def validate_brief(brief: str) -> Tuple[bool, str]:
    """
    Validate that the LLM-generated Technician Fault Brief contains
    all required sections. If any are missing, the caller should re-prompt.
 
    Returns:
        (True, "") if all sections present
        (False, error_message) listing missing sections
    """
    if not brief or len(brief.strip()) < 50:
        return False, "Fault brief is empty or too short."
 
    missing = [
        section for section in REQUIRED_BRIEF_SECTIONS
        if section not in brief.upper()
    ]
 
    if missing:
        return False, f"Fault brief missing required sections: {missing}"
 
    return True, ""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAILS — TRACK 2
# ─────────────────────────────────────────────────────────────────────────────
 
# Raw sensor values the owner should never see
FORBIDDEN_OWNER_PATTERNS = [
    r"\d+\.\d+\s*(PSI|V|°C|kPa|g/s|km/h|%)",  # numeric values with units
    r"OBD",
    r"P\d{4}",                                   # OBD fault codes (P0300 etc)
    r"sensor",                                    # technical term
    r"coolant temperature sensor",
    r"oil pressure sensor",
    r"battery voltage sensor",
]

BAHASA_INDICATORS = [
    "kendaraan", "kondisi", "segera", "servis", "bengkel",
    "risiko", "bahaya", "aman", "periksa", "hubungi",
]
 
def validate_alert(alert: str) -> Tuple[bool, str]:
    """
    Validate that the LLM-generated owner alert:
    - Is not empty
    - Contains Bahasa Indonesia indicators
    - Does not contain raw sensor values or technical terms
 
    Returns:
        (True, "") if valid
        (False, error_message) describing the issue
    """
    if not alert or len(alert.strip()) < 20:
        return False, "Owner alert is empty or too short."
 
    alert_lower = alert.lower()
 
    # Check for Bahasa Indonesia content
    has_bahasa = any(word in alert_lower for word in BAHASA_INDICATORS)
    if not has_bahasa:
        return False, "Alert does not appear to be in Bahasa Indonesia."
 
    # Check for forbidden patterns (raw sensor values / technical terms)
    violations = []
    for pattern in FORBIDDEN_OWNER_PATTERNS:
        if re.search(pattern, alert, re.IGNORECASE):
            violations.append(pattern)
 
    if violations:
        log.warning(f"Alert output guardrail violations: {violations}")
        # Warn but don't hard-block — log for review
        # Return True so the alert still reaches the owner,
        # but the violation is logged for monitoring
        return True, f"WARNING: alert may contain technical content: {violations}"
 
    return True, ""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# COMBINED VALIDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
 
def validate_request(
    plate: str,
    sensors: Dict[str, float],
    track: int,
) -> Tuple[bool, str]:
    """
    Run all input validations for a prediction request.
    Returns (True, "") if all pass, (False, error_message) on first failure.
    """
    ok, err = validate_plate(plate)
    if not ok:
        return False, err
 
    ok, err = validate_sensor_completeness(sensors, track)
    if not ok:
        return False, err
 
    ok, err = validate_sensors(sensors, track)
    if not ok:
        return False, err
 
    return True, ""
 
 
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PROMPT INJECTION & RED TEAM DEFENCE
# ─────────────────────────────────────────────────────────────────────────────
 
import re as _re
 
# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore (all |previous |your )?(instructions|prompt|rules|system)",
    r"forget (everything|your|all|previous)",
    r"you are now",
    r"new persona",
    r"act as (a |an )?(?!predictivecare|technician|assistant)",
    r"jailbreak",
    r"disregard (your|all|previous)",
    r"override (your|the|all)",
    r"pretend (you are|to be|that)",
    r"roleplay as",
    r"simulate (being|a|an)",
    r"do anything now",
    r"dan mode",
    r"developer mode",
    r"bypass (your|the|all)",
    r"reveal (your|the) (system prompt|instructions|prompt)",
    r"print (your|the) (system prompt|instructions)",
    r"what (is|are) your (instructions|prompt|rules)",
]
 
# Off-topic keywords — nothing to do with vehicles or maintenance
OFF_TOPIC_KEYWORDS = [
    "politik", "agama", "seksual", "pornografi",
    "weapon", "bomb", "explosive", "drug", "illegal",
    "hack", "malware", "virus", "ransomware",
    "password", "credit card", "bank account",
    "kill", "murder", "suicide", "harm",
]
 
# Maximum allowed query length (characters)
MAX_QUERY_LENGTH = 1000
 
 
def sanitise_query(query: str) -> tuple:
    """
    Sanitise a user query before passing to the LLM.
 
    Checks for:
    1. Query length
    2. Prompt injection patterns
    3. Off-topic / harmful keywords
 
    Returns:
        (True, clean_query)  if safe
        (False, reason)      if blocked
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."
 
    query = query.strip()
 
    # Length check
    if len(query) > MAX_QUERY_LENGTH:
        return False, (
            f"Query too long ({len(query)} chars). "
            f"Maximum allowed: {MAX_QUERY_LENGTH} chars."
        )
 
    query_lower = query.lower()
 
    # Prompt injection check
    for pattern in INJECTION_PATTERNS:
        if _re.search(pattern, query_lower):
            return False, (
                "Query contains instructions that attempt to override "
                "system behaviour. Please ask a question about your vehicle."
            )
 
    # Off-topic keyword check
    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in query_lower:
            return False, (
                "Query contains off-topic or inappropriate content. "
                "Please ask questions related to vehicle maintenance."
            )
 
    return True, query
 
 
def sanitise_chat_query(query: str, track: int) -> tuple:
    """
    Sanitise a follow-up chat message with track-specific context check.
 
    Track 1 (Technician): questions must relate to vehicle diagnosis/repair.
    Track 2 (Owner): questions must relate to vehicle condition/safety.
 
    Returns:
        (True, clean_query) if safe
        (False, reason)     if blocked
    """
    # Run base sanitisation first
    ok, result = sanitise_query(query)
    if not ok:
        return False, result
 
    query_lower = result.lower()
 
    # Track-specific context boundary
    vehicle_keywords_t1 = [
        "sensor", "engine", "oil", "brake", "coolant", "battery",
        "transmission", "alternator", "inspect", "repair", "replace",
        "fault", "misfire", "pressure", "temperature", "rpm", "voltage",
        "tool", "workshop", "technician", "check", "diagnostic", "fix",
        "part", "component", "service", "maintenance", "vehicle", "car",
    ]
 
    vehicle_keywords_t2 = [
        "kendaraan", "mobil", "servis", "bengkel", "aman", "bahaya",
        "ban", "mesin", "oli", "aki", "bahan bakar", "rem",
        "safe", "drive", "repair", "workshop", "vehicle", "car",
        "condition", "risk", "alert", "maintenance", "check",
        "temperature", "pressure", "fuel", "battery", "tyre",
    ]
 
    keywords = vehicle_keywords_t1 if track == 1 else vehicle_keywords_t2
    has_context = any(kw in query_lower for kw in keywords)
 
    if not has_context and len(result.split()) > 5:
        # Only block longer queries that clearly have no vehicle context
        return False, (
            "Please ask questions related to your vehicle condition or maintenance. "
            if track == 2 else
            "Please ask questions related to the vehicle diagnosis or repair procedure."
        )
 
    return True, result


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n=== Safety — Self Test ===\n")
 
    # Plate validation
    for plate, expected in [
        ("B 1234 ABC", True),
        ("T 4567 BCD", True),
        ("AB 123 DE",  True),
        ("INVALID",    False),
        ("",           False),
        ("B1234ABC",   False),
    ]:
        ok, err = validate_plate(plate)
        status = "PASS" if ok == expected else "FAIL"
        print(f"  [{status}] Plate '{plate}': valid={ok}  {err}")
 
    # Sensor range
    bad_sensors = {
        "O2 SENSOR V":      5.0,   # impossible
        "COOLANT TEMP C":   91.0,  # fine
        "OIL PRESSURE PSI": -5.0,  # impossible
        "BATTERY VOLTAGE V": 14.2, # fine
    }
    ok, err = validate_sensors(bad_sensors, track=1)
    print(f"\n  Sensor range check (2 bad values): valid={ok}\n  {err}")
 
    # Brief validation
    good_brief = """
    SENSOR FINDINGS
    PROBABLE CAUSE
    INSPECTION CHECKLIST
    RECOMMENDED ACTION
    SOP REFERENCE
    """
    ok, err = validate_brief(good_brief)
    print(f"\n  Brief with all sections: valid={ok}")
 
    bad_brief = "The vehicle has some issues."
    ok, err = validate_brief(bad_brief)
    print(f"  Brief missing sections: valid={ok}  {err}")
 
    # Alert validation
    good_alert = "🔴 Kendaraan Anda perlu segera diperiksa di bengkel resmi."
    ok, err = validate_alert(good_alert)
    print(f"\n  Good alert: valid={ok}")
 
    bad_alert = "Oil pressure sensor reading: 17.5 PSI"
    ok, err = validate_alert(bad_alert)
    print(f"  Alert with sensor value: valid={ok}  {err}")
 
    print("\n=== Self-test complete ===")
