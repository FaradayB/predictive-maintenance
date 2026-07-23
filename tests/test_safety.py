from predictivecare.safety import (
    validate_alert, validate_brief, validate_plate, validate_request,
    validate_sensors,
)


# --- plate -------------------------------------------------------------------
def test_valid_plate():
    ok, _ = validate_plate("B 1234 ABC")
    assert ok is True


def test_empty_plate_rejected():
    ok, _ = validate_plate("")
    assert ok is False


def test_malformed_plate_rejected():
    ok, _ = validate_plate("not-a-plate")
    assert ok is False


# --- sensors -----------------------------------------------------------------
def test_sensor_in_range_ok():
    ok, _ = validate_sensors({"COOLANT TEMP C": 90.0}, track=1)
    assert ok is True


def test_sensor_out_of_range_rejected():
    ok, _ = validate_sensors({"COOLANT TEMP C": 5000.0}, track=1)
    assert ok is False


# --- brief guardrail ---------------------------------------------------------
def test_brief_with_all_sections_valid():
    brief = (
        "TECHNICIAN FAULT BRIEF\n"
        "SENSOR FINDINGS: oil pressure low\n"
        "PROBABLE CAUSE: worn pump\n"
        "INSPECTION CHECKLIST: 1. check pump\n"
        "RECOMMENDED ACTION: replace pump\n"
        "SOP REFERENCE: SOP-001\n"
    )
    ok, _ = validate_brief(brief)
    assert ok is True


def test_brief_missing_section_rejected():
    ok, _ = validate_brief("SENSOR FINDINGS only, nothing else here at all.")
    assert ok is False


# --- alert guardrail (English) ----------------------------------------------
def test_english_alert_valid():
    ok, _ = validate_alert("Your vehicle needs a service at an authorised workshop soon.")
    assert ok is True


def test_empty_alert_rejected():
    ok, _ = validate_alert("")
    assert ok is False


def test_non_english_alert_rejected():
    # No English indicator words -> guardrail flags it as not in English.
    ok, _ = validate_alert("xxxxx yyyyy zzzzz qqqqq wwwww eeeee")
    assert ok is False


# --- request (composed) ------------------------------------------------------
def test_request_bad_plate_rejected():
    ok, _ = validate_request("bad!!", {}, track=1)
    assert ok is False
