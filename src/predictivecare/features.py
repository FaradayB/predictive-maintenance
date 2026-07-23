"""Canonical feature order and class labels for both tracks.

This is the single source of truth for feature ordering. Both training
(ml/train.py) and inference (api.py) import these lists, so the model always
receives its columns in the exact order it was trained on.
"""
from __future__ import annotations

# Track 1 - Technician fault classification (12 features, 8 classes)
TRACK1_FEATURES: list[str] = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT", "CRANK RPM",
    "CAM ADVANCE DEG", "KNOCK COUNT 30D", "COOLANT TEMP C",
    "OIL PRESSURE PSI", "MAP KPA", "EGR DUTY PCT",
    "BATTERY VOLTAGE V", "FUEL TEMP C",
]

TRACK1_LABELS: list[str] = [
    "Normal", "Battery Degradation", "Brake System Issue",
    "Cooling System Problem", "Engine Misfire", "Alternator Failure",
    "Oil Pressure Issue", "Transmission Problem",
]

# Track 2 - Owner risk detection (12 features, 4 classes)
TRACK2_FEATURES: list[str] = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT",
    "COOLANT TEMP C", "OIL PRESSURE PSI", "BATTERY VOLTAGE V",
    "TPMS PSI", "AMBIENT TEMP C", "CABIN HUMIDITY PCT",
    "FUEL LEVEL PCT", "BRAKE PEDAL EVENTS", "SPEED KMH",
]

TRACK2_LABELS: list[str] = ["No Risk", "Low Risk", "Medium Risk", "High Risk"]

TRACK1_TARGET = "FAULT CLASS"
TRACK2_TARGET = "RISK CLASS"
