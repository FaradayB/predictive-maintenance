"""
=============================================================================
 src/eval_runner.py
 Vehicle Predictive Maintenance — Batch Evaluation Runner
=============================================================================
 Runs the full prediction pipeline against all rows in the test dataset
 and outputs accuracy, F1, and per-class metrics to CSV.

 Usage:
   python -m src.eval_runner --track 1
   python -m src.eval_runner --track 2
   python -m src.eval_runner --track both

 Outputs:
   reports/eval_report_track1.csv
   reports/eval_report_track2.csv
   reports/eval_summary.csv
=============================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from dotenv import load_dotenv

# Add project root to path so imports work

from predictivecare.database import (
    get_all_plates,
    get_track1_avg_sensors,
    get_track2_avg_sensors,
)
from predictivecare.logger import log_request
from predictivecare.monitoring import record_track1_query, record_track2_query
from predictivecare.safety import validate_request

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRACK1_FEATURES = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT", "CRANK RPM",
    "CAM ADVANCE DEG", "KNOCK COUNT 30D", "COOLANT TEMP C",
    "OIL PRESSURE PSI", "MAP KPA", "EGR DUTY PCT",
    "BATTERY VOLTAGE V", "FUEL TEMP C",
]

TRACK2_FEATURES = [
    "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT",
    "COOLANT TEMP C", "OIL PRESSURE PSI", "BATTERY VOLTAGE V",
    "TPMS PSI", "AMBIENT TEMP C", "CABIN HUMIDITY PCT",
    "FUEL LEVEL PCT", "BRAKE PEDAL EVENTS", "SPEED KMH",
]

FAULT_LABELS = {
    0: "Normal", 1: "Battery Degradation", 2: "Brake System Issue",
    3: "Cooling System Problem", 4: "Engine Misfire",
    5: "Alternator Failure", 6: "Oil Pressure Issue",
    7: "Transmission Problem",
}

RISK_LABELS = {
    0: "No Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk",
}


# ─────────────────────────────────────────────────────────────────────────────
# TRACK 1 EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_track1_eval() -> pd.DataFrame:
    """
    Run ML predictions on all Track 1 test records.
    Returns a DataFrame with one row per test case.
    """
    log.info("Loading Track 1 classifier ...")
    model = joblib.load("ml_models/track1_fault_classifier.pkl")

    plates = get_all_plates(track=1)
    log.info(f"Evaluating Track 1 on {len(plates)} plates ...")

    rows = []
    for plate in plates:
        # Average all 30 daily readings — same logic as the live app
        record = get_track1_avg_sensors(plate)
        if not record:
            continue

        sensors = {k: record[k] for k in [
            "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT", "CRANK RPM",
            "CAM ADVANCE DEG", "KNOCK COUNT 30D", "COOLANT TEMP C",
            "OIL PRESSURE PSI", "MAP KPA", "EGR DUTY PCT",
            "BATTERY VOLTAGE V", "FUEL TEMP C",
        ]}

        # Safety check
        ok, err = validate_request(plate, sensors, track=1)
        if not ok:
            log.warning(f"  Skipped {plate}: {err}")
            rows.append({
                "test_id": record.get("test_id"), "plate_number": plate,
                "true_class": record["true_fault_class"],
                "true_label": record["true_fault_label"],
                "predicted_class": -1, "predicted_label": "INVALID",
                "correct": False, "safety_passed": False,
            })
            continue

        X = np.array([[sensors[f] for f in TRACK1_FEATURES]])
        pred_class = int(model.predict(X)[0])
        pred_label = FAULT_LABELS.get(pred_class, "Unknown")
        correct    = (pred_class == int(record["true_fault_class"]))

        rows.append({
            "test_id":         record.get("test_id"),
            "plate_number":    plate,
            "owner_name":      record.get("owner_name"),
            "car_model":       record.get("car_model"),
            "true_class":      int(record["true_fault_class"]),
            "true_label":      record["true_fault_label"],
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "correct":         correct,
            "safety_passed":   True,
        })

        # Structured log
        log_request(
            track=1, plate_number=plate,
            predicted_class=pred_class, predicted_label=pred_label,
            response_time_ms=0, context_chunks=0,
            llm_called=False, test_id=record.get("test_id"),
            true_class=int(record["true_fault_class"]),
            true_label=record["true_fault_label"],
        )
        record_track1_query({
            "fault_class":      pred_class,
            "fault_label":      pred_label,
            "response_time_ms": 0,
            "context_chunks":   0,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TRACK 2 EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_track2_eval() -> pd.DataFrame:
    """
    Run ML predictions on all Track 2 test records.
    Returns a DataFrame with one row per test case.
    """
    log.info("Loading Track 2 classifier ...")
    model = joblib.load("ml_models/track2_risk_classifier.pkl")

    plates = get_all_plates(track=2)
    log.info(f"Evaluating Track 2 on {len(plates)} plates ...")

    rows = []
    for plate in plates:
        # Average all 7 daily readings — same logic as the live app
        record = get_track2_avg_sensors(plate)
        if not record:
            continue

        sensors = {k: record[k] for k in [
            "O2 SENSOR V", "MAF G PER S", "THROTTLE POS PCT",
            "COOLANT TEMP C", "OIL PRESSURE PSI", "BATTERY VOLTAGE V",
            "TPMS PSI", "AMBIENT TEMP C", "CABIN HUMIDITY PCT",
            "FUEL LEVEL PCT", "BRAKE PEDAL EVENTS", "SPEED KMH",
        ]}

        ok, err = validate_request(plate, sensors, track=2)
        if not ok:
            log.warning(f"  Skipped {plate}: {err}")
            rows.append({
                "test_id": record.get("test_id"), "plate_number": plate,
                "true_class": record["true_risk_class"],
                "true_label": record["true_risk_label"],
                "predicted_class": -1, "predicted_label": "INVALID",
                "correct": False, "safety_passed": False,
            })
            continue

        X = np.array([[sensors[f] for f in TRACK2_FEATURES]])
        pred_class = int(model.predict(X)[0])
        pred_label = RISK_LABELS.get(pred_class, "Unknown")
        correct    = (pred_class == int(record["true_risk_class"]))

        rows.append({
            "test_id":         record.get("test_id"),
            "plate_number":    plate,
            "owner_name":      record.get("owner_name"),
            "car_model":       record.get("car_model"),
            "true_class":      int(record["true_risk_class"]),
            "true_label":      record["true_risk_label"],
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "correct":         correct,
            "safety_passed":   True,
        })

        log_request(
            track=2, plate_number=plate,
            predicted_class=pred_class, predicted_label=pred_label,
            response_time_ms=0, context_chunks=0,
            llm_called=False, test_id=record.get("test_id"),
            true_class=int(record["true_risk_class"]),
            true_label=record["true_risk_label"],
        )
        record_track2_query({
            "risk_class":       pred_class,
            "risk_label":       pred_label,
            "response_time_ms": 0,
            "context_chunks":   0,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, track: int) -> Dict[str, Any]:
    """
    Compute accuracy, weighted F1, and per-class breakdown from eval results.
    """
    valid = df[df["predicted_class"] >= 0]

    y_true = valid["true_class"].tolist()
    y_pred = valid["predicted_class"].tolist()
    labels = FAULT_LABELS if track == 1 else RISK_LABELS

    accuracy = accuracy_score(y_true, y_pred)
    f1_w     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=[labels.get(i, str(i)) for i in sorted(set(y_true + y_pred))],
        output_dict=True, zero_division=0,
    )

    return {
        "track":          track,
        "total_samples":  len(df),
        "valid_samples":  len(valid),
        "skipped":        len(df) - len(valid),
        "accuracy":       round(accuracy, 4),
        "f1_weighted":    round(f1_w,     4),
        "f1_macro":       round(f1_macro, 4),
        "report":         report,
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    track = metrics["track"]
    print(f"\n{'='*55}")
    print(f"  Track {track} Evaluation Results")
    print(f"{'='*55}")
    print(f"  Samples  : {metrics['valid_samples']} / {metrics['total_samples']}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 (Wt)  : {metrics['f1_weighted']:.4f}")
    print(f"  F1 (Mac) : {metrics['f1_macro']:.4f}")
    if metrics["skipped"] > 0:
        print(f"  Skipped  : {metrics['skipped']} (failed safety check)")
    print()

    labels = FAULT_LABELS if track == 1 else RISK_LABELS
    report = metrics["report"]
    print(f"  {'Class':<28} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*55}")
    for label in labels.values():
        if label in report:
            r = report[label]
            print(f"  {label:<28} {r['precision']:>9.3f} {r['recall']:>8.3f} {r['f1-score']:>8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def save_report(df: pd.DataFrame, metrics: Dict[str, Any], track: int) -> Path:
    """Save per-row results and metrics summary to CSV."""
    # Per-row results
    detail_path = REPORTS_DIR / f"eval_report_track{track}.csv"
    df.to_csv(detail_path, index=False)
    log.info(f"Saved: {detail_path}")

    # Summary row
    summary = {
        "track":         track,
        "total_samples": metrics["total_samples"],
        "valid_samples": metrics["valid_samples"],
        "accuracy":      metrics["accuracy"],
        "f1_weighted":   metrics["f1_weighted"],
        "f1_macro":      metrics["f1_macro"],
        "skipped":       metrics["skipped"],
    }
    summary_path = REPORTS_DIR / "eval_summary.csv"
    summary_df   = pd.DataFrame([summary])

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Replace row for this track if it exists
        existing = existing[existing["track"] != track]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df.to_csv(summary_path, index=False)
    log.info(f"Saved: {summary_path}")

    return detail_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run batch evaluation against the test dataset."
    )
    parser.add_argument(
        "--track",
        choices=["1", "2", "both"],
        default="both",
        help="Which track to evaluate (default: both)",
    )
    args = parser.parse_args()

    tracks = []
    if args.track in ("1", "both"):
        tracks.append(1)
    if args.track in ("2", "both"):
        tracks.append(2)

    all_metrics = []

    for track in tracks:
        log.info(f"\nRunning Track {track} evaluation ...")
        t0 = time.time()

        df      = run_track1_eval() if track == 1 else run_track2_eval()
        metrics = compute_metrics(df, track)
        path    = save_report(df, metrics, track)

        print_metrics(metrics)
        all_metrics.append(metrics)
        log.info(f"Track {track} done in {time.time()-t0:.1f}s → {path}")

    # Final summary
    print(f"\n{'='*55}")
    print("  Evaluation Complete")
    print(f"{'='*55}")
    for m in all_metrics:
        print(f"  Track {m['track']}: Accuracy={m['accuracy']:.4f}  F1={m['f1_weighted']:.4f}")
    print(f"\n  Reports saved to: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
