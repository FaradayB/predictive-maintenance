"""
=============================================================================
 stress_test.py
 Vehicle Predictive Maintenance — Stress Test Script
=============================================================================
 Tests both Track 1 and Track 2 API endpoints under load.

 Usage:
   # Default: 3 concurrent users, 2 rounds each track
   python3 stress_test.py

   # Custom: 5 concurrent users, 5 rounds
   python3 stress_test.py --users 5 --rounds 5

   # Single track only
   python3 stress_test.py --track 1
   python3 stress_test.py --track 2

   # Against a different host
   python3 stress_test.py --host http://predictivecare.example.com

 Metrics reported per test:
   - Total requests sent
   - Success rate (%)
   - Average / p50 / p95 / p99 response time (ms)
   - Requests per second (throughput)
   - LLM calls made vs skipped
   - Errors by type
=============================================================================
"""

import argparse
import concurrent.futures
import json
import time
import statistics
import sys
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

import requests

# ─────────────────────────────────────────────────────────────────────────────
# TEST DATA — all 20 plates from the test dataset
# ─────────────────────────────────────────────────────────────────────────────

ALL_PLATES = [
    "B 1234 ABC", "B 2345 PQR", "B 3456 DEF", "B 5678 EFG",
    "B 6789 TUV", "B 7890 NOP", "B 8901 YZA", "B 9012 CDE",
    "D 1234 WXY", "D 5678 XYZ", "D 6789 STU", "D 9012 HIJ",
    "F 1234 VWX", "F 3456 KLM", "F 5678 ZAB", "F 9012 GHJ",
    "T 2345 QRS", "T 3456 FGH", "T 4567 BCD", "T 7890 MNO",
]

# Invalid plates — for testing safety/validation
INVALID_PLATES = [
    "INVALID",
    "B1234ABC",
    "",
    "XX 9999 ZZZ",
]

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE REQUEST
# ─────────────────────────────────────────────────────────────────────────────

def make_request(
    host: str,
    track: int,
    plate: str,
    timeout: int = 90,
) -> Dict[str, Any]:
    """
    Make a single prediction request and return timing + result info.
    """
    endpoint = (
        f"{host}/api/v1/track1/diagnose" if track == 1
        else f"{host}/api/v1/track2/alert"
    )

    start = time.time()
    result = {
        "plate":        plate,
        "track":        track,
        "endpoint":     endpoint,
        "status":       None,
        "response_ms":  0,
        "llm_called":   False,
        "label":        None,
        "error":        None,
        "status_code":  None,
    }

    try:
        resp = requests.post(
            endpoint,
            json={"plate_number": plate},
            timeout=timeout,
        )
        result["response_ms"]  = int((time.time() - start) * 1000)
        result["status_code"]  = resp.status_code

        if resp.status_code == 200:
            data = resp.json()
            result["status"] = "success"

            if track == 1:
                result["label"]      = data.get("fault_label", "unknown")
                result["llm_called"] = data.get("fault_class", 0) != 0
            else:
                result["label"]      = data.get("risk_label", "unknown")
                result["llm_called"] = data.get("risk_class", 0) > 0

        elif resp.status_code == 422:
            result["status"] = "validation_error"
            result["error"]  = resp.json().get("detail", "validation failed")
        elif resp.status_code == 404:
            result["status"] = "not_found"
            result["error"]  = "plate not found in database"
        else:
            result["status"] = "error"
            result["error"]  = f"HTTP {resp.status_code}"

    except requests.exceptions.Timeout:
        result["response_ms"] = int((time.time() - start) * 1000)
        result["status"]      = "timeout"
        result["error"]       = f"Request timed out after {timeout}s"
    except requests.exceptions.ConnectionError as e:
        result["response_ms"] = int((time.time() - start) * 1000)
        result["status"]      = "connection_error"
        result["error"]       = str(e)
    except Exception as e:
        result["response_ms"] = int((time.time() - start) * 1000)
        result["status"]      = "exception"
        result["error"]       = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONCURRENT WORKER
# ─────────────────────────────────────────────────────────────────────────────

def worker(
    worker_id: int,
    host: str,
    track: int,
    plates: List[str],
    rounds: int,
) -> List[Dict[str, Any]]:
    """
    Simulate one user making `rounds` requests through all plates.
    """
    results = []
    for r in range(rounds):
        for plate in plates:
            result = make_request(host, track, plate)
            result["worker_id"] = worker_id
            result["round"]     = r + 1
            results.append(result)
            # Small delay between requests to avoid hammering
            time.sleep(0.2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STATS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(results: List[Dict[str, Any]], track: int) -> Dict[str, Any]:
    """Compute summary statistics from a list of request results."""
    total      = len(results)
    successes  = [r for r in results if r["status"] == "success"]
    failures   = [r for r in results if r["status"] != "success"]
    llm_called = [r for r in successes if r["llm_called"]]
    llm_skip   = [r for r in successes if not r["llm_called"]]

    response_times = [r["response_ms"] for r in successes]
    llm_times      = [r["response_ms"] for r in llm_called]

    def percentile(data, p):
        if not data: return 0
        data_s = sorted(data)
        idx    = int(len(data_s) * p / 100)
        return data_s[min(idx, len(data_s) - 1)]

    # Label distribution
    label_dist = defaultdict(int)
    for r in successes:
        label_dist[r["label"]] += 1

    # Error distribution
    error_dist = defaultdict(int)
    for r in failures:
        error_dist[r["status"]] += 1

    return {
        "total":            total,
        "success":          len(successes),
        "failure":          len(failures),
        "success_rate_pct": round(len(successes) / total * 100, 1) if total > 0 else 0,
        "llm_called":       len(llm_called),
        "llm_skipped":      len(llm_skip),
        "response_avg_ms":  round(statistics.mean(response_times), 0) if response_times else 0,
        "response_p50_ms":  percentile(response_times, 50),
        "response_p95_ms":  percentile(response_times, 95),
        "response_p99_ms":  percentile(response_times, 99),
        "response_min_ms":  min(response_times) if response_times else 0,
        "response_max_ms":  max(response_times) if response_times else 0,
        "llm_avg_ms":       round(statistics.mean(llm_times), 0) if llm_times else 0,
        "llm_p95_ms":       percentile(llm_times, 95),
        "label_dist":       dict(label_dist),
        "error_dist":       dict(error_dist),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    stats: Dict[str, Any],
    track: int,
    users: int,
    rounds: int,
    duration_s: float,
) -> None:
    track_name = "Track 1 — Technician Fault Classification" if track == 1 \
                 else "Track 2 — Owner Risk Detection"

    rps = round(stats["success"] / duration_s, 2) if duration_s > 0 else 0

    print(f"\n{'='*60}")
    print(f"  STRESS TEST RESULTS — {track_name}")
    print(f"{'='*60}")
    print(f"  Config : {users} user(s) × {rounds} round(s) × {len(ALL_PLATES)} plates")
    print(f"  Duration : {duration_s:.1f}s")
    print()

    print(f"  ── Request Summary ──────────────────────────────")
    print(f"  Total requests   : {stats['total']}")
    print(f"  Successful       : {stats['success']}  ({stats['success_rate_pct']}%)")
    print(f"  Failed           : {stats['failure']}")
    print(f"  Throughput       : {rps} req/s")
    print()

    print(f"  ── LLM Call Analysis ────────────────────────────")
    print(f"  LLM called       : {stats['llm_called']}")
    print(f"  LLM skipped      : {stats['llm_skipped']}  (healthy vehicle)")
    llm_rate = round(stats['llm_called'] / stats['success'] * 100, 1) if stats['success'] > 0 else 0
    print(f"  LLM call rate    : {llm_rate}%")
    print()

    print(f"  ── Response Time (all successful) ───────────────")
    print(f"  Average          : {stats['response_avg_ms']:.0f} ms")
    print(f"  p50 (median)     : {stats['response_p50_ms']} ms")
    print(f"  p95              : {stats['response_p95_ms']} ms")
    print(f"  p99              : {stats['response_p99_ms']} ms")
    print(f"  Min              : {stats['response_min_ms']} ms")
    print(f"  Max              : {stats['response_max_ms']} ms")
    print()

    if stats['llm_avg_ms'] > 0:
        print(f"  ── LLM Response Time (calls that hit Gemini) ────")
        print(f"  Average          : {stats['llm_avg_ms']:.0f} ms")
        print(f"  p95              : {stats['llm_p95_ms']} ms")
        print()

    if stats['label_dist']:
        label_key = "Fault" if track == 1 else "Risk"
        print(f"  ── {label_key} Distribution ─────────────────────────")
        for label, count in sorted(stats['label_dist'].items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 30)
            print(f"  {label:<30} {count:>4}  {bar}")
        print()

    if stats['error_dist']:
        print(f"  ── Errors ───────────────────────────────────────")
        for err_type, count in stats['error_dist'].items():
            print(f"  {err_type:<25} {count}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def health_check(host: str) -> bool:
    """Verify the API is up before starting the stress test."""
    try:
        resp = requests.get(f"{host}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n  API Health: {resp.status_code} OK")
            print(f"  ML models  : {'✓' if data.get('track1_model') else '✗'}")
            print(f"  Vectorstore: {'✓' if data.get('vectorstore') else '✗'}")
            print(f"  LLM chains : {'✓' if data.get('llm_chains') else '✗'}")
            return True
        else:
            print(f"\n  API returned {resp.status_code} — aborting.")
            return False
    except Exception as e:
        print(f"\n  Cannot reach API at {host}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_test(host: str, track: int) -> None:
    """
    Test that invalid inputs are correctly rejected by the safety layer.
    These should return 404 or 422 — NOT 200.
    """
    print(f"\n  ── Validation / Safety Tests ────────────────────")
    for plate in INVALID_PLATES:
        result = make_request(host, track, plate, timeout=10)
        expected = result["status"] in ("validation_error", "not_found", "error")
        status   = "PASS ✓" if expected else "FAIL ✗"
        display  = repr(plate) if plate == "" else plate
        print(f"  {status}  '{display}' → {result['status']} ({result['status_code']})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PredictiveCare API Stress Test"
    )
    parser.add_argument(
        "--host",    default="http://localhost:8010",
        help="API base URL (default: http://localhost:8010)"
    )
    parser.add_argument(
        "--users",   type=int, default=3,
        help="Number of concurrent users (default: 3)"
    )
    parser.add_argument(
        "--rounds",  type=int, default=2,
        help="Rounds per user per track (default: 2)"
    )
    parser.add_argument(
        "--track",   type=int, choices=[1, 2, 0], default=0,
        help="Track to test: 1, 2, or 0 for both (default: 0 = both)"
    )
    parser.add_argument(
        "--timeout", type=int, default=90,
        help="Request timeout in seconds (default: 90)"
    )
    args = parser.parse_args()

    tracks = [1, 2] if args.track == 0 else [args.track]

    print("\n" + "=" * 60)
    print("  PredictiveCare — API Stress Test")
    print("=" * 60)
    print(f"  Host    : {args.host}")
    print(f"  Users   : {args.users}")
    print(f"  Rounds  : {args.rounds}")
    print(f"  Tracks  : {tracks}")
    print(f"  Plates  : {len(ALL_PLATES)} per track")
    print(f"  Timeout : {args.timeout}s per request")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Health check first
    if not health_check(args.host):
        sys.exit(1)

    for track in tracks:
        track_name = "Track 1 (Technician)" if track == 1 else "Track 2 (Owner)"
        print(f"\n{'─'*60}")
        print(f"  Starting {track_name} stress test ...")
        print(f"  {args.users} concurrent users × {args.rounds} rounds × {len(ALL_PLATES)} plates")
        print(f"  = {args.users * args.rounds * len(ALL_PLATES)} total requests")
        print(f"{'─'*60}")

        # Validation tests first (single threaded)
        run_validation_test(args.host, track)

        # Concurrent load test
        print(f"\n  Starting concurrent load test ...")
        start_time = time.time()
        all_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as executor:
            futures = [
                executor.submit(
                    worker,
                    worker_id=i,
                    host=args.host,
                    track=track,
                    plates=ALL_PLATES,
                    rounds=args.rounds,
                )
                for i in range(1, args.users + 1)
            ]

            # Progress indicator
            done = 0
            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                all_results.extend(results)
                done += 1
                print(f"  Worker {done}/{args.users} done — {len(results)} requests")

        duration = time.time() - start_time
        stats    = compute_stats(all_results, track)
        print_report(stats, track, args.users, args.rounds, duration)

    print("=" * 60)
    print("  Stress test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
