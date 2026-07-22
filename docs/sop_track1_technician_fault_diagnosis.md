# Standard Operating Procedure — Track 1: Technician Fault Diagnosis
**Document ID:** SOP-T1-001  
**Version:** 1.0  
**Applicable Vehicle:** Compact SUV, Crossover, MPV (2020 onwards)  
**Detection Window:** 30-day OBD telemetry  
**Output:** Technician Fault Brief

---

## 1. Purpose

This SOP defines the procedure for interpreting ML anomaly detection results and generating a structured fault brief for workshop technicians. The system analyses 30 days of vehicle sensor telemetry, classifies fault type, retrieves relevant diagnostic context, and delivers a plain-language brief before the technician begins physical inspection.

---

## 2. Scope

This procedure applies to:
- authorised workshop technicians
- Service advisors conducting pre-inspection intake
- AI system generating automated fault briefs via LLM + RAG pipeline

---

## 3. Sensor Monitoring Scope (Track 1)

The following 12 sensors are monitored over the 30-day telemetry window:

| Sensor | Unit | Role in Diagnosis |
|---|---|---|
| O2 (Oxygen) Sensor | Volts (V) | Air-fuel ratio, emissions fault detection |
| MAF (Mass Air Flow) Sensor | g/s | Fuel delivery and air intake diagnosis |
| Throttle Position Sensor | % | Acceleration fault and injection timing |
| Crankshaft Position Sensor | RPM | Ignition timing and misfires |
| Camshaft Position Sensor | Degrees advance | Valve timing and injection synchronisation |
| Knock Sensor | Count (30-day) | Detonation events indicating fuel/ignition issues |
| Coolant Temperature Sensor | °C | Overheating and cooling system health |
| Oil Pressure Sensor | PSI | Lubrication system integrity |
| MAP (Manifold Absolute Pressure) Sensor | kPa | Intake pressure for fuel delivery (turbo models) |
| EGR Duty Sensor | % | Exhaust recirculation and emissions compliance |
| Battery Voltage Sensor | V | Charging system and battery health |
| Fuel Temperature Sensor | °C | Fuel system vapour lock risk |

---

## 4. Sensor Health Thresholds

| Sensor | Normal (No Action) | Warning (Monitor) | Critical (Inspect Now) |
|---|---|---|---|
| O2 Voltage | 0.10 – 0.50 V | 0.50 – 0.64 V | ≥ 0.65 V |
| MAF | 5.0 – 7.0 g/s | 4.0–4.9 or 7.5–9.0 g/s | < 3.5 or > 10.0 g/s |
| Throttle Position | 10% – 20% | < 5% or 20%–30% | > 30% (stuck) |
| Crankshaft RPM | 700 – 900 RPM | 900 – 1,100 RPM | > 1,100 or < 600 RPM |
| Cam Advance | 9° – 12° | 7°–9° or 12°–15° | < 6° or > 18° |
| Knock Count (30d) | 0 – 1 | 2 – 4 | > 5 |
| Coolant Temp | 85°C – 95°C | 95°C – 105°C | > 105°C |
| Oil Pressure | 35 – 50 PSI | 25 – 34 PSI | < 20 PSI |
| MAP | 30 – 40 kPa | 40 – 45 kPa | > 48 or < 20 kPa |
| EGR Duty | 15% – 25% | 25% – 35% | > 40% or < 5% |
| Battery Voltage | 13.8 – 14.5 V | 12.5 – 13.7 V | < 12.0 V |
| Fuel Temp | 25°C – 45°C | 45°C – 55°C | > 58°C |

---

## 5. Fault Classification Schema

The ML model classifies each 30-day telemetry window into one of 8 classes:

| Class ID | Fault Type | Priority |
|---|---|---|
| 0 | Normal | None — no action required |
| 1 | Battery Degradation | Medium — schedule within 14 days |
| 2 | Brake System Issue | High — inspect within 3 days |
| 3 | Cooling System Problem | High — inspect within 3 days |
| 4 | Engine Misfire | High — inspect within 3 days |
| 5 | Alternator Failure | Medium — schedule within 7 days |
| 6 | Oil Pressure Issue | Critical — do not drive, inspect immediately |
| 7 | Transmission Problem | High — inspect within 3 days |

---

## 6. Fault Diagnosis Procedures by Class

### 6.1 Class 1 — Battery Degradation

**Trigger signals:** Battery voltage consistently < 13.7 V during engine-on periods; voltage drops below 12.5 V overnight.

**Probable causes:**
- Battery cell degradation (age > 3 years)
- Parasitic drain from aftermarket accessories
- Early alternator output reduction

**Inspection steps:**
1. Load-test battery with 12V battery tester — check cold cranking amps (CCA) against rated spec.
2. Measure alternator output voltage at idle (should be 13.8–14.5 V) and at 2,000 RPM.
3. Check for parasitic drain: with ignition off, measure current draw (should be < 50 mA after 10 min).
4. Inspect battery terminals for corrosion and loose clamps.
5. Check charging cable condition between alternator and battery.

**Recommended action:** Replace battery if CCA < 80% of rated spec. If alternator output is low, proceed to Class 5 procedure.

---

### 6.2 Class 2 — Brake System Issue

**Trigger signals:** Elevated crankshaft RPM variance; abnormal deceleration patterns in speed sensor data; ABS wheel speed imbalance.

**Probable causes:**
- Brake pad wear beyond minimum thickness
- Brake fluid contamination or low level
- ABS sensor malfunction

**Inspection steps:**
1. Visually inspect brake pad thickness through wheel spokes — minimum 3 mm.
2. Check brake fluid level in reservoir — should be between MIN and MAX marks.
3. Test brake fluid boiling point if mileage exceeds 40,000 km since last change.
4. Inspect brake discs for scoring, warping, or rust beyond surface level.
5. Read ABS fault codes with diagnostic tool — check all four wheel speed sensors.
6. Test brake pedal firmness — spongy pedal indicates air in lines or failing master cylinder.

**Recommended action:** Replace pads if < 3 mm. Flush and replace brake fluid if contaminated. Replace faulty ABS sensor.

---

### 6.3 Class 3 — Cooling System Problem

**Trigger signals:** Coolant temperature sensor reading ≥ 98°C sustained over multiple readings; temperature spikes beyond 105°C.

**Probable causes:**
- Low coolant level (leak or evaporation)
- Thermostat stuck closed
- Radiator blockage or fan failure
- Water pump wear

**Inspection steps:**
1. Check coolant level in expansion tank — cold engine only. Should be between MIN and MAX.
2. Inspect coolant colour — brown/rusty indicates contamination; flush required.
3. Run engine to operating temp and observe radiator fan activation — should engage at ~95°C.
4. Pressure-test cooling system for leaks (cap, hoses, radiator, water pump seal).
5. Check thermostat operation — remove and test in hot water; should open at 82–88°C.
6. Inspect water pump drive belt or chain for wear.

**Recommended action:** Top up and pressure-test for leaks. Replace thermostat if stuck. Replace water pump if bearing noise present or seal leaking.

---

### 6.4 Class 4 — Engine Misfire

**Trigger signals:** O2 sensor voltage > 0.65 V (rich mixture); elevated knock count; crankshaft RPM fluctuation at idle; cam advance deviation > 15°.

**Probable causes:**
- Spark plug wear or fouling
- Ignition coil failure
- Fuel injector clogging or leaking
- Camshaft/crankshaft timing error

**Inspection steps:**
1. Read engine fault codes — note specific cylinder misfires (P0300–P0308).
2. Inspect spark plugs — check electrode gap, carbon fouling, oil contamination.
3. Swap ignition coils between cylinders to confirm/eliminate coil failure.
4. Check fuel injector spray pattern with injector tester or scope.
5. Perform compression test on all cylinders — should be > 1,200 kPa, within 10% of each other.
6. Verify cam and crank timing marks align with manufacturer specification.

**Recommended action:** Replace fouled or worn spark plugs. Replace failed ignition coil. Clean or replace clogged injectors. Correct timing if deviated.

---

### 6.5 Class 5 — Alternator Failure

**Trigger signals:** Battery voltage < 13.5 V with engine running; gradual voltage decline over 30-day window; battery warning light history.

**Probable causes:**
- Worn alternator brushes or slip rings
- Failed rectifier diode
- Loose or worn drive belt
- Failed voltage regulator

**Inspection steps:**
1. Measure alternator output voltage at idle and 2,000 RPM — should be 13.8–14.5 V.
2. Perform alternator ripple test with oscilloscope — excessive AC ripple indicates diode failure.
3. Inspect drive belt tension and condition — check for cracking, glazing, or slipping.
4. Check wiring harness to alternator — inspect for fraying, loose connectors, or corrosion.
5. Load-test alternator output with electrical load applied (headlights, AC, fan at max).

**Recommended action:** Replace alternator if output voltage < 13.5 V under load or ripple is excessive. Replace drive belt if worn.

---

### 6.6 Class 6 — Oil Pressure Issue

**Trigger signals:** Oil pressure sensor reading < 25 PSI at idle; pressure drops below 20 PSI under load. **This is a critical fault — continued operation risks engine seizure.**

**Probable causes:**
- Low oil level (leak or burn-off)
- Oil pump wear or failure
- Oil pressure sensor malfunction
- Sludge blockage in oil passages
- Main or rod bearing wear

**Inspection steps:**
1. **Stop engine immediately if oil pressure warning light is on.**
2. Check oil level with dipstick — cold engine, vehicle on level surface.
3. Inspect underside for oil leaks — sump gasket, rocker cover, oil filter, drain plug.
4. Confirm sensor accuracy — fit a mechanical oil pressure gauge to cross-check reading.
5. If mechanical pressure also low: drain and inspect oil — check for metallic particles.
6. Perform engine oil analysis if available — check for bearing metal contamination.
7. Check oil viscosity specification — wrong grade oil can cause low pressure in hot conditions.

**Recommended action:** Top up oil immediately if low. If pressure remains low with correct oil level, do not allow vehicle to be driven. Inspect oil pump and bearings.

---

### 6.7 Class 7 — Transmission Problem

**Trigger signals:** Crankshaft RPM spikes disproportionate to vehicle speed; throttle position > 30% with no corresponding acceleration; RPM flare during gear changes.

**Probable causes:**
- Low or contaminated transmission fluid
- Worn clutch pack or bands (automatic)
- Solenoid valve failure
- Torque converter wear

**Inspection steps:**
1. Check transmission fluid level and condition — should be red/pink, not brown or burnt-smelling.
2. Read transmission fault codes with diagnostic tool.
3. Perform stall speed test (automatic) — compare result against manufacturer specification.
4. Check for slipping during controlled acceleration in each gear.
5. Inspect transmission fluid for metallic particles.
6. Check transmission mounts — worn mounts can cause vibration misread as slip.

**Recommended action:** Flush and replace transmission fluid if contaminated. Replace faulty solenoid. Rebuild or replace transmission if clutch pack is worn.

---

## 7. Fault Brief Output Format

The LLM generates a structured Technician Fault Brief containing the following fields:

```
TECHNICIAN FAULT BRIEF
======================
Vehicle       : [Make / Model / Year]
Fault Class   : [Class ID] — [Fault Type]
Priority      : [None / Medium / High / Critical]
Detection     : 30-day telemetry anomaly

SENSOR FINDINGS
---------------
[List of sensors outside normal range with actual vs. expected values]

PROBABLE CAUSE
--------------
[Top 1–3 probable causes ranked by likelihood based on sensor pattern]

INSPECTION CHECKLIST
--------------------
[Ordered steps from this SOP, numbered 1 to N]

RECOMMENDED ACTION
------------------
[Clear action statement from this SOP]

SOURCE
------
[Document and section cited from RAG retrieval]
```

---

## 8. Escalation Criteria

Escalate to senior technician or service manager if:
- Class 6 (Oil Pressure Issue) confirmed with mechanical gauge < 15 PSI
- Multiple fault classes detected simultaneously (e.g., Class 3 + Class 6)
- Customer reports vehicle stalling, smoke, or burning smell
- Post-repair telemetry shows fault class persisting after 7 days

---

## 9. Documentation Requirements

After completing inspection and repair:
- Log fault class, sensor readings at intake, and repair action in MMID service history
- Record part numbers replaced and mileage at time of repair
- Upload post-repair sensor snapshot to central database within 24 hours of vehicle collection

---

*SOP-T1-001 | Vehicle Predictive Maintenance System | Version 1.0 | May 2026*
