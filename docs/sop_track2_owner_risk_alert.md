# Standard Operating Procedure, Track 2: Vehicle Owner Risk Alert

**Document ID:** SOP-T2-001  
**Version:** 1.0  
**Applicable Vehicle:** Compact SUV, Crossover, MPV (2020 onwards)  
**Detection Window:** 12-hour daily window (07:00 – 19:00 WIB)  
**Output:** Push Alert in English (plain language, urgency + action)

---

## 1. Tujuan (Purpose)

This SOP defines the procedure for processing daily vehicle sensor data, determining the risk level, and generating an easy-to-understand push alert to the vehicle owner in English. The system runs automatically every day based on IoT data collected throughout the day.

This SOP defines the procedure for processing daily vehicle sensor data, determining risk level, and generating push alert notifications to vehicle owners in plain English. The system operates automatically each day based on IoT data collected throughout the operating window.

---

## 2. Lingkup (Scope)

This SOP applies to:

- Private vehicle owners
- Commercial fleet operators
- The AI system that generates automatic notifications through the Owner Alert Module

This procedure applies to:

- Private vehicle owners
- Commercial fleet operators
- AI system generating automated notifications via the Owner Alert Module

---

## 3. Sensor Monitoring Scope (Track 2)

The following 12 sensors are monitored across the 12-hour daily window (07:00–19:00):

| Sensor                     | Unit      | Monitored For                        |
| -------------------------- | --------- | ------------------------------------ |
| O2 (Oxygen) Sensor         | Volts (V) | Fuel mixture imbalance               |
| MAF (Mass Air Flow) Sensor | g/s       | Fuel efficiency degradation          |
| Throttle Position Sensor   | %         | Driving pattern anomalies            |
| Coolant Temperature Sensor | °C        | Overheating risk                     |
| Oil Pressure Sensor        | PSI       | Lubrication risk                     |
| Battery Voltage Sensor     | V         | Electrical system health             |
| TPMS (Tyre Pressure)       | PSI       | Tyre safety and blowout risk         |
| Ambient Temperature Sensor | °C        | Environmental heat stress on vehicle |
| Cabin Humidity Sensor      | %         | Climate system health                |
| Fuel Level Sensor          | %         | Low fuel risk                        |
| Brake Pedal Events         | Count     | Brake wear indicator                 |
| Vehicle Speed Sensor       | km/h      | Driving behaviour risk               |

---

## 4. Sensor Health Thresholds (Owner Perspective)

| Sensor             | Normal        | Watch                   | Danger          |
| ------------------ | ------------- | ----------------------- | --------------- |
| TPMS               | 30 – 34 PSI   | 27 – 29 PSI             | < 25 PSI        |
| Coolant Temp       | 85°C – 95°C   | 95°C – 105°C            | > 105°C         |
| Oil Pressure       | 35 – 50 PSI   | 25 – 34 PSI             | < 20 PSI        |
| Battery Voltage    | 13.8 – 14.5 V | 12.5 – 13.7 V           | < 12.0 V        |
| Fuel Level         | 25% – 100%    | 10% – 24%               | < 10%           |
| Ambient Temp       | 20°C – 35°C   | 35°C – 38°C             | > 40°C          |
| Cabin Humidity     | 40% – 65%     | 65% – 75%               | > 80%           |
| Brake Pedal Events | 5 – 20        | 20 – 35                 | > 40            |
| Avg Speed          | 20 – 60 km/h  | 60 – 90 km/h            | > 100 km/h      |

---

## 5. Risk Classification Schema

The Owner Alert Module classifies daily sensor data into one of 4 risk levels:

| Class | Risk Level  | Notification Trigger                   |
| ----- | ----------- | -------------------------------------- |
| 0     | No Risk     | No notification sent                   |
| 1     | Low Risk    | Monitor, informational alert           |
| 2     | Medium Risk | Schedule maintenance, action required  |
| 3     | High Risk   | Immediate inspection, urgent alert     |

### Risk Level Logic

The higher the number of sensors deviating from normal AND the further each deviation, the higher the risk class assigned.

**Class 0: No Risk**

- All sensors within normal operating range
- Typical readings: Battery ~14.2V, Coolant ~90°C, Oil Pressure ~40 PSI, TPMS ~32 PSI
- No notification sent to owner

**Class 1: Low Risk**

- One or two sensors showing minor deviation
- Example: Battery 13.6V (early alternator wear signal), TPMS 29.5 PSI (slightly underinflated)
- Vehicle still driveable; owner should monitor over next 3–5 days

**Class 2: Medium Risk**

- Multiple sensors showing notable deviation in combination
- Example: Coolant ~98°C + Oil Pressure ~30 PSI + Battery ~12.8V + TPMS ~27 PSI
- No single reading is catastrophic, but the combined pattern requires a workshop visit within 7 days

**Class 3: High Risk**

- Multiple sensors in danger territory simultaneously
- Example: Coolant ~112°C (overheating) + Oil Pressure ~18 PSI (near seizure) + Battery ~11.2V
- Continuing to drive risks serious mechanical damage or a safety incident
- Owner must stop driving and contact workshop immediately

---

## 6. Alert Generation Procedure

### Step 1: Data Collection

IoT system collects sensor readings every 1–2 hours between 07:00 and 19:00 WIB.
Up to 12 data points per sensor per day are aggregated into a daily summary report.

### Step 2: Risk Classification

The Owner Alert Module processes the daily summary through the ML risk classification model.
Output: Risk Class 0, 1, 2, or 3 for that vehicle on that day.

### Step 3: Alert Decision

- If **Class 0**: No alert sent. System logs "no risk" status.
- If **Class 1, 2, or 3**: Alert generation is triggered.

### Step 4: LLM Summarisation

The LLM Summarizer receives:

- The risk class and its label
- The specific sensor readings that triggered the classification
- Relevant SOP context retrieved from the RAG pipeline
- Instruction to output in plain English

### Step 5: Push Alert Delivery

Alert is sent to owner's registered mobile device with:

- Risk level indicator (emoji + label)
- Plain-language explanation of what was detected
- Clear urgency statement
- Specific recommended action
- No raw sensor values or technical codes visible to owner

---

## 7. Push Alert Templates by Risk Class

### Class 1: Low Risk

```
🟡 YOUR VEHICLE HEALTH CHECK

The system detected the following condition in your vehicle today:
[Description of the condition in plain language]

The vehicle is still safe to drive, but this condition should be monitored
over the next 3-5 days.

ACTION: Monitor the vehicle. If the condition does not improve,
schedule a service at the nearest authorised workshop soon.
```

### Class 2: Medium Risk

```
🟠 ATTENTION: SCHEDULE A VEHICLE SERVICE

The system detected a combination of conditions that need prompt attention:
[Description of the condition in plain language]

The vehicle can still be driven for short distances, but preferably
not for long trips before it is serviced.

ACTION: Schedule a service at an authorised workshop within 7 days.
Call emergency service: +62-800-000-0000
```

### Class 3: High Risk

```
🔴 IMPORTANT WARNING: IMMEDIATE INSPECTION REQUIRED

The system detected a critical condition in your vehicle:
[Description of the specific condition in plain language]

DO NOT DRIVE THE VEHICLE until it has been inspected by a technician.
Continuing to drive may cause serious engine damage
or put your safety at risk.

IMMEDIATE ACTIONS:
1. Stop the vehicle in a safe place
2. Turn off the engine
3. Call emergency service: +62-800-000-0000
4. Do not start the vehicle until a technician has inspected it
```

---

## 8. Alert Content Guidelines

The LLM must follow these rules when generating alert content:

### Must Do

- Use simple language that a non-technical vehicle owner can understand
- Include a clear urgency level (monitor / schedule / immediate)
- Give a concrete action the owner can take
- Refer to the affected component or system in everyday terms (e.g. "the cooling system" not "coolant temperature sensor")
- Match the tone to the risk level: informative for Class 1, firm for Class 3

### Must Not Do

- Do not show raw sensor values (e.g. "Coolant: 112°C")
- Do not use OBD error codes
- Do not name technical sensors directly
- Do not overly alarm the owner for Class 1
- Do not understate the urgency for Class 3

### Plain Language Translations for Owner Alerts

| Technical Term              | Owner-Facing Language (English)                             |
| --------------------------- | ----------------------------------------------------------- |
| Coolant temperature high    | The engine is starting to overheat beyond the normal limit           |
| Oil pressure low            | Engine oil pressure is dropping, risking engine damage          |
| Battery voltage degrading   | The vehicle battery is weakening                              |
| TPMS below threshold        | Tire pressure is below standard, risking a flat or blowout |
| Brake pedal events elevated | The braking system is working harder than usual                |
| Fuel level critical         | Fuel is almost empty                                    |
| Alternator voltage low      | The vehicle's charging system has a problem                  |

---

## 9. Daily Monitoring Schedule

| Time          | Action                                            |
| ------------- | ------------------------------------------------- |
| 07:00         | IoT collection window opens                       |
| 07:00 – 19:00 | Sensor readings captured every 1–2 hours          |
| 19:00         | Collection window closes; daily summary compiled  |
| 19:15         | Risk classification model processes daily summary |
| 19:30         | LLM generates alerts for Class 1, 2, 3 vehicles   |
| 20:00         | Push alerts delivered to owner devices            |

---

## 10. Escalation to Track 1

If a vehicle receives Class 3 risk classification on two or more consecutive days, the case is automatically escalated to Track 1 (Workshop / Mechanic). The system:

1. Flags the vehicle for 30-day telemetry analysis
2. Generates a Technician Fault Brief via the Track 1 RAG pipeline
3. Sends an alert to the nearest authorised workshop
4. Notifies the owner that a workshop has been informed

---

## 11. No-Alert Scenarios (Class 0)

No push notification is sent when all sensors are within normal range. This is a deliberate design choice: alert fatigue reduces owner responsiveness. Silence means the vehicle is healthy. Owners can view their vehicle health history at any time in the owner portal.

---

## 12. Privacy and Data Handling

- Sensor data is stored per-vehicle with anonymised vehicle ID
- Owner personal data (name, contact) is stored separately from telemetry data
- Alert content references vehicle condition only, with no location or personal data included
- Data retention: 90 days for raw telemetry, 12 months for risk classification logs

---

_SOP-T2-001 | Vehicle Predictive Maintenance System | Version 1.0 | May 2026_
