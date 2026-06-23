---
source_type: synthetic
is_synthetic: true
vehicle_program: PROG_X
doc_type: internal
authority: internal
region: EU
dummy_type: Hybrid III
regulation: PROG_X_FT_002
source_pdf: PROG_X_FT_002_frontal_crash_test.md
---

> **SYNTHETIC TEST DATA — NOT A REAL TEST RESULT**
> Program: PROG_X | is_synthetic: true | vehicle_program: PROG_X

# PROG_X Frontal Crash Test Report — FT-PROG-X-002

## Test identification
| Field | Value |
| --- | --- |
| test_id | FT-PROG-X-002 |
| impact_mode | frontal |
| dummy_type | Hybrid III 50th percentile male |
| vehicle_program | PROG_X |
| date | 2026-01-22 |

## Configuration
Full-width rigid barrier, 50 km/h, driver only, pretensioner + load limiter active.

## Measurements (fabricated)
| Channel | Measured | Unit |
| --- | --- | --- |
| Lower anchorage resultant | 1,428 | daN |
| Upper anchorage resultant | 1,367 | daN |
| Lap belt force peak | 5.1 | kN |
| Chest deflection | 41.2 | mm |
| Femur force (left) | 3.8 | kN |

## Regulatory assessment
| Requirement | Cited clause | Limit | Measured | Result |
| --- | --- | --- | --- | --- |
| Lower anchorage M1 | UN R14 §6.4.1 | 1,350 ± 20 daN | 1,428 daN | **MARGINAL — within tolerance band** |
| Upper anchorage M1 | UN R14 §6.4.2 | 1,350 ± 20 daN | 1,367 daN | **PASS** |
| Dynamic belt test | UN R16 §7.7 | no release | no release | **PASS** |

## Observation
Lower anchorage approached upper tolerance; review bracket stiffness for next build.
