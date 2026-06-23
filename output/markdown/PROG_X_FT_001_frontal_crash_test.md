---
source_type: synthetic
is_synthetic: true
vehicle_program: PROG_X
doc_type: internal
authority: internal
region: EU
dummy_type: Hybrid III
regulation: PROG_X_FT_001
source_pdf: PROG_X_FT_001_frontal_crash_test.md
---

> **SYNTHETIC TEST DATA — NOT A REAL TEST RESULT**
> Program: PROG_X | is_synthetic: true | vehicle_program: PROG_X

# PROG_X Frontal Crash Test Report — FT-PROG-X-001

## Test identification
| Field | Value |
| --- | --- |
| test_id | FT-PROG-X-001 |
| impact_mode | frontal |
| dummy_type | Hybrid III 50th percentile male |
| vehicle_program | PROG_X |
| date | 2025-11-14 |

## Configuration
ODB 40 % offset deformable barrier, 56 km/h. Driver and passenger three-point belts.

## Measurements (fabricated)
| Channel | Measured | Unit |
| --- | --- | --- |
| Upper anchorage resultant (driver) | 1,218 | daN |
| Lower anchorage resultant (driver) | 1,305 | daN |
| Belt webbing force (shoulder) | 4.2 | kN |
| HIC15 (driver) | 412 | — |
| Chest deflection (driver) | 38.4 | mm |

## Regulatory assessment
| Requirement | Cited clause | Limit | Measured | Result |
| --- | --- | --- | --- | --- |
| Anchorage strength M1/N1 | UN R14 §6.4.2 | 1,350 ± 20 daN | 1,305 daN (lower) | **PASS** |
| Anchorage strength M1/N1 | UN R14 §6.4.2 | 1,350 ± 20 daN | 1,218 daN (upper) | **PASS** |
| Belt assembly performance | UN R16 §7.7 (dynamic) | per approval | no separation | **PASS** |

## Observation
All anchorage loads remained below UN R14 Rev.7 limits. Belt routing per UN R16 geometry checks satisfied.
