---
source_type: synthetic
is_synthetic: true
vehicle_program: PROG_X
doc_type: internal
authority: internal
region: EU
dummy_type: Hybrid III
regulation: PROG_X_CAE_002
source_pdf: PROG_X_CAE_002_correlation.md
---

> **SYNTHETIC TEST DATA — NOT A REAL TEST RESULT**
> Program: PROG_X | is_synthetic: true | vehicle_program: PROG_X

# PROG_X CAE Correlation Report — CAE-PROG-X-002 vs FT-PROG-X-002

## Correlation summary
| Field | Value |
| --- | --- |
| CAE_model_version | PROG_X_v2.4.0_20260115 |
| reference_test | FT-PROG-X-002 |
| vehicle_program | PROG_X |

## Delta table (sim vs test)
| Metric | Test (FT-PROG-X-002) | CAE (v2.4.0) | Delta | % |
| --- | --- | --- | --- | --- |
| Lower anchorage load | 1,428 daN | 1,362 daN | −66 daN | −4.6 % |
| Upper anchorage load | 1,367 daN | 1,351 daN | −16 daN | −1.2 % |
| Chest deflection | 41.2 mm | 43.8 mm | +2.6 mm | +6.3 % |

## Conclusion
Lower anchorage under-predicted in CAE; stiffen B-pillar bracket mesh before sign-off.
