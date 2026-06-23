---
source_type: synthetic
is_synthetic: true
vehicle_program: PROG_X
doc_type: internal
authority: internal
region: EU
dummy_type: Hybrid III
regulation: PROG_X_CAE_001
source_pdf: PROG_X_CAE_001_correlation.md
---

> **SYNTHETIC TEST DATA — NOT A REAL TEST RESULT**
> Program: PROG_X | is_synthetic: true | vehicle_program: PROG_X

# PROG_X CAE Correlation Report — CAE-PROG-X-001 vs FT-PROG-X-001

## Correlation summary
| Field | Value |
| --- | --- |
| CAE_model_version | PROG_X_v2.3.1_20251110 |
| reference_test | FT-PROG-X-001 |
| vehicle_program | PROG_X |

## Delta table (sim vs test)
| Metric | Test (FT-PROG-X-001) | CAE (v2.3.1) | Delta | % |
| --- | --- | --- | --- | --- |
| Lower anchorage load | 1,305 daN | 1,278 daN | −27 daN | −2.1 % |
| Upper anchorage load | 1,218 daN | 1,241 daN | +23 daN | +1.9 % |
| Chest deflection | 38.4 mm | 36.9 mm | −1.5 mm | −3.9 % |
| HIC15 | 412 | 398 | −14 | −3.4 % |

## Conclusion
Model within ±5 % on primary restraint metrics. Acceptable for design iteration per internal gate.
