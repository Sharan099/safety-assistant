# Ingestion coverage audit

Total chunks: **12805**
PDFs on disk: **14**

| Regulation | Chunks | CONTEXT % | APPLICABILITY % | Clause % | Tier |
|------------|--------|-----------|-----------------|----------|------|
| CAE Companion | 888 | 100.0 | 0.0 | 11.5 | engineering_ref |
| Euro NCAP | 464 | 100.0 | 0.0 | 56.7 | rating_protocol |
| FMVSS | 5931 | 100.0 | 0.0 | 7.6 | legal_binding |
| NCAP 2023 Nissan Altima 4 DR AWD | 1 | 0.0 | 0.0 | 0.0 | historical_data |
| NCAP 2024 Hyundai Elantra 4 DR FWD | 1 | 0.0 | 0.0 | 0.0 | historical_data |
| NCAP 2024 Toyota CAMRY 4 DR AWD | 1 | 0.0 | 0.0 | 0.0 | historical_data |
| PROG_X CAE-001 | 2 | 0.0 | 0.0 | 0.0 | synthetic |
| PROG_X CAE-002 | 2 | 0.0 | 0.0 | 0.0 | synthetic |
| PROG_X DR minutes | 1 | 0.0 | 0.0 | 0.0 | synthetic |
| PROG_X FT-001 | 5 | 0.0 | 0.0 | 0.0 | synthetic |
| PROG_X FT-002 | 5 | 0.0 | 0.0 | 0.0 | synthetic |
| PROG_X RCA-001 | 1 | 0.0 | 0.0 | 0.0 | synthetic |
| PROG_X status | 2 | 0.0 | 0.0 | 0.0 | synthetic |
| Safety Companion | 1 | 0.0 | 0.0 | 0.0 | engineering_ref |
| UN R127 | 24 | 100.0 | 0.0 | 0.0 | legal_binding |
| UN R135 | 479 | 100.0 | 0.0 | 60.8 | legal_binding |
| UN R137 | 585 | 100.0 | 0.0 | 61.0 | legal_binding |
| UN R14 | 492 | 94.7 | 0.0 | 68.5 | legal_binding |
| UN R16 | 1371 | 100.0 | 0.0 | 65.9 | legal_binding |
| UN R17 | 917 | 100.0 | 0.0 | 67.0 | legal_binding |
| UN R94 | 758 | 100.0 | 0.0 | 57.8 | legal_binding |
| UN R95 | 874 | 100.0 | 0.0 | 86.3 | legal_binding |

## Gaps

- NCAP_2023_NISSAN_ALTIMA_18390: only 0.0% chunks have CONTEXT annotation
- NCAP_2024_HYUNDAI_ELANTRA_19543: only 0.0% chunks have CONTEXT annotation
- NCAP_2024_TOYOTA_CAMRY_19427: only 0.0% chunks have CONTEXT annotation
- PROG_X_CAE_001: only 0.0% chunks have CONTEXT annotation
- PROG_X_CAE_002: only 0.0% chunks have CONTEXT annotation
- PROG_X_DR: only 0.0% chunks have CONTEXT annotation
- PROG_X_FT_001: only 0.0% chunks have CONTEXT annotation
- PROG_X_FT_002: only 0.0% chunks have CONTEXT annotation
- PROG_X_RCA_001: only 0.0% chunks have CONTEXT annotation
- PROG_X_STATUS: only 0.0% chunks have CONTEXT annotation
- SAFETY_REFERENCE: only 0.0% chunks have CONTEXT annotation

Full JSON: `H:\AutoSafety_RAG\output\ingestion_audit_report.json`