# UNECE 403 diagnosis (UN_R11)

**Date:** 2026-06-27

## Test

```
GET https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R011r5e.pdf
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36
```

## Result

| Method | Status | Body |
|--------|--------|------|
| GET | **403** | HTML error page (`<!DOCTYPE...`, 5735 bytes) — **not** `%PDF-` |
| HEAD | **403** | empty |

## Verdict

**Network / UNECE WAF blocking this egress** — not a missing User-Agent. A standard Chrome UA still returns 403 HTML.

Crawler UA upgrade would not fix acquisition on this machine. Re-run `scripts/prep_missing_regs_fetch.py` from an open network.

No staging re-run performed (would repeat 403). No `storage/` changes.
