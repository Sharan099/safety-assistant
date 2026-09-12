---
name: pre-merge-gate
description: Run before declaring a rebuild phase or pull request complete; verifies tests, security, docs, cleanup, and diff scope.
---

# Pre-Merge Gate

1. inspect `git diff` and `git status`;
2. reject unrelated changes;
3. run relevant lint/type checks;
4. run focused tests;
5. run broader affected tests;
6. run retrieval regression if RAG changed;
7. run security tests if trust boundary changed;
8. run frontend build/Playwright if UI changed;
9. verify migrations if schema changed;
10. check secrets/generated junk;
11. update docs/contracts;
12. report checks not run separately from passing checks.

Never convert an unavailable check into a claimed pass.
