---
name: security-review
description: Use for auth, RBAC, uploads, document scope, caching, provider routing, admin operations, external URLs, telemetry, and user/workspace data access changes.
---

# Security Review

Threat-model the exact change.

Check:
1. authentication;
2. authorization;
3. user/workspace/org scope;
4. input validation;
5. output exposure;
6. cache identity;
7. logs/telemetry;
8. secrets;
9. prompt/indirect injection;
10. SSRF/path/file risks;
11. rate/resource exhaustion;
12. audit events for privileged actions.

Required adversarial access tests:
- guessed UUID;
- other user;
- other workspace;
- insufficient role.

Never accept frontend-only authorization.
