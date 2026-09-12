---
name: database-change-review
description: Use before changing SQLAlchemy models, PostgreSQL schema, pgvector metadata, migrations, constraints, or data-retention relationships.
---

# Database Change Review

Before editing:
1. inspect current model/migration head;
2. classify additive/destructive;
3. identify existing-data impact;
4. specify forward migration;
5. specify rollback/compatibility;
6. identify indexes/constraints;
7. identify authorization-query impact.

Do not reset/squash migrations unless clean-baseline conditions are explicitly approved.

After:
- migrate clean DB;
- run integration tests;
- verify representative upgrade path;
- inspect query/index implications.
