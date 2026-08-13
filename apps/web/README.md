# apps/web

Next.js investigation workspace — UI_UX_DESIGN_BRIEF.md. An engineering
workstation (Dashboard, Runs, Investigation workspace, Knowledge search),
not a chatbot: the AI assistant is not built yet (Phase 15's LangGraph agent
is triggered from the "Run full agent" button, but there is no free-form
chat panel by design — see UI_UX_DESIGN_BRIEF.md Section 1/18).

## Ports

This host also runs a sibling project's servers on the common defaults
(`3000`, `8000` — see `docs/ADR/0004`), so this app uses:

- Frontend dev server: **3010** (`npm run dev`, already set in `package.json`)
- Backend API it talks to: **8010** (`NEXT_PUBLIC_API_URL`, see `.env.local.example`)

## Getting started

```powershell
# from the repo root, in a separate terminal:
docker compose up -d postgres
uv run alembic upgrade head
uv run uvicorn apps.api.main:app --port 8010

# then, in apps/web:
cp .env.local.example .env.local
npm install
npm run dev
```

Open http://localhost:3010.

## Structure

```
src/
├── app/
│   ├── page.tsx                    Dashboard
│   ├── runs/                       Run Browser + Run Identity
│   ├── investigations/new/         New Investigation form
│   ├── investigations/[id]/        Investigation workspace (the main screen)
│   └── knowledge/                  Knowledge search
├── components/                     StatusBadge, QualityCheckTable,
│                                    ComparabilityMatrix, SignalChart,
│                                    EvidenceCard, HypothesisCard, AppShell
└── lib/
    ├── api.ts                      Typed fetch client for apps/api
    └── types.ts                    Mirrors apps/api/schemas.py
```

## Design

- One dark theme (no light/dark toggle) — low visual noise, UI_UX_DESIGN_BRIEF.md Section 6.
- Every status uses a symbol + word, never color alone (Section 5) — see `StatusBadge`.
- TanStack Query for all server state; no client-side data duplication.
- ECharts (`echarts-for-react`) for the Signal Workspace time-history chart —
  the one charting library chosen per TRD.md Section 2 ("choose only one").

## Not built in V1

Mechanism/animation review (no animation data exists yet), report
generation, and a dedicated historical-cases browser (the underlying data
is legitimately empty until investigations have been closed — see
`packages/agent/tools.py`).
