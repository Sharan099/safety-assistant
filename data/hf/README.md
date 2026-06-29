# HF Space corpus bundle

The root `Dockerfile` (Hugging Face Docker Space) copies **`safety_registry.db`** from this folder into the container at build time.

## Prepare locally

```powershell
# After ingest (or if safety_registry.db already exists at repo root):
python scripts/hf_export_db.py
```

## Push via Git LFS (required — file is ~580 MB)

```powershell
git lfs install
git lfs track data/hf/safety_registry.db
git add .gitattributes data/hf/safety_registry.db
git commit -m "Add HF corpus bundle (LFS)"
git push origin main
```

HF Space rebuilds automatically when `main` updates.

## HF Space secrets (Settings → Variables)

| Variable | Required |
|----------|----------|
| `GROQ_API_KEY` | Yes |
| `AUTH_SEED_PASSWORD` | Yes (login for engineer_a / engineer_b / lead) |
| `REGISTRY_CORS_ORIGINS` | Yes — e.g. `https://safety-assistant-tan.vercel.app` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `ENABLE_GATEWAY` | `true` |
| `ANTHROPIC_API_KEY` | Recommended (failover) |

`SKIP_WORKER_HEALTH`, `DISABLE_SCHEDULER`, and `DATABASE_URL` are set in the Dockerfile.
