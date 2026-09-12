# ADR-0028 — Deployment: hardened container on managed services (ALB → ECS Fargate → RDS + S3), no Kubernetes

Status: accepted · Date: 2026-09-12 · Supersedes: 0004, 0010 (machine-specific setup)

## Decision
Multi-stage `uv` image, non-root uid 10001, read-only root filesystem, embedding model baked at build time, healthcheck; `infra/terraform` provisions an ALB (TLS 1.3), an ECS Fargate service with circuit-breaker rollback, RDS PostgreSQL 16 (forced TLS, encrypted, 14-day backups/PITR, multi-AZ in production), a versioned encrypted S3 bucket for artifacts, Secrets Manager for `DATABASE_URL`, and CloudWatch alarms. Development uses `docker compose` (pgvector) and, for the full stack, `infra/docker/compose.yaml` with MinIO.

## Alternatives rejected
Kubernetes: one stateless service plus two managed stores; no operational reason for a cluster.

## Status of evidence
The image build and scan run in CI (`ci.yml`, Trivy, SBOM). Terraform is structurally complete but **has not been applied** from this repository (no cloud account available) — recorded as a known limitation.
