"""Operator CLI: `safety-assistant <command>` (also `python -m safety_assistant.cli`).

safety-assistant ingest [source_key ...] [--force] [--no-activate]
safety-assistant migrate
safety-assistant eval-retrieval [--legs full sparse ...]
safety-assistant ready            # exit 0 when readiness dependencies pass
safety-assistant worker [--poll-seconds 2] [--once]
safety-assistant users add --email E --name N --role engineer|knowledge_admin|auditor|org_admin [--workspace W]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def _ingest(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from safety_assistant.ingestion.sources import get_registry
    from safety_assistant.ingestion.workflows import ingest_source
    from safety_assistant.persistence import get_engine

    registry = get_registry(str(args.registry))
    failures = 0
    for key in args.source_keys or registry.keys():
        t0 = time.perf_counter()
        with Session(get_engine(), expire_on_commit=False) as session:
            out = ingest_source(
                session, key, registry=registry, force=args.force, activate=not args.no_activate, actor="cli"
            )
        line = {
            "source_key": key,
            "run_status": out.status,
            "version_status": out.final_version_status,
            "seconds": round(time.perf_counter() - t0, 1),
            **{k: v for k, v in out.stats.items() if k in ("pages", "chunks", "embed_new", "embed_reused")},
        }
        if out.error:
            line["error"] = out.error.splitlines()[0][:200]
            failures += 1
        print(json.dumps(line), flush=True)
    return 1 if failures else 0


def _migrate(args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    root = pathlib.Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "migrations" / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(cfg, args.revision)
    return 0


def _worker(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from safety_assistant.observability.logging import configure_logging
    from safety_assistant.persistence import get_engine
    from safety_assistant.workers.ingestion import run_forever, run_once

    configure_logging()
    if args.once:
        with Session(get_engine(), expire_on_commit=False) as session:
            job = run_once(session)
        print(json.dumps({"job_id": str(job.id), "status": job.status} if job else {"job_id": None}))
        return 0
    run_forever(poll_seconds=args.poll_seconds)
    return 0


def _users_add(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from safety_assistant.identity.service import create_user, create_workspace, default_organization, user_by_email
    from safety_assistant.persistence import get_engine

    with Session(get_engine()) as session:
        if user_by_email(session, args.email) is not None:
            print(f"user {args.email} already exists", file=sys.stderr)
            return 1
        user = create_user(session, email=args.email, display_name=args.name, role=args.role)
        if args.workspace:
            org = default_organization(session)
            create_workspace(session, organization_id=org.id, name=args.workspace, owner=user)
        session.commit()
        print(json.dumps({"user_id": str(user.id), "email": user.email, "role": args.role}))
    return 0


def _eval_retrieval(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from safety_assistant.evaluation import format_summary, load_dataset, run_evaluation, write_report
    from safety_assistant.persistence import get_engine

    dataset = load_dataset(pathlib.Path(args.dataset), source=args.source, query_types=args.types or None)
    with Session(get_engine()) as session:
        report = run_evaluation(session, dataset, legs=args.legs)
    path = write_report(report, pathlib.Path(args.out))
    print(format_summary(report))
    print(f"written: {path}")
    return 0


def _ready(args: argparse.Namespace) -> int:
    from safety_assistant.api.routes.health import _db_check, _embedding_check

    ok = True
    for name, fn in (("database", _db_check), ("embeddings", _embedding_check)):
        try:
            print(name, json.dumps(fn()))
        except Exception as exc:  # noqa: BLE001
            print(name, "FAIL", f"{type(exc).__name__}: {exc}")
            ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="safety-assistant", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("source_keys", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-activate", action="store_true")
    p.add_argument("--registry", default="knowledge/00_registry/sources.yaml")
    p.set_defaults(fn=_ingest)
    p = sub.add_parser("migrate")
    p.add_argument("revision", nargs="?", default="head")
    p.set_defaults(fn=_migrate)
    p = sub.add_parser("eval-retrieval")
    p.add_argument("--dataset", default="evals/datasets/regulatory_v2.yaml")
    p.add_argument("--legs", nargs="*", default=None)
    p.add_argument("--source", choices=["human", "llm_generated", "llm_generated_reviewed"], default=None)
    p.add_argument("--types", nargs="*", default=None, help="restrict to these query_type values")
    p.add_argument("--out", default="evals/results")
    p.set_defaults(fn=_eval_retrieval)
    p = sub.add_parser("ready")
    p.set_defaults(fn=_ready)
    p = sub.add_parser("worker")
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--once", action="store_true", help="process at most one job and exit")
    p.set_defaults(fn=_worker)
    users = sub.add_parser("users").add_subparsers(dest="users_cmd", required=True)
    p = users.add_parser("add")
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--role", required=True, choices=["engineer", "knowledge_admin", "auditor", "org_admin"])
    p.add_argument("--workspace", default=None, help="also create this workspace with the user as owner")
    p.set_defaults(fn=_users_add)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
