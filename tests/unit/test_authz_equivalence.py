"""The SQL predicate and the in-memory BM25 evaluator must agree on every (document, authz) case
(ADR-0029 §4). The SQL side is evaluated against an in-memory SQLite table with the same columns,
so the check needs no PostgreSQL."""

from __future__ import annotations

import itertools
import uuid

from sqlalchemy import Column, MetaData, String, Table, create_engine, select, text

from safety_assistant.persistence.models import Regulation
from safety_assistant.retrieval.authz import Authz, DocumentRef, anonymous_allows, anonymous_sql

ORG_A, ORG_B = uuid.uuid4(), uuid.uuid4()
WS_1, WS_2 = uuid.uuid4(), uuid.uuid4()
ALICE, BOB = uuid.uuid4(), uuid.uuid4()

DOCS = [
    DocumentRef(uuid.uuid4(), "AUTHORITATIVE_ORG", ORG_A, None, None),
    DocumentRef(uuid.uuid4(), "AUTHORITATIVE_ORG", ORG_B, None, None),
    DocumentRef(uuid.uuid4(), "WORKSPACE", ORG_A, WS_1, ALICE),
    DocumentRef(uuid.uuid4(), "WORKSPACE", ORG_A, WS_2, BOB),
    DocumentRef(uuid.uuid4(), "PRIVATE_USER", ORG_A, None, ALICE),
    DocumentRef(uuid.uuid4(), "PRIVATE_USER", ORG_A, None, BOB),
    DocumentRef(uuid.uuid4(), "PRIVATE_USER", ORG_B, None, ALICE),
    DocumentRef(uuid.uuid4(), "AUTHORITATIVE_ORG", ORG_A, None, None, archived=True),
]

SCOPE_SETS = [
    ("AUTHORITATIVE_ORG",),
    ("WORKSPACE",),
    ("PRIVATE_USER",),
    ("AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER"),
    ("WORKSPACE", "PRIVATE_USER"),
]
PRINCIPALS = [
    Authz(ALICE, (ORG_A,), (WS_1,)),
    Authz(BOB, (ORG_A,), (WS_2,)),
    Authz(ALICE, (ORG_A, ORG_B), (WS_1, WS_2)),
    Authz(ALICE, (), ()),  # user without memberships
    Authz(None, (ORG_A,), ()),  # organization-only principal
]


def _sqlite_docs():  # type: ignore[no-untyped-def]
    """A stand-in table with Regulation's predicate columns; UUIDs stored as 32-hex like the literal binds."""
    md = MetaData()
    t = Table(
        "regulations",
        md,
        Column("id", String, primary_key=True),
        Column("scope", String),
        Column("organization_id", String),
        Column("workspace_id", String),
        Column("owner_user_id", String),
        Column("archived_at", String),
    )
    eng = create_engine("sqlite://")
    md.create_all(eng)
    with eng.begin() as c:
        c.execute(
            t.insert(),
            [
                {
                    "id": d.document_id.hex,
                    "scope": d.scope,
                    "organization_id": d.organization_id.hex,
                    "workspace_id": d.workspace_id.hex if d.workspace_id else None,
                    "owner_user_id": d.owner_user_id.hex if d.owner_user_id else None,
                    "archived_at": "x" if d.archived else None,
                }
                for d in DOCS
            ],
        )
    return eng, t


def _sql_visible(eng, t, predicate) -> set[str]:  # type: ignore[no-untyped-def]
    # Rebind the ORM predicate (built on Regulation columns) to the stand-in table by compiling
    # it to a string with literal binds — the same expression text SQLite evaluates.
    from sqlalchemy.dialects import sqlite

    compiled = predicate.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True})
    sql = str(compiled).replace("regulations.", "r.")
    with eng.connect() as c:
        r = t.alias("r")
        rows = c.execute(select(r.c.id).where(text(sql))).all()
    return {r[0] for r in rows}


def test_sql_and_memory_evaluators_agree_on_every_case() -> None:
    eng, t = _sqlite_docs()
    cases = 0
    for base, scopes, doc_filter in itertools.product(PRINCIPALS, SCOPE_SETS, (False, True)):
        authz = Authz(
            base.user_id,
            base.organization_ids,
            base.workspace_ids,
            scopes,
            (DOCS[0].document_id, DOCS[4].document_id) if doc_filter else (),
        )
        memory = {d.document_id.hex for d in DOCS if authz.allows(d)}
        assert _sql_visible(eng, t, authz.sql()) == memory, (authz, scopes)
        cases += 1
    assert cases == len(PRINCIPALS) * len(SCOPE_SETS) * 2
    # no identity: authoritative only, across organizations
    memory = {d.document_id.hex for d in DOCS if anonymous_allows(d)}
    assert _sql_visible(eng, t, anonymous_sql()) == memory == {DOCS[0].document_id.hex, DOCS[1].document_id.hex}


def test_private_material_never_leaks_to_other_users_or_orgs() -> None:
    alice = Authz(ALICE, (ORG_A,), (WS_1,), ("AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER"))
    visible = [d for d in DOCS if alice.allows(d)]
    assert all(d.owner_user_id in (None, ALICE) for d in visible if d.scope == "PRIVATE_USER")
    assert all(d.workspace_id == WS_1 for d in visible if d.scope == "WORKSPACE")
    assert all(d.organization_id == ORG_A for d in visible if d.scope == "AUTHORITATIVE_ORG")
    # Alice's private doc in ORG_B is hers regardless of org membership: ownership is the key
    assert DOCS[6] in visible
    assert Regulation.scope is not None  # the ORM column the predicate is built on exists


def test_document_focus_unions_with_named_regulations() -> None:
    from safety_assistant.retrieval.base import ScopeFilter, document_in_scope

    report, r94 = uuid.uuid4(), uuid.uuid4()
    authz = Authz(user_id=uuid.uuid4(), organization_ids=(uuid.uuid4(),), workspace_ids=(), document_ids=(report,))
    focused = ScopeFilter(authz=authz)
    assert document_in_scope(focused, report, "ACME-TR-2026-0417")
    assert not document_in_scope(focused, r94, "UN-R94")  # focus alone: only the report
    named = ScopeFilter(authz=authz, regulation_keys=("UN-R94",))
    assert document_in_scope(named, report, "ACME-TR-2026-0417")  # the report stays
    assert document_in_scope(named, r94, "UN-R94")  # and the named regulation joins it
    assert not document_in_scope(named, uuid.uuid4(), "UN-R95")
    assert document_in_scope(ScopeFilter(authz=Authz(user_id=None, organization_ids=(), workspace_ids=())), r94, "X")
