from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session
from database.models import Test, TestAuditLog


def is_model_authorized(model_key: str, model_id: str) -> bool:
    """
    Checks if the model is authorized for the confidential tier.
    Only local Ollama, paid Groq, paid Anthropic, and paid OpenRouter Llama/Claude are authorized.
    Any free-tier or Gemini model is unauthorized (NOT free Gemini).
    """
    model_key_lower = str(model_key).lower()
    model_id_lower = str(model_id).lower()

    # Block any free-tier or Gemini models
    if "free" in model_key_lower or "free" in model_id_lower:
        return False
    if "gemini" in model_key_lower or "gemini" in model_id_lower:
        return False

    # Allow local ollama
    if "ollama" in model_key_lower or "local" in model_key_lower:
        return True

    # Allow paid Groq/Anthropic
    allowed_providers = ["groq", "anthropic"]
    for p in allowed_providers:
        if p in model_key_lower:
            return True

    # OpenRouter is VERIFY-BEFORE-USE (blocked by default until verified)
    if "openrouter" in model_key_lower:
        return False

    return False


def user_can_access_test(test: Test, user_id: str | None) -> bool:
    """Public harness rows are readable by anyone; confidential rows require matching owner."""
    if not test.confidential_tier:
        return True
    if not user_id:
        return False
    if not test.owner_user_id:
        return False
    return test.owner_user_id == user_id


def filter_tests_for_user(tests: list[Test], user_id: str | None) -> list[Test]:
    return [t for t in tests if user_can_access_test(t, user_id)]


def audit_access(db: Session, user_id: str, resource: str, action: str, model_used: str, details: str | None = None) -> None:
    """Log the access to the audit trail database."""
    log_entry = TestAuditLog(
        user_id=user_id,
        resource=resource,
        action=action,
        model_used=model_used,
        details=details
    )
    db.add(log_entry)
    db.commit()


def check_harness_access(db: Session, model_key: str, model_id: str, user_id: str, test_ids: list[str]) -> None:
    """
    Enforces authorized access to harness data.
    - Confidential tests: caller must own the row (server-verified user_id).
    - Model must be authorized for confidential tier.
    Throws HTTPException 401/403 on failure; logs all attempts.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required for harness data")

    authorized_model = is_model_authorized(model_key, model_id)
    model_status = "AUTHORIZED" if authorized_model else "UNAUTHORIZED"

    for test_id in test_ids:
        test = db.query(Test).filter(Test.test_id == test_id).first()
        owner_ok = user_can_access_test(test, user_id) if test else False
        if test and test.confidential_tier and not owner_ok:
            audit_access(
                db=db,
                user_id=user_id,
                resource=test_id,
                action="SELECT",
                model_used=f"{model_key}/{model_id}",
                details="Access classification: OWNER_DENIED",
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: confidential test '{test_id}' belongs to another user.",
            )

        audit_access(
            db=db,
            user_id=user_id,
            resource=test_id,
            action="SELECT",
            model_used=f"{model_key}/{model_id}",
            details=f"Access classification: {model_status}",
        )

    if not authorized_model:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Access Denied: The requested model '{model_key}' is not authorized for "
                f"confidential harness data (free/training tiers are blocked to prevent data leakage)."
            )
        )
