"""Load HF_TOKEN from .env and authenticate huggingface_hub (once per process)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_done = False


def ensure_hf_auth() -> bool:
    """Set Hub token env vars and login. Safe to call repeatedly."""
    global _done
    if _done:
        return bool(
            (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
        )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass

    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        _done = True
        logger.debug("No HF_TOKEN / HUGGING_FACE_HUB_TOKEN set — Hub calls stay anonymous")
        return False

    # huggingface_hub reads either name; keep both stripped.
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logger.info("Hugging Face Hub authenticated via HF_TOKEN")
    except Exception as exc:  # noqa: BLE001
        # Env vars alone are usually enough for hf_hub_download; login is best-effort.
        logger.warning("huggingface_hub.login failed (%s); using env token only", exc)

    _done = True
    return True
