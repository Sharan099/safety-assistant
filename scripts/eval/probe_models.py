"""Probe an OpenAI-compatible gateway's models with the product's grounded prompt: latency, HTTP
status, JSON-schema compliance and whether the answer cites the right evidence id. Read-only,
no database. Used to choose LLM_MODEL / LLM_FALLBACK_MODEL among free routes.

    OPENROUTER=1 uv run python scripts/eval/probe_models.py --base-url https://openrouter.ai/api/v1 \
        --api-key-env OPEN_ROUTER --models google/gemma-4-31b-it:free nvidia/nemotron-3.5-lightning:free
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import sys
import time

from safety_assistant.generation.prompts.grounded_v1 import SYSTEM
from safety_assistant.generation.schemas import GroundedDraft
from safety_assistant.providers.llm import LLMError, LLMMessage
from safety_assistant.providers.llm.openai_compatible import OpenAICompatibleProvider

USER = """<scope>intent=lookup; route=standard; regulations=UN-R94; as_of=current</scope>

<evidence id="E1" regulation="UN-R94" version="Rev.4 (04 series)" status="ACTIVE" section="5.2.1.4" pages="12-13" valid_from="2021-06-09" valid_to="open" normative="True">
UN R94 › 5 Specifications › 5.2.1.4
5.2.1.4. The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;
</evidence>

<evidence id="E2" regulation="UN-R95" version="Rev.3 (04 series)" status="ACTIVE" section="5.2.1.2" pages="10-11" valid_from="2021-01-03" valid_to="open" normative="True">
UN R95 › 5 Specifications › 5.2.1.2
5.2.1.2. The thorax performance criteria shall be: (a) Rib Deflection Criterion (RDC) less than or equal to 42 mm;
</evidence>

<question>
What is the ThCC limit in the frontal impact test?
</question>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)
    key = os.environ.get(args.api_key_env, "").strip()
    print(f"{'model':<52}{'status':<10}{'ms':>7}  {'json':<6}{'cites E1':<9}answer")
    for m in args.models:
        p = OpenAICompatibleProvider(base_url=args.base_url, api_key=key, model=m, timeout=60)
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            try:
                r = p.generate(
                    [LLMMessage(role="system", content=SYSTEM), LLMMessage(role="user", content=USER)],
                    schema=GroundedDraft,
                    max_tokens=600,
                )
                d: GroundedDraft = r.parsed  # type: ignore[assignment]
                cites = any("E1" in c.evidence_ids for c in d.claims)
                print(
                    f"{m:<52}{'ok':<10}{(time.perf_counter() - t0) * 1000:7.0f}  {'yes':<6}{str(cites):<9}{d.answer[:70]!r}"
                )
            except LLMError as exc:
                kind = type(exc).__name__
                print(
                    f"{m:<52}{kind:<10}{(time.perf_counter() - t0) * 1000:7.0f}  {'no' if 'Schema' in kind else '-':<6}{'-':<9}{str(exc)[:70]!r}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
