"""
pipeline.py — Passive Safety Regulation GraphRAG Pipeline

Production-ready end-to-end pipeline for:
- UN Regulations
- FMVSS
- Euro NCAP
- ISO Standards
- Occupant Protection
- Crashworthiness
- Homologation Engineering

Features:
- Hybrid GraphRAG retrieval
- Neo4j knowledge graph
- Regulation-aware prompting
- Cerebras/OpenAI-compatible LLM pipeline
- Thread-safe singleton pipeline
- Graph visualization support
- Hybrid semantic retrieval

Author:
Sharan — Passive Safety GraphRAG
"""

import threading
import time

from typing import Optional

from loguru import logger

# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

from graph.builder import (
    RegulationGraphBuilder
)

# ─────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────

from retrieval.retrieval import (
    HybridRetriever
)

# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

from llm.GroqClient import (
    GroqClient
    
)

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

class SafetyGraphPipeline:

    def __init__(self):

        logger.info(
            "Initialising Passive Safety GraphRAG..."
        )

        self.graph = None

        self.retriever = None

        self.llm = None

        try:

            # graph
            self.graph = (
                RegulationGraphBuilder()
            )

            logger.info(
                "Graph builder initialised"
            )

            # retriever
            self.retriever = (
                HybridRetriever(
                    self.graph
                )
            )

            logger.info(
                "Retriever initialised"
            )

            # llm
            self.llm = (
                GroqClient()
            )

            logger.info(
                "Groq client initialised"
            )

        except Exception:

            logger.exception(
                "Pipeline initialisation failed"
            )

            self.close()

            raise

        logger.info(
            "Passive Safety GraphRAG ready"
        )

    # ─────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────
    def build_prompt(

        self,

        query: str,

        context: str

    ) -> str:

        return f"""
You are PSA AI,
an expert passive safety and homologation engineering assistant.

You specialize in:
- UN R14
- UN R16
- occupant restraint systems
- crashworthiness
- seat belt anchorages
- crash calculations
- impact physics
- injury biomechanics
- restraint engineering

==================================================
IMPORTANT RESPONSE RULES
==================================================

1. ALWAYS use professional markdown formatting.

2. NEVER write huge paragraphs.

3. ALWAYS structure answers EXACTLY like this:

# Title

## Summary

Short explanation.

## Key Requirements / Results

- Point 1
- Point 2
- Point 3

## Calculations (ONLY if needed)

### Formula

Write the engineering formula.

### Inputs

- Parameter = value

### Step-by-Step Solution

Show calculations clearly.

### Final Result

- Final value with units

## Engineering Notes

- Design implications
- Safety meaning
- Engineering interpretation

## References

- UN R14 Paragraph X
- UN R16 Paragraph Y

==================================================
CALCULATION RULES
==================================================

If the user asks:
- calculate
- compute
- estimate
- derive
- force
- energy
- acceleration
- belt load
- stopping distance
- crash pulse
- speed conversion
- kinetic energy

THEN:

1. Solve mathematically step-by-step.

When solving calculations:
- think step-by-step,
- verify units,
- verify dimensional consistency,
- explain engineering significance.

2. Use engineering formulas.

3. ALWAYS include:
   - formulas
   - substitutions
   - units
   - final numeric result

4. DO NOT say:
   "consult an engineer"

5. Perform calculations directly.

6. Use SI units.

==================================================
FORMATTING RULES
==================================================

1. NEVER output:
   ### 3.1
   broken numbering
   corrupted references

2. ALWAYS convert:
   5. ### 3.1
   → 5.3.1

3. Use bullet points heavily.

4. Put references BELOW each requirement.

5. Make answers visually easy to scan.

==================================================
USER QUESTION
==================================================

{query}

==================================================
RETRIEVED REGULATION CONTEXT
==================================================

{context}

==================================================
ANSWER
==================================================
"""

    # ─────────────────────────────
    # QUERY
    # ─────────────────────────────

    def query(

        self,

        user_query: str
    ) -> dict:

        t0 = time.time()

        try:

            # retrieve context
            result = self.retriever.retrieve(
                user_query
            )

            context = result.get(
                "context",
                ""
            ).strip()

            entities = result.get(
                "entities",
                []
            )

            graph_viz = result.get(
                "graph_viz",
                {}
            )

            # no context
            if not context:

                return {

                    "answer":

                        "No relevant passive safety "
                        "regulation information found.",

                    "entities":
                        entities,

                    "graph_viz":
                        graph_viz,

                    "timing": {

                        **result.get(
                            "timing",
                            {}
                        ),

                        "total_ms":

                            round(

                                (
                                    time.time()
                                    -
                                    t0
                                ) * 1000,

                                1
                            )
                    }
                }

            # build prompt
            prompt = self.build_prompt(

                user_query,

                context
            )

            # LLM generation
            llm_start = time.time()

            raw_answer = self.llm.generate(
                prompt
            )

            answer = (

                (raw_answer or "")
                .strip()

                or

                "Model returned an empty response."
            )

            timing = {

                **result.get(
                    "timing",
                    {}
                ),

                "llm_generation_ms":

                    round(

                        (
                            time.time()
                            -
                            llm_start
                        ) * 1000,

                        1
                    ),

                "total_ms":

                    round(

                        (
                            time.time()
                            -
                            t0
                        ) * 1000,

                        1
                    )
            }

            return {

                "answer":
                    answer,

                "entities":
                    entities,

                "graph_viz":
                    graph_viz,

                "timing":
                    timing
            }

        except Exception as e:

            logger.exception(
                "Query failed"
            )

            return {

                "answer":
                    f"Error: {e}",

                "entities":
                    [],

                "graph_viz":
                    {},

                "timing": {

                    "total_ms":

                        round(

                            (
                                time.time()
                                -
                                t0
                            ) * 1000,

                            1
                        )
                }
            }

    # ─────────────────────────────
    # CLOSE
    # ─────────────────────────────

    def close(self):

        try:

            if self.graph:

                self.graph.close()

                logger.info(
                    "Graph connection closed"
                )

        except Exception:

            logger.exception(
                "Graph close failed"
            )

# ─────────────────────────────────────────────
# THREAD-SAFE SINGLETON
# ─────────────────────────────────────────────

_pipeline: Optional[
    SafetyGraphPipeline
] = None

_lock = threading.Lock()

# ─────────────────────────────────────────────
# GET PIPELINE
# ─────────────────────────────────────────────

def get_pipeline():

    global _pipeline

    if _pipeline is not None:

        return _pipeline

    with _lock:

        if _pipeline is None:

            _pipeline = (
                SafetyGraphPipeline()
            )

    return _pipeline

# ─────────────────────────────────────────────
# MAIN QUERY API
# ─────────────────────────────────────────────

def query(user_query: str):

    return (

        get_pipeline()

        .query(user_query)
    )

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    q = """
What are the UN R14 requirements
for seat belt anchorage strength?
"""

    result = query(q)

    print("\n========================")

    print("ANSWER")

    print("========================\n")

    print(result["answer"])

    print("\n========================")

    print("TIMING")

    print("========================\n")

    print(result["timing"])