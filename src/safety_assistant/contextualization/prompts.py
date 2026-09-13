"""Version-controlled prompt for document retrieval summaries.

Bump ``SUMMARY_PROMPT_VERSION`` whenever the wording changes: it is part of the summary
cache key, so every stored summary is regenerated exactly once under the new prompt.
"""

SUMMARY_PROMPT_VERSION = "2"

SUMMARY_MAX_TOKENS = 700  # headroom above the 100–250 token target; a cut-off answer is rejected, not indexed

SUMMARY_SYSTEM = """You are creating retrieval context for a document.

Read the document excerpt and produce a short factual summary of 100 to 250 tokens (at most 180
words) that helps a search system distinguish this document from other similar documents.

Include only information explicitly supported by the excerpt.

Focus on:
- what the document governs;
- its overall scope;
- major categories of requirements;
- major test/procedure areas;
- important applicability context when explicitly stated.

Do not provide legal interpretation.
Do not invent applicability.
Do not invent dates.
Do not invent vehicle categories.
Do not invent numerical thresholds.
Do not quote requirements unless needed to distinguish the document.

The summary is retrieval metadata, not regulatory evidence.

Return only the summary as plain prose: one or two paragraphs. No headings, no bullet points,
no markdown, no preamble, no explanation of what you are doing."""


def summary_user_message(document_context: str, excerpt: str) -> str:
    return (
        f"<document_context>\n{document_context}\n</document_context>\n\n"
        f"<document_excerpt>\n{excerpt}\n</document_excerpt>"
    )
