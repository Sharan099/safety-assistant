"""Generation package — grounded answer synthesis."""

__all__ = [
    "AnswerResponse",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "LLMRole",
    "RateLimitError",
    "SourceChunk",
    "answer_question",
]


def __getattr__(name: str):
    if name in {"AnswerResponse", "SourceChunk", "answer_question"}:
        from generation import answer as mod

        return getattr(mod, name)
    if name in {"LLMClient", "LLMError", "LLMResult", "LLMRole", "RateLimitError"}:
        from generation import llm_client as mod

        return getattr(mod, name)
    raise AttributeError(name)
