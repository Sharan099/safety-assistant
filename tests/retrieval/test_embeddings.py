import math

from packages.retrieval.embeddings import HashingEmbeddingProvider


def test_embed_is_deterministic() -> None:
    provider = HashingEmbeddingProvider()
    a = provider.embed("frontal collision dummy chest deflection")
    b = provider.embed("frontal collision dummy chest deflection")
    assert a == b


def test_embed_is_unit_normalized() -> None:
    provider = HashingEmbeddingProvider()
    vector = provider.embed("belt force limiter pretensioner")
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-9


def test_different_text_gives_different_vector() -> None:
    provider = HashingEmbeddingProvider()
    a = provider.embed("belt force limiter")
    b = provider.embed("airbag deployment timing")
    assert a != b


def test_empty_text_gives_zero_vector() -> None:
    provider = HashingEmbeddingProvider(dimensions=16)
    vector = provider.embed("")
    assert vector == [0.0] * 16


def test_dimensions_respected() -> None:
    provider = HashingEmbeddingProvider(dimensions=64)
    vector = provider.embed("hourglass energy")
    assert len(vector) == 64
