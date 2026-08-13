import math

from packages.retrieval.embeddings import FastEmbedProvider, HashingEmbeddingProvider


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


# FastEmbedProvider — docs/ADR/0014. No @requires_db needed: these exercise
# the provider directly, not the database. Downloads the small ONNX model
# to the local fastembed cache on first run (~67 MB) if not already cached.
def test_fastembed_produces_real_semantic_vectors_of_declared_dimension() -> None:
    provider = FastEmbedProvider()
    vector = provider.embed("frontal collision occupant protection")
    assert len(vector) == provider.dimensions
    assert provider.dimensions == 384
    assert all(isinstance(v, float) for v in vector)


def test_fastembed_is_deterministic() -> None:
    provider = FastEmbedProvider()
    a = provider.embed("chest deflection increased")
    b = provider.embed("chest deflection increased")
    assert a == b


def test_fastembed_similar_sentences_are_closer_than_unrelated_ones() -> None:
    """The actual point of a *semantic* embedding, unlike the hashing
    placeholder: paraphrases should be closer in cosine distance than an
    unrelated sentence, even without exact word overlap."""
    provider = FastEmbedProvider()

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b)

    anchor = provider.embed("the vehicle's chest deflection increased between the two runs")
    paraphrase = provider.embed("chest displacement went up comparing the two simulations")
    unrelated = provider.embed("the recipe calls for two cups of flour and a pinch of salt")

    assert cosine(anchor, paraphrase) > cosine(anchor, unrelated)


def test_fastembed_model_name_and_version_are_set() -> None:
    provider = FastEmbedProvider()
    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"  # docs/ADR/0014's benchmark winner
    assert provider.model_version == "v1"
