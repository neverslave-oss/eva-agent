"""Tests for embedding_client in kernel-evolving (native + http backends)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unittest.mock import patch, MagicMock


def test_cosine_similarity_identical():
    from core.memory.embedding_client import cosine_similarity
    v = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    from core.memory.embedding_client import cosine_similarity
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-6


def test_embed_client_caches():
    from core.memory.embedding_client import EmbeddingClient
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
    with patch("core.memory.embedding_client.SentenceTransformer", return_value=mock_model):
        client = EmbeddingClient(backend="native", model_path="/fake/path")
        v1 = client.get("hello")
        v2 = client.get("hello")
        assert v1 == v2
        # encode() only called once — second call uses cache
        assert mock_model.encode.call_count == 1


def test_native_backend_calls_sentence_transformers():
    """backend=native uses SentenceTransformer, not urllib."""
    from core.memory.embedding_client import EmbeddingClient
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = [np.array([0.5, 0.5])]
    with patch("core.memory.embedding_client.SentenceTransformer", return_value=mock_model) as mock_st:
        client = EmbeddingClient(backend="native", model_path="/mocked/path")
        vec = client.get("test input")
        mock_st.assert_called_once_with("/mocked/path")
        mock_model.encode.assert_called_once()
        assert vec is not None
        assert len(vec) == 2


def test_native_backend_fallback_on_model_load_error():
    """If SentenceTransformer raises on instantiation, get() returns None gracefully."""
    from core.memory.embedding_client import EmbeddingClient
    import core.memory.embedding_client as ec
    # SentenceTransformer is importable but the constructor raises
    mock_st_cls = MagicMock(side_effect=RuntimeError("model load failed"))
    orig = ec.SentenceTransformer
    ec.SentenceTransformer = mock_st_cls
    try:
        client = EmbeddingClient(backend="native", model_path="/mocked/path")
        result = client.get("hello")
        assert result is None
    finally:
        ec.SentenceTransformer = orig


def test_http_backend_posts_to_url():
    """backend=http posts to embedding_url via urllib, not SentenceTransformer."""
    import json
    from core.memory.embedding_client import EmbeddingClient
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"data": [[0.1, 0.2]]}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("core.memory.embedding_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
        with patch("core.memory.embedding_client.SentenceTransformer") as mock_st:
            client = EmbeddingClient(backend="http", embedding_url="http://localhost:8770/embeddings")
            vec = client.get("hello")
            mock_st.assert_not_called()
            mock_open.assert_called_once()
            assert vec == [0.1, 0.2]


def test_embed_client_fallback_on_error():
    """Any embed error returns None gracefully (both backends)."""
    from core.memory.embedding_client import EmbeddingClient
    import core.memory.embedding_client as ec
    mock_st_cls = MagicMock(side_effect=RuntimeError("gpu oom"))
    orig = ec.SentenceTransformer
    ec.SentenceTransformer = mock_st_cls
    try:
        client = EmbeddingClient(backend="native", model_path="/fake")
        result = client.get("hello")
        assert result is None
    finally:
        ec.SentenceTransformer = orig


def test_score_match_semantic_fallback():
    """_score_match_semantic returns 0.0 when embedding is unavailable."""
    from core.evolution.evolver import _score_match_semantic
    from core.memory.embedding_client import EmbeddingClient
    with patch.object(EmbeddingClient, "similarity", return_value=None):
        skill = {"name": "test-skill", "description": "does stuff", "commands": ["/test"]}
        client = EmbeddingClient(backend="native", model_path="/fake")
        score = _score_match_semantic(skill, "find a skill to do stuff", client)
        assert score == 0.0
