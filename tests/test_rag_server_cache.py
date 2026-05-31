import rag_server


def setup_function():
    rag_server._clear_rag_state()


def teardown_function():
    rag_server._clear_rag_state()


def test_pipeline_cache_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(rag_server, "RAG_PIPELINE_CACHE_SIZE", 2)

    first = object()
    second = object()
    third = object()

    rag_server._remember_pipeline("first", first)
    rag_server._remember_pipeline("second", second)
    assert list(rag_server._pipelines) == ["first", "second"]

    assert rag_server._get_cached_pipeline("first") is first
    rag_server._remember_pipeline("third", third)

    assert "second" not in rag_server._pipelines
    assert list(rag_server._pipelines) == ["first", "third"]


def test_pipeline_cache_minimum_size_is_one(monkeypatch):
    monkeypatch.setattr(rag_server, "RAG_PIPELINE_CACHE_SIZE", 1)

    rag_server._remember_pipeline("first", object())
    rag_server._remember_pipeline("second", object())

    assert list(rag_server._pipelines) == ["second"]
