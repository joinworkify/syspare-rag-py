import pytest

from syspare_rag.config import RagConfig, load_rag_config_from_env


def test_rag_config_defaults():
    cfg = RagConfig(project_id="test-project")
    assert cfg.text_embedding_dimension == 768
    assert cfg.embedding_size == 1408
    assert cfg.enable_ocr_fallback is True


def test_load_rag_config_from_env(monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "env-project")
    monkeypatch.setenv("LOCATION", "europe-west4")
    monkeypatch.setenv("OCR_LANG", "eng")
    monkeypatch.setenv("CACHE_DIR", "./custom-cache")

    cfg = load_rag_config_from_env()
    assert cfg.project_id == "env-project"
    assert cfg.location == "europe-west4"
    assert cfg.ocr_lang == "eng"
    assert cfg.paths.cache_dir == "./custom-cache"
