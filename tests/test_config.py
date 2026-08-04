import pytest

from syspare_rag import db
from syspare_rag.config import (
    ManualConfig,
    ManualNotFoundError,
    ManualRegistry,
    RagConfig,
    load_manual_registry_from_db,
    load_rag_config_from_env,
)


def _manual(manual_id: str, organization_id: str | None = None) -> ManualConfig:
    return ManualConfig(
        manual_id=manual_id,
        display_name=manual_id,
        pdf_folder=f"manuals/{manual_id}/pdf",
        cache_dir=f"manuals/{manual_id}/cache",
        image_dir=f"manuals/{manual_id}/cache/images",
        organization_id=organization_id,
    )


@pytest.fixture
def mixed_registry() -> ManualRegistry:
    """One global manual, plus one private manual each for two different orgs."""
    return ManualRegistry(
        [
            _manual("global_manual"),
            _manual("org_a_manual", organization_id="org-a"),
            _manual("org_b_manual", organization_id="org-b"),
        ],
        default_id="global_manual",
    )


def test_global_manual_visible_to_anyone(mixed_registry):
    assert mixed_registry.resolve("global_manual", None).manual_id == "global_manual"
    assert mixed_registry.resolve("global_manual", "org-a").manual_id == "global_manual"
    assert mixed_registry.resolve("global_manual", "org-b").manual_id == "global_manual"


def test_private_manual_visible_to_its_own_org_only(mixed_registry):
    assert mixed_registry.resolve("org_a_manual", "org-a").manual_id == "org_a_manual"


def test_private_manual_rejected_for_different_org(mixed_registry):
    with pytest.raises(ManualNotFoundError):
        mixed_registry.resolve("org_a_manual", "org-b")


def test_private_manual_rejected_for_no_org(mixed_registry):
    with pytest.raises(ManualNotFoundError):
        mixed_registry.resolve("org_a_manual", None)


def test_list_for_no_org_returns_only_global(mixed_registry):
    ids = {m.manual_id for m in mixed_registry.list_for(None)}
    assert ids == {"global_manual"}


def test_list_for_org_returns_global_plus_own_private(mixed_registry):
    ids = {m.manual_id for m in mixed_registry.list_for("org-a")}
    assert ids == {"global_manual", "org_a_manual"}


def test_list_for_does_not_leak_other_orgs_private_manual(mixed_registry):
    ids = {m.manual_id for m in mixed_registry.list_for("org-a")}
    assert "org_b_manual" not in ids


def test_get_and_list_stay_unfiltered_for_internal_use(mixed_registry):
    """get()/list() are the pre-existing, unchecked methods -- internal/admin call sites
    intentionally keep using these, so they must keep seeing everything regardless of org."""
    assert mixed_registry.get("org_a_manual").manual_id == "org_a_manual"
    assert {m.manual_id for m in mixed_registry.list()} == {
        "global_manual",
        "org_a_manual",
        "org_b_manual",
    }


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


def _db_row(manual_id: str, organization_id: str | None = None, is_default: bool = False) -> dict:
    return {
        "manual_id": manual_id,
        "display_name": manual_id,
        "pdf_folder": f"manuals/{manual_id}/pdf",
        "cache_dir": f"manuals/{manual_id}/cache",
        "image_dir": f"manuals/{manual_id}/cache/images",
        "ocr_lang": "eng",
        "description": "",
        "organization_id": organization_id,
        "is_default": is_default,
    }


def test_load_manual_registry_from_db_builds_registry(monkeypatch):
    rows = [
        _db_row("global_manual", is_default=True),
        _db_row("org_a_manual", organization_id="org-a"),
    ]
    monkeypatch.setattr(db, "list_manuals", lambda: rows)
    monkeypatch.delenv("DEFAULT_MANUAL_ID", raising=False)

    registry = load_manual_registry_from_db()
    ids = {m.manual_id for m in registry.list()}
    assert ids == {"global_manual", "org_a_manual"}
    assert registry.get("org_a_manual").organization_id == "org-a"
    assert registry.get("global_manual").organization_id is None


def test_load_manual_registry_from_db_default_from_is_default_row(monkeypatch):
    rows = [
        _db_row("first_manual"),
        _db_row("second_manual", is_default=True),
    ]
    monkeypatch.setattr(db, "list_manuals", lambda: rows)
    monkeypatch.delenv("DEFAULT_MANUAL_ID", raising=False)

    registry = load_manual_registry_from_db()
    assert registry.default_id == "second_manual"


def test_load_manual_registry_from_db_default_env_overrides_is_default_row(monkeypatch):
    rows = [
        _db_row("first_manual", is_default=True),
        _db_row("second_manual"),
    ]
    monkeypatch.setattr(db, "list_manuals", lambda: rows)
    monkeypatch.setenv("DEFAULT_MANUAL_ID", "second_manual")

    registry = load_manual_registry_from_db()
    assert registry.default_id == "second_manual"


def test_load_manual_registry_from_db_empty_table_raises(monkeypatch):
    monkeypatch.setattr(db, "list_manuals", lambda: [])
    with pytest.raises(RuntimeError):
        load_manual_registry_from_db()
