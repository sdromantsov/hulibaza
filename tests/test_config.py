"""Unit tests for config loading, resolution & the validity gate."""

import pytest
import yaml
from pydantic import ValidationError

from hulibaza.config import (
    Defaults,
    GlobalConfig,
    ModelSpec,
    ResolvedSectionConfig,
    SectionConfig,
    discover_sections,
    load_global_config,
    load_section_config,
    resolve_section_config,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def tokenizer_file(tmp_path):
    """A stand-in tokenizer.json (the validator only checks existence)."""
    f = tmp_path / "tok.json"
    f.write_text("{}")
    return str(f)


# ── ModelSpec ──


def test_modelspec_requires_existing_tokenizer(tmp_path):
    with pytest.raises(ValidationError):
        ModelSpec(max_context=2048, tokenizer_path=str(tmp_path / "missing.json"))


def test_modelspec_defaults_batch_16(tokenizer_file):
    spec = ModelSpec(max_context=2048, tokenizer_path=tokenizer_file)
    assert spec.embed_batch_size == 16


def test_modelspec_max_context_bounds(tokenizer_file):
    with pytest.raises(ValidationError):
        ModelSpec(max_context=8, tokenizer_path=tokenizer_file)  # < 32


# ── GlobalConfig ──


def test_global_defaults_and_new_knobs():
    cfg = GlobalConfig()
    assert cfg.size_cap_text_bytes == 10 * 1024 * 1024
    assert cfg.size_cap_other_bytes == 1 * 1024 * 1024
    assert cfg.deletion_grace_days == 7
    assert cfg.daemon_enabled is True
    assert cfg.daemon_poll_seconds == 60


def test_global_wiki_dir_resolved_absolute():
    cfg = GlobalConfig(wiki_dir="./docs")
    assert cfg.wiki_dir.startswith("/")


def test_global_empty_registry_skips_validation():
    # No models declared → no validation error even with an odd default model.
    cfg = GlobalConfig(defaults=Defaults(embed_model="whatever"))
    assert cfg.models == {}


def test_global_default_model_must_be_in_registry(tokenizer_file):
    with pytest.raises(ValidationError):
        GlobalConfig(
            models={"m": ModelSpec(max_context=2048, tokenizer_path=tokenizer_file)},
            defaults=Defaults(embed_model="not-m"),
        )


def test_global_default_chunk_size_over_capacity(tokenizer_file):
    with pytest.raises(ValidationError):
        GlobalConfig(
            models={"m": ModelSpec(max_context=100, tokenizer_path=tokenizer_file)},
            defaults=Defaults(embed_model="m", chunk_size=200, headroom_ratio=0.02),
        )


def test_global_valid_with_registry(tokenizer_file):
    cfg = GlobalConfig(
        models={"m": ModelSpec(max_context=2048, tokenizer_path=tokenizer_file)},
        defaults=Defaults(embed_model="m", chunk_size=512),
    )
    assert cfg.defaults.embed_model == "m"


# ── resolve_section_config ──


def _models(tokenizer_file, **over):
    base = dict(max_context=2048, tokenizer_path=tokenizer_file, embed_batch_size=8)
    base.update(over)
    return {"m": ModelSpec(**base)}


def test_resolve_inherits_defaults(tmp_path, tokenizer_file):
    defaults = Defaults(embed_model="m", chunk_size=512, chunk_overlap_ratio=0.1)
    r = resolve_section_config("s", tmp_path, SectionConfig(), defaults, _models(tokenizer_file))
    assert r.embed_model == "m"
    assert r.chunk_size == 512
    assert r.chunk_overlap == int(512 * 0.1)  # 51
    assert r.embed_batch_size == 8  # from the model spec
    assert r.enabled


def test_resolve_section_overrides(tmp_path, tokenizer_file):
    defaults = Defaults(embed_model="m", chunk_size=512, chunk_overlap_ratio=0.1)
    sec = SectionConfig(chunk_size=1000, chunk_overlap_ratio=0.2)
    r = resolve_section_config("s", tmp_path, sec, defaults, _models(tokenizer_file))
    assert r.chunk_size == 1000
    assert r.chunk_overlap == 200


def test_resolve_unknown_model_disables(tmp_path, tokenizer_file):
    defaults = Defaults(embed_model="m")
    sec = SectionConfig(embed_model="ghost")
    r = resolve_section_config("s", tmp_path, sec, defaults, _models(tokenizer_file))
    assert not r.enabled
    assert "ghost" in r.disabled_reason


def test_resolve_oversized_chunk_disables(tmp_path, tokenizer_file):
    defaults = Defaults(embed_model="m")
    sec = SectionConfig(chunk_size=30000, headroom_ratio=0.02)
    r = resolve_section_config("s", tmp_path, sec, defaults, _models(tokenizer_file, max_context=2048))
    assert not r.enabled
    assert "exceeds effective max" in r.disabled_reason


def test_resolve_no_registry_is_enabled(tmp_path):
    # Empty registry → validity gate not applied → section enabled.
    r = resolve_section_config("s", tmp_path, SectionConfig(), Defaults(), models=None)
    assert r.enabled
    assert r.embed_batch_size == 16  # falls back to default


# ── load / discover ──


def test_load_global_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_global_config(str(tmp_path / "nope.yaml"))


def test_load_global_config_from_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "wiki_dir": str(tmp_path),
        "deletion_grace_days": 3,
        "daemon_poll_seconds": 120,
    }))
    cfg = load_global_config(str(cfg_path))
    assert cfg.deletion_grace_days == 3
    assert cfg.daemon_poll_seconds == 120


def test_load_section_config_absent(tmp_path):
    assert load_section_config(tmp_path) is None


def test_load_section_config_present(tmp_path):
    (tmp_path / "section.yaml").write_text(yaml.safe_dump({
        "description": "docs",
        "chunk_size": 256,
    }))
    sec = load_section_config(tmp_path)
    assert sec.description == "docs"
    assert sec.chunk_size == 256


def test_discover_sections(tmp_path, tokenizer_file):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "section.yaml").write_text("description: A\n")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "section.yaml").write_text("embed_model: ghost\n")
    (tmp_path / "no_section").mkdir()  # ignored (no section.yaml)
    (tmp_path / "loose.txt").write_text("x")  # not a dir

    cfg = GlobalConfig(
        wiki_dir=str(tmp_path),
        models={"m": ModelSpec(max_context=2048, tokenizer_path=tokenizer_file)},
        defaults=Defaults(embed_model="m"),
    )
    sections = discover_sections(cfg)
    names = [s.name for s in sections]
    assert names == ["alpha", "beta"]  # sorted, no_section excluded
    by_name = {s.name: s for s in sections}
    assert by_name["alpha"].enabled
    assert not by_name["beta"].enabled  # ghost model → disabled, still listed


def test_discover_sections_missing_wiki_dir(tmp_path):
    cfg = GlobalConfig(wiki_dir=str(tmp_path / "gone"))
    assert discover_sections(cfg) == []
