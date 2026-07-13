"""Configuration loading & validation via Pydantic v2.

One global `config.yaml` (path via CONFIG_PATH) declares infra URLs, the model
registry, defaults, and the global knobs (size caps, deletion grace, daemon).
Each section overrides `defaults` through its own `section.yaml`. Resolution
produces a frozen ResolvedSectionConfig per section.

Validity gate: a section whose resolved config is invalid — unknown
embed_model, or chunk_size beyond the model's effective capacity — is NOT
dropped; it carries a `disabled_reason` and is listed but refused for
ingest/search. Bad GLOBAL defaults, by contrast, are a hard startup failure
.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


class ModelSpec(BaseModel):
    """One declared embedding model: capacity, local tokenizer, batch size."""

    max_context: int = Field(..., ge=32, le=1_000_000)
    tokenizer_path: str = Field(..., description="Absolute path to tokenizer.json on disk.")
    embed_batch_size: int = Field(default=16, ge=1, le=512)

    @field_validator("tokenizer_path")
    @classmethod
    def ensure_path_exists(cls, v: str) -> str:
        # Token counts are computed locally; a missing tokenizer is a hard
        # startup failure.
        if not Path(v).exists():
            raise ValueError(f"tokenizer file not found: {v}")
        return v


class Defaults(BaseModel):
    """Global defaults a section may override via section.yaml."""

    embed_model: str = "nomic-embed-text"
    chunk_size: int = Field(default=512, ge=32, le=32768)
    # Overlap as a fraction of chunk_size (0.1 = 10%).
    chunk_overlap_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    # Fraction of max_context reserved for special tokens (BOS/EOS/...) the
    # tokenizer adds; applied when validating chunk_size against a model.
    headroom_ratio: float = Field(default=0.02, ge=0.0, le=0.5)
    # Background embedder/model health recheck cadences.
    embedder_retry_interval_seconds: int = Field(default=30, ge=5, le=3600)
    model_retry_interval_seconds: int = Field(default=300, ge=10, le=86400)


class GlobalConfig(BaseModel):
    embedding_url: str = "http://localhost:11434"
    embedding_timeout: int = Field(default=600, ge=10, le=3600)
    qdrant_url: str = "http://localhost:6333"
    postgres_url: str = "postgresql://wiki:wiki@localhost:5432/wiki"
    wiki_dir: str = "./docs"

    # File-gating & lifecycle knobs. Byte caps — the whitelist (PDF /
    # known-text) is exempt from the small "other" cap.
    size_cap_text_bytes: int = Field(default=10 * _MB, ge=1)
    size_cap_other_bytes: int = Field(default=1 * _MB, ge=1)
    deletion_grace_days: int = Field(default=7, ge=0, le=365)
    daemon_enabled: bool = True
    daemon_poll_seconds: int = Field(default=60, ge=5, le=86400)

    models: dict[str, ModelSpec] = Field(default_factory=dict)
    defaults: Defaults = Field(default_factory=Defaults)

    @field_validator("wiki_dir")
    @classmethod
    def resolve_wiki_dir(cls, v: str) -> str:
        return str(Path(v).resolve())

    @model_validator(mode="after")
    def validate_defaults_against_models(self) -> "GlobalConfig":
        # An empty registry skips validation (bootstrap / tests). With a
        # registry present, the global defaults must themselves be valid.
        if not self.models:
            return self
        if self.defaults.embed_model not in self.models:
            raise ValueError(
                f"defaults.embed_model '{self.defaults.embed_model}' not declared in "
                f"models registry. Available: {sorted(self.models.keys())}"
            )
        spec = self.models[self.defaults.embed_model]
        effective_max = int(spec.max_context * (1 - self.defaults.headroom_ratio))
        if self.defaults.chunk_size > effective_max:
            raise ValueError(
                f"defaults.chunk_size ({self.defaults.chunk_size}) exceeds effective max "
                f"({effective_max} = max_context {spec.max_context} × "
                f"(1 − headroom_ratio {self.defaults.headroom_ratio}))"
            )
        return self


class SectionConfig(BaseModel):
    """Raw section.yaml: every override is optional (None ⇒ inherit default)."""

    description: str = ""
    embed_model: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=32, le=32768)
    chunk_overlap_ratio: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    headroom_ratio: Optional[float] = Field(default=None, ge=0.0, le=0.5)


class ResolvedSectionConfig(BaseModel):
    name: str
    path: Path
    description: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int  # absolute, computed = int(chunk_size × ratio)
    chunk_overlap_ratio: float  # kept for fingerprint + status reporting
    headroom_ratio: float
    embed_batch_size: int
    disabled_reason: Optional[str] = None

    model_config = {"frozen": True}

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None


def resolve_section_config(
    name: str,
    path: Path,
    section: SectionConfig,
    defaults: Defaults,
    models: dict[str, ModelSpec] | None = None,
) -> ResolvedSectionConfig:
    """Merge a section over the global defaults and apply the validity gate."""
    embed_model = section.embed_model or defaults.embed_model
    chunk_size = section.chunk_size if section.chunk_size is not None else defaults.chunk_size
    chunk_overlap_ratio = (
        section.chunk_overlap_ratio
        if section.chunk_overlap_ratio is not None
        else defaults.chunk_overlap_ratio
    )
    headroom_ratio = (
        section.headroom_ratio
        if section.headroom_ratio is not None
        else defaults.headroom_ratio
    )
    chunk_overlap = int(chunk_size * chunk_overlap_ratio)

    disabled_reason: Optional[str] = None
    embed_batch_size = 16  # ModelSpec default; overridden below when the model is known
    if models:
        if embed_model not in models:
            disabled_reason = (
                f"embed_model '{embed_model}' not declared in global models registry. "
                f"Available: {sorted(models.keys())}"
            )
        else:
            spec = models[embed_model]
            embed_batch_size = spec.embed_batch_size
            effective_max = int(spec.max_context * (1 - headroom_ratio))
            if chunk_size > effective_max:
                disabled_reason = (
                    f"chunk_size ({chunk_size}) exceeds effective max "
                    f"({effective_max} = max_context {spec.max_context} × "
                    f"(1 − headroom_ratio {headroom_ratio})). "
                    f"Reduce chunk_size or headroom_ratio in section.yaml."
                )

    return ResolvedSectionConfig(
        name=name,
        path=path,
        description=section.description,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_overlap_ratio=chunk_overlap_ratio,
        headroom_ratio=headroom_ratio,
        embed_batch_size=embed_batch_size,
        disabled_reason=disabled_reason,
    )


def load_global_config(config_path: str | None = None) -> GlobalConfig:
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = GlobalConfig(**(raw or {}))
    logger.info("Loaded global config from %s", path)
    return config


def load_section_config(section_dir: Path) -> SectionConfig | None:
    yaml_path = section_dir / "section.yaml"
    if not yaml_path.exists():
        return None

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    return SectionConfig(**(raw or {}))


def discover_sections(global_config: GlobalConfig) -> list[ResolvedSectionConfig]:
    """A section is a wiki_dir subdir containing section.yaml.

    Rescanned on every call so new sections appear without a restart. Disabled
    sections are still returned (listed with a reason); the caller decides.
    """
    wiki_dir = Path(global_config.wiki_dir)
    if not wiki_dir.exists():
        logger.warning("wiki_dir does not exist: %s", wiki_dir)
        return []

    sections: list[ResolvedSectionConfig] = []
    for entry in sorted(wiki_dir.iterdir()):
        if not entry.is_dir():
            continue
        section_cfg = load_section_config(entry)
        if section_cfg is None:
            continue
        resolved = resolve_section_config(
            name=entry.name,
            path=entry.resolve(),
            section=section_cfg,
            defaults=global_config.defaults,
            models=global_config.models,
        )
        if resolved.disabled_reason:
            logger.warning("Section '%s' disabled: %s", resolved.name, resolved.disabled_reason)
        else:
            logger.info("Discovered section: %s", resolved.name)
        sections.append(resolved)

    return sections
