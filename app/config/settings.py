"""Application configuration.

Single source of truth for all runtime settings. Values are loaded from the
``.env`` file (via ``pydantic-settings``), validated, and exposed as a typed,
immutable ``Settings`` object.

Design goals:
    * Secrets (API keys) are read from the environment only — never hardcoded.
    * Every tunable (chunk size, top-K, model IDs, paths, limits) lives here.
    * Invalid configuration fails fast at startup with a clear error.

Usage:
    >>> from app.config.settings import get_settings
    >>> settings = get_settings()
    >>> settings.llm_model
    'llama-3.3-70b-versatile'
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingBackend(str, Enum):
    """Supported embedding providers.

    ``LOCAL`` runs fully offline via sentence-transformers and needs no API key.
    ``VOYAGE`` and ``OPENAI`` are optional API backends that require a key.
    """

    LOCAL = "local"
    VOYAGE = "voyage"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Typed application settings loaded and validated from the environment.

    Attributes are grouped to mirror the ``.env.example`` template: LLM,
    embeddings, chunking, retrieval, vector store, file upload/security, and
    logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM (Groq) ----------------------------------------------------------
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
        description="Groq API key. Required for answer generation.",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="LLM_MODEL",
        description="Groq model ID used to generate answers.",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=1,
        validation_alias="LLM_MAX_TOKENS",
        description="Maximum tokens for the generated answer.",
    )

    # ---- Embeddings ----------------------------------------------------------
    embedding_backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.LOCAL,
        validation_alias="EMBEDDING_BACKEND",
        description="Which embedding provider to use: local | voyage | openai.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias="EMBEDDING_MODEL",
        description="Embedding model name for the chosen backend.",
    )
    voyage_api_key: Optional[str] = Field(
        default=None,
        validation_alias="VOYAGE_API_KEY",
        description="API key for the Voyage embedding backend (optional).",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="API key for the OpenAI embedding backend (optional).",
    )

    # ---- Chunking ------------------------------------------------------------
    chunk_size: int = Field(
        default=1000,
        ge=1,
        validation_alias="CHUNK_SIZE",
        description="Characters per chunk.",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        validation_alias="CHUNK_OVERLAP",
        description="Character overlap between consecutive chunks.",
    )

    # ---- Retrieval -----------------------------------------------------------
    top_k: int = Field(
        default=4,
        ge=1,
        validation_alias="TOP_K",
        description="Number of chunks to retrieve per question.",
    )
    use_mmr: bool = Field(
        default=True,
        validation_alias="USE_MMR",
        description=(
            "Re-rank candidates with Maximal Marginal Relevance so the final "
            "top-K spans multiple documents instead of collapsing onto the "
            "single most-similar one (improves cross-document questions)."
        ),
    )
    fetch_k: int = Field(
        default=20,
        ge=1,
        validation_alias="FETCH_K",
        description=(
            "Size of the candidate pool fetched before MMR re-ranking. Larger "
            "values give MMR more documents to diversify across. Ignored when "
            "USE_MMR is false."
        ),
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias="MMR_LAMBDA",
        description=(
            "MMR trade-off in [0, 1]: 1.0 = pure relevance (no diversity), "
            "0.0 = pure diversity. 0.5 balances both."
        ),
    )

    # ---- Vector store (ChromaDB) ---------------------------------------------
    chroma_persist_dir: Path = Field(
        default=Path("./chroma_db"),
        validation_alias="CHROMA_PERSIST_DIR",
        description="Directory where the vector DB is persisted.",
    )
    chroma_collection_name: str = Field(
        default="talk_to_your_data",
        validation_alias="CHROMA_COLLECTION_NAME",
        description="Name of the Chroma collection holding document chunks.",
    )

    # ---- File upload / security ----------------------------------------------
    documents_dir: Path = Field(
        default=Path("./documents"),
        validation_alias="DOCUMENTS_DIR",
        description="Directory where uploaded PDFs are stored.",
    )
    max_file_size_mb: int = Field(
        default=25,
        ge=1,
        validation_alias="MAX_FILE_SIZE_MB",
        description="Maximum allowed size per uploaded file, in megabytes.",
    )

    # ---- Logging -------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level: DEBUG | INFO | WARNING | ERROR.",
    )
    log_file: Path = Field(
        default=Path("./app.log"),
        validation_alias="LOG_FILE",
        description="Path to the log file.",
    )

    # ---- Validators ----------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Ensure the log level is a recognized name."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.strip().upper()
        if normalized not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(allowed)}, got '{value}'."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        """Cross-field validation that can only run once all fields are set."""
        # Overlap must be strictly smaller than chunk size or chunking loops.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )

        # If an API embedding backend is selected, its key must be present.
        if self.embedding_backend is EmbeddingBackend.VOYAGE and not self.voyage_api_key:
            raise ValueError(
                "EMBEDDING_BACKEND=voyage requires VOYAGE_API_KEY to be set."
            )
        if self.embedding_backend is EmbeddingBackend.OPENAI and not self.openai_api_key:
            raise ValueError(
                "EMBEDDING_BACKEND=openai requires OPENAI_API_KEY to be set."
            )
        return self

    # ---- Convenience helpers -------------------------------------------------
    @property
    def max_file_size_bytes(self) -> int:
        """Maximum upload size expressed in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create on-disk directories the app depends on, if they don't exist.

        Called once at startup so that the vector store, document storage, and
        log file have valid parent directories.
        """
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated ``Settings`` instance.

    The ``lru_cache`` ensures the ``.env`` file is parsed and validated only
    once per process, and that every module shares the same configuration
    object.

    Returns:
        The application settings.

    Raises:
        pydantic.ValidationError: If any setting is missing or invalid.
    """
    return Settings()
