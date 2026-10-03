"""Settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI — only used for answer generation (iter 2); ingestion runs without it.
    openai_api_key: SecretStr | None = None

    # Embeddings — local open-source model via sentence-transformers.
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str | None = None  # None = auto (mps on Apple Silicon, else cuda/cpu)
    embedding_batch_size: int = 32
    # BGE recommends this prefix on queries (not on documents) for retrieval.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "financebench"

    # Chunking (in embedding-model tokens). BGE reads max 512 incl. [CLS]/[SEP].
    chunk_size: int = 500
    chunk_overlap: int = 64

    # Paths
    data_dir: Path = Path("data")

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "raw" / "pdfs"

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "raw" / "questions.parquet"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


@lru_cache
def get_settings() -> Settings:
    return Settings()
