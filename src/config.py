"""Environment-driven configuration; a .env file is optional."""
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    data_dir: Path = ROOT / "data"
    storage_dir: Path = ROOT / "storage"
    prompts_dir: Path = ROOT / "prompts"
    collection_name: str = "rag_lab1"
    embedding_model: str = "intfloat/multilingual-e5-small"
    chunk_size: int = Field(1000, ge=100)
    chunk_overlap: int = Field(150, ge=0)
    top_n: int = Field(12, ge=1)
    final_k: int = Field(5, ge=1)
    min_dense_score: float = 0.20
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    rrf_k: int = 60
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b"

    @model_validator(mode="after")
    def valid_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

