"""
Configuration management for GenCare Assistant
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4o"
    
    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "gencare-brs-index"
    
    # MongoDB
    mongodb_uri: str
    mongodb_db_name: str = "gencare_db"
    mongodb_collection_name: str = "chat_sessions"
    
    # RAG Configuration
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k_results: int = 5
    
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cache settings instance"""
    return Settings()