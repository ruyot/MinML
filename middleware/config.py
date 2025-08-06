"""
Configuration management using Pydantic BaseSettings.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")
    development: bool = Field(default=False, env="DEVELOPMENT")
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    valid_api_keys: List[str] = Field(default=[], env="VALID_API_KEYS")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_timeout: int = Field(default=30, env="OPENAI_TIMEOUT")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    
    local_model_url: Optional[str] = Field(default=None, env="LOCAL_MODEL_URL")
    local_model_timeout: int = Field(default=60, env="LOCAL_MODEL_TIMEOUT")
    
    # Tokenizer Configuration
    tokenizer_path: str = Field(default="models/custom_tokenizer", env="TOKENIZER_PATH")
    default_tokenizer: str = Field(default="gpt2", env="DEFAULT_TOKENIZER")
    tokenizer_cache_size: int = Field(default=10, env="TOKENIZER_CACHE_SIZE")
    
    # Compression Configuration
    compression_level: int = Field(default=2, env="COMPRESSION_LEVEL", ge=1, le=3)
    max_text_length: int = Field(default=10000, env="MAX_TEXT_LENGTH")
    compression_timeout: int = Field(default=10, env="COMPRESSION_TIMEOUT")
    
    # Database Configuration
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    # Monitoring Configuration
    prometheus_enabled: bool = Field(default=False, env="PROMETHEUS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_timeout: int = Field(default=5, env="HEALTH_CHECK_TIMEOUT")
    
    # Request Timeouts
    default_request_timeout: int = Field(default=30, env="DEFAULT_REQUEST_TIMEOUT")
    compress_timeout: int = Field(default=10, env="COMPRESS_TIMEOUT")
    evaluate_timeout: int = Field(default=60, env="EVALUATE_TIMEOUT")
    train_timeout: int = Field(default=300, env="TRAIN_TIMEOUT")
    
    # Performance Settings
    max_batch_size: int = Field(default=100, env="MAX_BATCH_SIZE")
    thread_pool_size: int = Field(default=4, env="THREAD_POOL_SIZE")
    enable_gzip: bool = Field(default=True, env="ENABLE_GZIP")
    
    # Logging Configuration
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    access_log: bool = Field(default=True, env="ACCESS_LOG")
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse comma-separated allowed hosts."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v
    
    @validator("valid_api_keys", pre=True)
    def parse_api_keys(cls, v):
        """Parse comma-separated API keys."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v
    
    @validator("compression_level")
    def validate_compression_level(cls, v):
        """Validate compression level is within allowed range."""
        if v not in [1, 2, 3]:
            raise ValueError("compression_level must be 1, 2, or 3")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v.lower()
    
    @validator("log_format")
    def validate_log_format(cls, v):
        """Validate log format."""
        if v.lower() not in ["json", "text"]:
            raise ValueError("log_format must be 'json' or 'text'")
        return v.lower()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on allowed hosts."""
        origins = []
        for host in self.allowed_hosts:
            if host in ["localhost", "127.0.0.1"]:
                origins.extend([
                    f"http://{host}:{self.port}",
                    f"https://{host}:{self.port}",
                    f"http://{host}:3000",  # Common frontend port
                ])
            else:
                origins.extend([f"https://{host}", f"http://{host}"])
        return origins
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return not self.debug and not self.development
    
    def get_database_config(self) -> dict:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "echo": self.debug,
            "pool_pre_ping": True,
            "pool_recycle": 300,
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for dependency injection)."""
    return settings 