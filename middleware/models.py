"""
Pydantic models for API requests and responses.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CompressionRequest(BaseModel):
    """Request model for text compression."""
    text: str = Field(..., description="Text to compress", min_length=1, max_length=10000)
    compression_level: Optional[int] = Field(2, description="Compression level (1-3)", ge=1, le=3)
    include_stats: Optional[bool] = Field(False, description="Include compression statistics")


class CompressionResponse(BaseModel):
    """Response model for text compression."""
    original: str = Field(..., description="Original input text")
    compressed: str = Field(..., description="Compressed text")
    stats: Optional[Dict[str, Any]] = Field(None, description="Compression statistics")


class DecompressionRequest(BaseModel):
    """Request model for text decompression."""
    compressed_text: str = Field(..., description="Compressed text to decompress", min_length=1)
    compression_level: Optional[int] = Field(2, description="Original compression level", ge=1, le=3)


class DecompressionResponse(BaseModel):
    """Response model for text decompression."""
    compressed: str = Field(..., description="Original compressed text")
    decompressed: str = Field(..., description="Decompressed text approximation")


class LLMQueryRequest(BaseModel):
    """Request model for LLM query with compression."""
    prompt: str = Field(..., description="Prompt to send to LLM", min_length=1, max_length=8000)
    model: Optional[str] = Field("gpt-3.5-turbo", description="LLM model to use")
    use_compression: Optional[bool] = Field(True, description="Whether to apply compression")
    compression_level: Optional[int] = Field(2, description="Compression level", ge=1, le=3)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response", ge=1, le=4000)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    include_compression_stats: Optional[bool] = Field(False, description="Include compression statistics")


class LLMQueryResponse(BaseModel):
    """Response model for LLM query."""
    original_prompt: str = Field(..., description="Original prompt")
    compressed_prompt: Optional[str] = Field(None, description="Compressed prompt (if compression used)")
    llm_response: str = Field(..., description="Response from LLM")
    model_used: str = Field(..., description="LLM model that was used")
    compression_stats: Optional[Dict[str, Any]] = Field(None, description="Compression statistics")
    usage_stats: Optional[Dict[str, Any]] = Field(None, description="Token usage statistics")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    compression_levels_available: List[int] = Field(..., description="Available compression levels")
    tokenizer_loaded: bool = Field(..., description="Whether custom tokenizer is loaded")


class EvaluationRequest(BaseModel):
    """Request model for compression evaluation."""
    texts: List[str] = Field(..., description="List of texts to evaluate", min_items=1, max_items=100)
    compression_level: Optional[int] = Field(2, description="Compression level", ge=1, le=3)
    cost_per_token: Optional[float] = Field(0.00002, description="Cost per token for savings calculation", ge=0.0)


class EvaluationResponse(BaseModel):
    """Response model for compression evaluation."""
    total_texts: int = Field(..., description="Number of texts evaluated")
    average_compression_ratio: float = Field(..., description="Average compression ratio")
    average_token_compression_ratio: float = Field(..., description="Average token compression ratio")
    total_original_tokens: int = Field(..., description="Total original tokens")
    total_compressed_tokens: int = Field(..., description="Total compressed tokens")
    estimated_cost_savings: float = Field(..., description="Estimated cost savings")
    savings_percent: float = Field(..., description="Percentage of cost savings")
    detailed_results: List[Dict[str, Any]] = Field(..., description="Detailed results for each text")


class TokenizerTrainingRequest(BaseModel):
    """Request model for tokenizer training."""
    corpus_texts: List[str] = Field(..., description="Training corpus texts", min_items=10, max_items=10000)
    vocab_size: Optional[int] = Field(8000, description="Vocabulary size", ge=1000, le=50000)
    min_frequency: Optional[int] = Field(2, description="Minimum frequency for inclusion", ge=1)


class TokenizerTrainingResponse(BaseModel):
    """Response model for tokenizer training."""
    status: str = Field(..., description="Training status")
    vocab_size: int = Field(..., description="Final vocabulary size")
    training_time: float = Field(..., description="Training time in seconds")
    sample_tokenization: Dict[str, Any] = Field(..., description="Sample tokenization result")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details") 