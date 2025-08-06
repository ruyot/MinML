#!/usr/bin/env python3
"""
FastAPI application for semantic compression service.
"""

import os
import time
import traceback
import uuid
import hashlib
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

# Import our modules
from compression.compression_pipeline import CompressionPipeline
from compression.registry import compressor_registry, create_compression_pipeline_from_config
from tokenizer.tokenizer_utils import load_tokenizer, get_default_tokenizer

from .config import Settings, get_settings
from .auth import create_auth_components, verify_auth_and_rate_limit, RateLimiter, APIKeyAuth
from .exceptions import (
    global_exception_handler, validation_exception_handler, 
    timeout_handler, CompressionError, LLMError
)
from .metrics import metrics_collector, MetricsMiddleware, CompressionMetrics, LLMMetrics
from .models import *

# Global variables for pipeline and tokenizer
pipeline: Optional[CompressionPipeline] = None
tokenizer = None
rate_limiter: Optional[RateLimiter] = None
api_auth: Optional[APIKeyAuth] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global pipeline, tokenizer, rate_limiter, api_auth
    
    # Startup
    settings = get_settings()
    print(f"Starting compression service (env: {'production' if settings.is_production() else 'development'})...")
    
    try:
        # Load tokenizer
        if os.path.exists(settings.tokenizer_path):
            tokenizer = load_tokenizer(settings.tokenizer_path)
            print(f"Loaded custom tokenizer from {settings.tokenizer_path}")
        else:
            tokenizer = get_default_tokenizer()
            print(f"Loaded default tokenizer: {settings.default_tokenizer}")
        
        # Initialize compression pipeline
        pipeline_config = {
            "compression_level": settings.compression_level,
            "stop_word_remover": {"language": "english"},
            "keyword_extractor": {"max_keywords": 10},
            "shorthand_compressor": {"remove_spaces": True}
        }
        pipeline = create_compression_pipeline_from_config(pipeline_config, tokenizer)
        print(f"Initialized compression pipeline with level {settings.compression_level}")
        
        # Initialize auth components
        rate_limiter, api_auth, _ = create_auth_components(settings)
        print(f"Auth components initialized (rate_limit: {settings.rate_limit_enabled}, api_keys: {len(settings.valid_api_keys)} keys)")
        
        # Register any additional plugins
        try:
            compressor_registry.discover_plugins("compression.plugins")
        except Exception as e:
            print(f"No additional plugins found: {e}")
        
        print(f"Available compressors: {compressor_registry.list_compressors()}")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        tokenizer = get_default_tokenizer()
        pipeline = CompressionPipeline(tokenizer=tokenizer, compression_level=2)
        rate_limiter, api_auth, _ = create_auth_components(Settings())
        print("Fallback initialization completed")
    
    yield
    
    # Shutdown
    print("Shutting down compression service...")
    print(f"Final metrics: {metrics_collector.get_compression_stats(24)}")


def create_app() -> FastAPI:
    """Create FastAPI application with all middleware and configuration."""
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="Minimal - Semantic Compression API",
        description="High-performance semantic compression for LLM prompts and responses",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json" if not settings.is_production() else None
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        request.state.is_production = settings.is_production()
        
        # Add API key hash to state for metrics
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            request.state.api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    
    # Add metrics middleware
    app.add_middleware(
        MetricsMiddleware,
        collector=metrics_collector
    )
    
    # Add GZIP compression if enabled
    if settings.enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handlers
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Include enhanced endpoints
    from .enhanced_endpoints import router as enhanced_router
    app.include_router(enhanced_router, prefix="/api/v1", tags=["enhanced"])
    
    return app

# Create the app instance
app = create_app()


async def verify_request_auth(request: Request, settings: Settings = Depends(get_settings)):
    """Dependency to verify authentication and rate limiting."""
    api_key, rate_limit_info = await verify_auth_and_rate_limit(
        request, settings, rate_limiter, api_auth
    )
    request.state.api_key = api_key
    request.state.rate_limit_info = rate_limit_info
    return api_key


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service information."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        compression_levels_available=[1, 2, 3],
        tokenizer_loaded=tokenizer is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        compression_levels_available=[1, 2, 3],
        tokenizer_loaded=tokenizer is not None
    )


@app.post("/compress", response_model=CompressionResponse)
async def compress_text(
    request: CompressionRequest,
    auth: Optional[str] = Depends(verify_request_auth),
    settings: Settings = Depends(get_settings)
):
    """Compress text using the compression pipeline."""
    start_time = time.time()
    
    try:
        # Create pipeline with requested compression level  
        pipeline_config = {
            "compression_level": request.compression_level,
            "stop_word_remover": {"language": "english"},
            "keyword_extractor": {"max_keywords": 10},
            "shorthand_compressor": {"remove_spaces": True}
        }
        
        temp_pipeline = await timeout_handler(
            create_compression_pipeline_from_config(pipeline_config, tokenizer),
            settings.compress_timeout,
            "compression pipeline creation"
        )
        
        if request.include_stats:
            result = await timeout_handler(
                temp_pipeline.compress_with_stats(request.text),
                settings.compress_timeout,
                "compression with stats"
            )
            
            # Record metrics
            compression_time = (time.time() - start_time) * 1000
            compression_metrics = CompressionMetrics(
                timestamp=datetime.utcnow(),
                original_tokens=result['stats']['original_tokens'],
                compressed_tokens=result['stats']['compressed_tokens'],
                original_length=len(result['original']),
                compressed_length=len(result['compressed']),
                compression_level=request.compression_level,
                compression_time_ms=compression_time,
                compression_ratio=result['stats']['compression_ratio'],
                token_compression_ratio=result['stats']['token_compression_ratio'],
                tokens_saved=result['stats']['tokens_saved'],
                estimated_cost_savings=temp_pipeline.estimate_cost_savings(request.text)['cost_savings'],
                endpoint="/compress",
                api_key_hash=getattr(request.state, 'api_key_hash', None) if hasattr(request, 'state') else None
            )
            await metrics_collector.record_compression(compression_metrics)
            
            return CompressionResponse(
                original=result['original'],
                compressed=result['compressed'],
                stats=result['stats']
            )
        else:
            compressed = await timeout_handler(
                temp_pipeline.compress(request.text),
                settings.compress_timeout,
                "compression"
            )
            return CompressionResponse(
                original=request.text,
                compressed=compressed
            )
    
    except Exception as e:
        if "timeout" in str(e).lower():
            raise CompressionError(f"Compression timed out: {str(e)}", "COMPRESSION_TIMEOUT")
        raise CompressionError(f"Compression failed: {str(e)}", "COMPRESSION_ERROR")


@app.post("/decompress", response_model=DecompressionResponse)
async def decompress_text(request: DecompressionRequest):
    """Decompress compressed text (best effort approximation)."""
    try:
        # Create pipeline with same compression level used for compression
        temp_pipeline = CompressionPipeline(
            tokenizer=tokenizer,
            compression_level=request.compression_level
        )
        
        decompressed = temp_pipeline.decompress(request.compressed_text)
        
        return DecompressionResponse(
            compressed=request.compressed_text,
            decompressed=decompressed
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decompression failed: {str(e)}")


@app.post("/compress-and-query", response_model=LLMQueryResponse)
async def compress_and_query_llm(request: LLMQueryRequest):
    """Compress prompt and query LLM."""
    try:
        # Prepare the prompt
        original_prompt = request.prompt
        
        if request.use_compression:
            # Create pipeline with requested compression level
            temp_pipeline = CompressionPipeline(
                tokenizer=tokenizer,
                compression_level=request.compression_level
            )
            
            if request.include_compression_stats:
                compression_result = temp_pipeline.compress_with_stats(original_prompt)
                compressed_prompt = compression_result['compressed']
                compression_stats = compression_result['stats']
            else:
                compressed_prompt = temp_pipeline.compress(original_prompt)
                compression_stats = None
            
            prompt_to_send = compressed_prompt
        else:
            compressed_prompt = None
            compression_stats = None
            prompt_to_send = original_prompt
        
        # Query LLM
        llm_response, model_used, usage_stats = await query_llm(
            prompt=prompt_to_send,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return LLMQueryResponse(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            llm_response=llm_response,
            model_used=model_used,
            compression_stats=compression_stats,
            usage_stats=usage_stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_compression(request: EvaluationRequest):
    """Evaluate compression performance on multiple texts."""
    try:
        # Create pipeline with requested compression level
        temp_pipeline = CompressionPipeline(
            tokenizer=tokenizer,
            compression_level=request.compression_level
        )
        
        detailed_results = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        total_compression_ratios = []
        total_token_compression_ratios = []
        
        for text in request.texts:
            result = temp_pipeline.compress_with_stats(text)
            stats = result['stats']
            
            # Cost savings calculation
            cost_savings = temp_pipeline.estimate_cost_savings(text, request.cost_per_token)
            
            detailed_result = {
                'original': text,
                'compressed': result['compressed'],
                'compression_ratio': stats['compression_ratio'],
                'token_compression_ratio': stats['token_compression_ratio'],
                'original_tokens': stats['original_tokens'],
                'compressed_tokens': stats['compressed_tokens'],
                'cost_savings': cost_savings['cost_savings'],
                'savings_percent': cost_savings['savings_percent']
            }
            detailed_results.append(detailed_result)
            
            total_original_tokens += stats['original_tokens']
            total_compressed_tokens += stats['compressed_tokens']
            total_compression_ratios.append(stats['compression_ratio'])
            total_token_compression_ratios.append(stats['token_compression_ratio'])
        
        # Calculate averages
        avg_compression_ratio = sum(total_compression_ratios) / len(total_compression_ratios)
        avg_token_compression_ratio = sum(total_token_compression_ratios) / len(total_token_compression_ratios)
        
        # Calculate total cost savings
        total_original_cost = total_original_tokens * request.cost_per_token
        total_compressed_cost = total_compressed_tokens * request.cost_per_token
        total_cost_savings = total_original_cost - total_compressed_cost
        savings_percent = (total_cost_savings / total_original_cost * 100) if total_original_cost > 0 else 0
        
        return EvaluationResponse(
            total_texts=len(request.texts),
            average_compression_ratio=avg_compression_ratio,
            average_token_compression_ratio=avg_token_compression_ratio,
            total_original_tokens=total_original_tokens,
            total_compressed_tokens=total_compressed_tokens,
            estimated_cost_savings=total_cost_savings,
            savings_percent=savings_percent,
            detailed_results=detailed_results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/train-tokenizer", response_model=TokenizerTrainingResponse)
async def train_custom_tokenizer(request: TokenizerTrainingRequest, background_tasks: BackgroundTasks):
    """Train a custom tokenizer on provided corpus."""
    try:
        from tokenizer.trainer import train_tokenizer
        
        start_time = time.time()
        
        # Save corpus to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in request.corpus_texts:
                f.write(text + '\n')
            corpus_path = f.name
        
        try:
            # Train tokenizer
            trained_tokenizer = train_tokenizer(
                corpus_path=corpus_path,
                output_dir="models/custom_tokenizer",
                vocab_size=request.vocab_size,
                min_frequency=request.min_frequency
            )
            
            training_time = time.time() - start_time
            
            # Get sample tokenization
            sample_text = request.corpus_texts[0][:100]  # First 100 chars
            tokens = trained_tokenizer.encode(sample_text)
            
            # Reload the pipeline with new tokenizer (in background)
            background_tasks.add_task(reload_pipeline)
            
            return TokenizerTrainingResponse(
                status="completed",
                vocab_size=len(trained_tokenizer.get_vocab()),
                training_time=training_time,
                sample_tokenization={
                    "text": sample_text,
                    "tokens": tokens.tokens,
                    "token_count": len(tokens.ids)
                }
            )
        
        finally:
            # Clean up temporary file
            os.unlink(corpus_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenizer training failed: {str(e)}")


@app.get("/stats")
async def get_pipeline_stats():
    """Get compression pipeline statistics."""
    try:
        if pipeline:
            return pipeline.get_pipeline_stats()
        else:
            return {"error": "Pipeline not initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/reset-stats")
async def reset_pipeline_stats():
    """Reset compression pipeline statistics."""
    try:
        if pipeline:
            pipeline.reset_stats()
            return {"message": "Statistics reset successfully"}
        else:
            return {"error": "Pipeline not initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset stats: {str(e)}")


async def query_llm(prompt: str, model: str, max_tokens: Optional[int] = None, 
                   temperature: float = 0.7) -> tuple[str, str, Dict[str, Any]]:
    """Query LLM with the given prompt."""
    
    # Try OpenAI API first
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and model.startswith(("gpt-", "text-")):
        try:
            import openai
            
            client = openai.OpenAI(api_key=openai_api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return (
                response.choices[0].message.content,
                model,
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
        
        except Exception as e:
            print(f"OpenAI API error: {e}")
    
    # Fallback to local model or mock response
    local_model_url = os.getenv("LOCAL_MODEL_URL")
    if local_model_url:
        try:
            import requests
            
            response = requests.post(
                f"{local_model_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens or 512,
                    "temperature": temperature
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return (
                    result.get("generated_text", ""),
                    result.get("model", "local"),
                    result.get("usage", {})
                )
        
        except Exception as e:
            print(f"Local model error: {e}")
    
    # Mock response for demonstration
    return (
        f"Mock response to: {prompt[:100]}...",
        "mock-model",
        {"prompt_tokens": len(prompt.split()), "completion_tokens": 20, "total_tokens": len(prompt.split()) + 20}
    )


async def reload_pipeline():
    """Reload the compression pipeline with new tokenizer."""
    global pipeline, tokenizer
    
    try:
        tokenizer_path = os.getenv("TOKENIZER_PATH", "models/custom_tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = load_tokenizer(tokenizer_path)
            compression_level = int(os.getenv("COMPRESSION_LEVEL", "2"))
            pipeline = CompressionPipeline(tokenizer=tokenizer, compression_level=compression_level)
            print("Pipeline reloaded with new tokenizer")
    except Exception as e:
        print(f"Error reloading pipeline: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 