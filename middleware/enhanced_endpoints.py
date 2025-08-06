"""
Enhanced API endpoints for advanced functionality.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from .config import Settings, get_settings
from .auth import verify_request_auth
from .metrics import metrics_collector
from .models import *
from compression.registry import compressor_registry

# Create router for enhanced endpoints
router = APIRouter()


@router.get("/compressors", response_model=Dict[str, Any])
async def list_compressors(
    auth: Optional[str] = Depends(verify_request_auth),
    settings: Settings = Depends(get_settings)
):
    """List all available compression techniques."""
    compressors = {}
    
    for name in compressor_registry.list_compressors():
        info = compressor_registry.get_compressor_info(name)
        compressors[name] = {
            "name": info["name"],
            "class_name": info["class"],
            "description": info["docstring"][:200] + "..." if len(info["docstring"]) > 200 else info["docstring"]
        }
    
    return {
        "total_compressors": len(compressors),
        "compressors": compressors,
        "builtin_compressors": ["stop_word_remover", "keyword_extractor", "shorthand_compressor"],
        "custom_compressors": [name for name in compressors.keys() 
                               if name not in ["stop_word_remover", "keyword_extractor", "shorthand_compressor"]]
    }


@router.get("/compressors/{compressor_name}", response_model=Dict[str, Any])
async def get_compressor_info(
    compressor_name: str,
    auth: Optional[str] = Depends(verify_request_auth),
    settings: Settings = Depends(get_settings)
):
    """Get detailed information about a specific compressor."""
    try:
        info = compressor_registry.get_compressor_info(compressor_name)
        compressor_class = compressor_registry.get_compressor(compressor_name)
        
        # Get configuration options from __init__ signature
        import inspect
        init_signature = inspect.signature(compressor_class.__init__)
        config_params = []
        
        for param_name, param in init_signature.parameters.items():
            if param_name not in ['self', 'config']:
                config_params.append({
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None
                })
        
        return {
            **info,
            "configuration_parameters": config_params,
            "supports_decompression": hasattr(compressor_class, 'decompress'),
            "example_usage": {
                "config": {"example_param": "example_value"},
                "usage": f"compressor_registry.create_compressor('{compressor_name}', config)"
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/metrics/detailed", response_model=Dict[str, Any])
async def get_detailed_metrics(
    hours: int = Query(24, description="Number of hours to include in metrics", ge=1, le=168),
    auth: Optional[str] = Depends(verify_request_auth),
    settings: Settings = Depends(get_settings)
):
    """Get detailed metrics including trends and breakdowns."""
    return {
        "compression_stats": metrics_collector.get_compression_stats(hours),
        "performance_stats": metrics_collector.get_performance_stats(),
        "llm_stats": metrics_collector.get_llm_stats(hours),
        "hourly_trends": metrics_collector.get_hourly_trends(hours),
        "summary": {
            "period_hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "metrics_available": {
                "compression_metrics": len(metrics_collector.compression_metrics),
                "request_metrics": len(metrics_collector.request_metrics),
                "llm_metrics": len(metrics_collector.llm_metrics)
            }
        }
    }


@router.get("/metrics/export")
async def export_metrics(
    format: str = Query("json", description="Export format (json)"),
    hours: int = Query(24, description="Number of hours to include", ge=1, le=168),
    auth: Optional[str] = Depends(verify_request_auth),
    settings: Settings = Depends(get_settings)
):
    """Export metrics in various formats."""
    try:
        exported_data = metrics_collector.export_metrics(format)
        
        if format.lower() == "json":
            return JSONResponse(
                content=exported_data,
                headers={
                    "Content-Disposition": f"attachment; filename=metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        else:
            return PlainTextResponse(
                content=exported_data,
                headers={
                    "Content-Disposition": f"attachment; filename=metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}"
                }
            )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compression/custom", response_model=CompressionResponse)
async def compress_with_custom_pipeline(
    request: Dict[str, Any],
    auth: Optional[str] = Depends(verify_request_auth),
    settings: Settings = Depends(get_settings)
):
    """Compress text using a custom compression pipeline configuration."""
    try:
        text = request.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        pipeline_config = request.get("pipeline_config", {})
        if not pipeline_config:
            raise HTTPException(status_code=400, detail="pipeline_config field is required")
        
        # Validate compressors exist
        if "compressors" in pipeline_config:
            for comp_config in pipeline_config["compressors"]:
                comp_name = comp_config if isinstance(comp_config, str) else comp_config.get("name")
                if comp_name not in compressor_registry.list_compressors():
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unknown compressor: {comp_name}. Available: {compressor_registry.list_compressors()}"
                    )
        
        # Create and use custom pipeline
        from compression.registry import create_compression_pipeline_from_config
        from tokenizer.tokenizer_utils import get_default_tokenizer
        
        tokenizer = get_default_tokenizer()  # Or load from settings
        pipeline = create_compression_pipeline_from_config(pipeline_config, tokenizer)
        
        include_stats = request.get("include_stats", False)
        
        if include_stats:
            result = pipeline.compress_with_stats(text)
            return CompressionResponse(
                original=result['original'],
                compressed=result['compressed'],
                stats=result['stats']
            )
        else:
            compressed = pipeline.compress(text)
            return CompressionResponse(
                original=text,
                compressed=compressed
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom compression failed: {str(e)}")


@router.get("/info", response_model=Dict[str, Any])
async def get_service_info(
    settings: Settings = Depends(get_settings)
):
    """Get comprehensive service information."""
    return {
        "service": {
            "name": "Minimal Compression API",
            "version": "1.0.0",
            "environment": "production" if settings.is_production() else "development",
            "started_at": datetime.utcnow().isoformat(),  # This would be stored in app state
        },
        "configuration": {
            "compression_levels": [1, 2, 3],
            "default_compression_level": settings.compression_level,
            "max_text_length": settings.max_text_length,
            "compression_timeout": settings.compress_timeout,
            "rate_limiting_enabled": settings.rate_limit_enabled,
            "authentication_enabled": bool(settings.valid_api_keys)
        },
        "capabilities": {
            "available_compressors": compressor_registry.list_compressors(),
            "supports_custom_pipelines": True,
            "supports_metrics": True,
            "supports_decompression": True,
            "supports_llm_integration": bool(settings.openai_api_key or settings.local_model_url)
        },
        "limits": {
            "max_text_length": settings.max_text_length,
            "max_batch_size": settings.max_batch_size,
            "rate_limit_requests": settings.rate_limit_requests if settings.rate_limit_enabled else None,
            "rate_limit_window": settings.rate_limit_window if settings.rate_limit_enabled else None
        }
    }


@router.get("/docs/api-examples")
async def get_api_examples():
    """Get interactive API examples and documentation."""
    return {
        "examples": {
            "basic_compression": {
                "description": "Basic text compression",
                "endpoint": "POST /compress",
                "request": {
                    "text": "Please explain machine learning concepts in detail",
                    "compression_level": 2,
                    "include_stats": True
                },
                "curl_example": """curl -X POST "http://localhost:8000/compress" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Please explain machine learning concepts", "compression_level": 2}'"""
            },
            "custom_pipeline": {
                "description": "Custom compression pipeline",
                "endpoint": "POST /compression/custom",
                "request": {
                    "text": "Your text here",
                    "pipeline_config": {
                        "compressors": [
                            {"name": "stop_word_remover", "language": "english"},
                            {"name": "keyword_extractor", "max_keywords": 5}
                        ]
                    },
                    "include_stats": True
                }
            },
            "llm_query": {
                "description": "Compress and query LLM",
                "endpoint": "POST /compress-and-query",
                "request": {
                    "prompt": "Explain quantum computing",
                    "model": "gpt-3.5-turbo",
                    "use_compression": True,
                    "compression_level": 2
                }
            },
            "metrics": {
                "description": "Get compression metrics",
                "endpoint": "GET /metrics/detailed?hours=24",
                "curl_example": """curl "http://localhost:8000/metrics/detailed?hours=24" """
            }
        },
        "authentication": {
            "api_key": {
                "header": "X-API-Key",
                "example": "curl -H 'X-API-Key: your-key-here' ..."
            }
        },
        "response_formats": {
            "standard_response": {
                "original": "Original text",
                "compressed": "Compressed text",
                "stats": "Optional statistics object"
            },
            "error_response": {
                "error": {
                    "code": "ERROR_CODE",
                    "message": "Human readable message",
                    "details": "Additional error details"
                }
            }
        }
    } 