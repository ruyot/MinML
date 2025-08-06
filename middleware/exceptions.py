"""
Enhanced error handling with structured responses and timeouts.
"""

import traceback
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Structured error detail."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Request trace ID")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: ErrorDetail
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class CompressionError(Exception):
    """Base exception for compression-related errors."""
    
    def __init__(self, message: str, code: str = "COMPRESSION_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class TokenizerError(Exception):
    """Exception for tokenizer-related errors."""
    
    def __init__(self, message: str, code: str = "TOKENIZER_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class LLMError(Exception):
    """Exception for LLM-related errors."""
    
    def __init__(self, message: str, code: str = "LLM_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class TimeoutError(Exception):
    """Exception for timeout errors."""
    
    def __init__(self, message: str, timeout_seconds: int, operation: str = "unknown"):
        self.message = message
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(message)


async def timeout_handler(coro, timeout_seconds: int, operation: str = "operation"):
    """Handle async operations with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"{operation} timed out after {timeout_seconds} seconds",
            timeout_seconds=timeout_seconds,
            operation=operation
        )


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None
) -> JSONResponse:
    """Create standardized error response."""
    
    error_detail = ErrorDetail(
        code=error_code,
        message=message,
        details=details,
        trace_id=trace_id
    )
    
    error_response = ErrorResponse(
        error=error_detail,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for standardized error responses."""
    
    # Extract request info
    request_id = getattr(request.state, "request_id", None)
    trace_id = request.headers.get("X-Trace-ID")
    
    # Log the exception
    logger.error(
        f"Unhandled exception in {request.method} {request.url}: {exc}",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "exception": str(exc),
            "traceback": traceback.format_exc()
        }
    )
    
    # Handle different exception types
    if isinstance(exc, CompressionError):
        return create_error_response(
            error_code=exc.code,
            message=exc.message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=exc.details,
            request_id=request_id,
            trace_id=trace_id
        )
    
    elif isinstance(exc, TokenizerError):
        return create_error_response(
            error_code=exc.code,
            message=exc.message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=exc.details,
            request_id=request_id,
            trace_id=trace_id
        )
    
    elif isinstance(exc, LLMError):
        return create_error_response(
            error_code=exc.code,
            message=exc.message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=exc.details,
            request_id=request_id,
            trace_id=trace_id
        )
    
    elif isinstance(exc, TimeoutError):
        return create_error_response(
            error_code="TIMEOUT_ERROR",
            message=exc.message,
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            details={
                "timeout_seconds": exc.timeout_seconds,
                "operation": exc.operation
            },
            request_id=request_id,
            trace_id=trace_id
        )
    
    elif isinstance(exc, HTTPException):
        return create_error_response(
            error_code="HTTP_ERROR",
            message=exc.detail,
            status_code=exc.status_code,
            details={"headers": exc.headers} if exc.headers else None,
            request_id=request_id,
            trace_id=trace_id
        )
    
    elif isinstance(exc, RequestValidationError):
        return create_error_response(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"validation_errors": exc.errors()},
            request_id=request_id,
            trace_id=trace_id
        )
    
    elif isinstance(exc, ValueError):
        return create_error_response(
            error_code="VALUE_ERROR",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
            request_id=request_id,
            trace_id=trace_id
        )
    
    # Generic server error
    return create_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"exception_type": type(exc).__name__} if not request.state.__dict__.get("is_production", True) else None,
        request_id=request_id,
        trace_id=trace_id
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors with detailed information."""
    
    request_id = getattr(request.state, "request_id", None)
    trace_id = request.headers.get("X-Trace-ID")
    
    # Process validation errors for better user experience
    simplified_errors = []
    for error in exc.errors():
        simplified_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={
            "errors": simplified_errors,
            "error_count": len(simplified_errors)
        },
        request_id=request_id,
        trace_id=trace_id
    ) 