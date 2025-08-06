"""
Enhanced metrics tracking for token savings and performance monitoring.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Metrics for a single compression operation."""
    timestamp: datetime
    original_tokens: int
    compressed_tokens: int
    original_length: int
    compressed_length: int
    compression_level: int
    compression_time_ms: float
    compression_ratio: float
    token_compression_ratio: float
    tokens_saved: int
    estimated_cost_savings: float
    endpoint: str
    user_id: Optional[str] = None
    api_key_hash: Optional[str] = None


@dataclass
class RequestMetrics:
    """Metrics for API requests."""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_id: Optional[str] = None
    api_key_hash: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class LLMMetrics:
    """Metrics for LLM API calls."""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_ms: float
    cost_estimate: float
    success: bool
    error_type: Optional[str] = None


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.compression_metrics: deque = deque(maxlen=max_history)
        self.request_metrics: deque = deque(maxlen=max_history)
        self.llm_metrics: deque = deque(maxlen=max_history)
        
        # Real-time aggregations
        self._hourly_stats = defaultdict(lambda: {
            "requests": 0,
            "total_tokens_saved": 0,
            "total_cost_savings": 0.0,
            "avg_compression_ratio": 0.0,
            "errors": 0
        })
        
        # Performance tracking
        self._performance_window = deque(maxlen=1000)  # Last 1000 requests
        
        self._lock = asyncio.Lock()
    
    async def record_compression(self, metrics: CompressionMetrics):
        """Record compression metrics."""
        async with self._lock:
            self.compression_metrics.append(metrics)
            
            # Update hourly stats
            hour_key = metrics.timestamp.replace(minute=0, second=0, microsecond=0)
            stats = self._hourly_stats[hour_key]
            stats["total_tokens_saved"] += metrics.tokens_saved
            stats["total_cost_savings"] += metrics.estimated_cost_savings
            
            # Update running average
            if stats["requests"] == 0:
                stats["avg_compression_ratio"] = metrics.compression_ratio
            else:
                # Exponential moving average
                alpha = 0.1
                stats["avg_compression_ratio"] = (
                    alpha * metrics.compression_ratio + 
                    (1 - alpha) * stats["avg_compression_ratio"]
                )
            
            logger.info(
                "Compression recorded",
                extra={
                    "tokens_saved": metrics.tokens_saved,
                    "compression_ratio": metrics.compression_ratio,
                    "cost_savings": metrics.estimated_cost_savings,
                    "compression_time_ms": metrics.compression_time_ms
                }
            )
    
    async def record_request(self, metrics: RequestMetrics):
        """Record request metrics."""
        async with self._lock:
            self.request_metrics.append(metrics)
            self._performance_window.append(metrics.response_time_ms)
            
            # Update hourly stats
            hour_key = metrics.timestamp.replace(minute=0, second=0, microsecond=0)
            stats = self._hourly_stats[hour_key]
            stats["requests"] += 1
            
            if metrics.status_code >= 400:
                stats["errors"] += 1
    
    async def record_llm_call(self, metrics: LLMMetrics):
        """Record LLM API call metrics."""
        async with self._lock:
            self.llm_metrics.append(metrics)
            
            logger.info(
                "LLM call recorded",
                extra={
                    "model": metrics.model,
                    "total_tokens": metrics.total_tokens,
                    "response_time_ms": metrics.response_time_ms,
                    "cost_estimate": metrics.cost_estimate,
                    "success": metrics.success
                }
            )
    
    def get_compression_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get compression statistics for the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.compression_metrics if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {"message": "No compression data available"}
        
        total_tokens_saved = sum(m.tokens_saved for m in recent_metrics)
        total_cost_savings = sum(m.estimated_cost_savings for m in recent_metrics)
        avg_compression_ratio = sum(m.compression_ratio for m in recent_metrics) / len(recent_metrics)
        avg_token_compression_ratio = sum(m.token_compression_ratio for m in recent_metrics) / len(recent_metrics)
        avg_compression_time = sum(m.compression_time_ms for m in recent_metrics) / len(recent_metrics)
        
        # Compression level breakdown
        level_stats = defaultdict(list)
        for m in recent_metrics:
            level_stats[m.compression_level].append(m.compression_ratio)
        
        level_breakdown = {}
        for level, ratios in level_stats.items():
            level_breakdown[f"level_{level}"] = {
                "count": len(ratios),
                "avg_compression_ratio": sum(ratios) / len(ratios)
            }
        
        return {
            "period_hours": hours,
            "total_compressions": len(recent_metrics),
            "total_tokens_saved": total_tokens_saved,
            "total_cost_savings": total_cost_savings,
            "average_compression_ratio": avg_compression_ratio,
            "average_token_compression_ratio": avg_token_compression_ratio,
            "average_compression_time_ms": avg_compression_time,
            "compression_levels": level_breakdown,
            "tokens_saved_per_hour": total_tokens_saved / hours,
            "cost_savings_per_hour": total_cost_savings / hours
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._performance_window:
            return {"message": "No performance data available"}
        
        response_times = list(self._performance_window)
        response_times.sort()
        
        n = len(response_times)
        p50 = response_times[n // 2]
        p95 = response_times[int(n * 0.95)]
        p99 = response_times[int(n * 0.99)]
        avg = sum(response_times) / n
        
        return {
            "total_requests": n,
            "avg_response_time_ms": avg,
            "p50_response_time_ms": p50,
            "p95_response_time_ms": p95,
            "p99_response_time_ms": p99,
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times)
        }
    
    def get_llm_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.llm_metrics if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {"message": "No LLM data available"}
        
        total_calls = len(recent_metrics)
        successful_calls = sum(1 for m in recent_metrics if m.success)
        total_tokens = sum(m.total_tokens for m in recent_metrics)
        total_cost = sum(m.cost_estimate for m in recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / total_calls
        
        # Model breakdown
        model_stats = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
        for m in recent_metrics:
            stats = model_stats[m.model]
            stats["calls"] += 1
            stats["tokens"] += m.total_tokens
            stats["cost"] += m.cost_estimate
        
        return {
            "period_hours": hours,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_response_time_ms": avg_response_time,
            "models": dict(model_stats),
            "calls_per_hour": total_calls / hours,
            "tokens_per_hour": total_tokens / hours,
            "cost_per_hour": total_cost / hours
        }
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly trend data."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Clean old data
        for hour_key in list(self._hourly_stats.keys()):
            if hour_key < cutoff:
                del self._hourly_stats[hour_key]
        
        # Sort by hour
        sorted_hours = sorted(self._hourly_stats.items())
        
        return {
            "hours": [hour.isoformat() for hour, _ in sorted_hours],
            "requests": [stats["requests"] for _, stats in sorted_hours],
            "tokens_saved": [stats["total_tokens_saved"] for _, stats in sorted_hours],
            "cost_savings": [stats["total_cost_savings"] for _, stats in sorted_hours],
            "avg_compression_ratio": [stats["avg_compression_ratio"] for _, stats in sorted_hours],
            "errors": [stats["errors"] for _, stats in sorted_hours]
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        data = {
            "compression_stats": self.get_compression_stats(),
            "performance_stats": self.get_performance_stats(),
            "llm_stats": self.get_llm_stats(),
            "hourly_trends": self.get_hourly_trends(),
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MetricsMiddleware:
    """Middleware to automatically collect request metrics."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def __call__(self, request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Estimate sizes (not perfectly accurate but good enough)
        request_size = len(str(request.url)) + sum(len(f"{k}: {v}") for k, v in request.headers.items())
        response_size = response.headers.get("content-length", 0)
        
        # Record metrics
        metrics = RequestMetrics(
            timestamp=datetime.utcnow(),
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size,
            response_size_bytes=int(response_size) if response_size else 0,
            user_id=getattr(request.state, "user_id", None),
            api_key_hash=getattr(request.state, "api_key_hash", None)
        )
        
        await self.collector.record_request(metrics)
        
        return response 