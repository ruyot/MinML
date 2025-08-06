#!/usr/bin/env python3
"""
Tests for FastAPI middleware endpoints.
"""

import pytest
import asyncio
import os
import sys
from httpx import AsyncClient
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware.main import app


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "compression_levels_available" in data
        assert data["compression_levels_available"] == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert isinstance(data["tokenizer_loaded"], bool)


class TestCompressionEndpoints:
    """Test compression-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_compress_endpoint(self):
        """Test text compression endpoint."""
        payload = {
            "text": "Please explain machine learning concepts in simple terms.",
            "compression_level": 2,
            "include_stats": False
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "original" in data
        assert "compressed" in data
        assert data["original"] == payload["text"]
        assert len(data["compressed"]) <= len(data["original"])
    
    @pytest.mark.asyncio
    async def test_compress_with_stats(self):
        """Test compression with statistics."""
        payload = {
            "text": "Please explain machine learning algorithms and neural networks.",
            "compression_level": 2,
            "include_stats": True
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert data["stats"] is not None
        
        stats = data["stats"]
        assert "compression_ratio" in stats
        assert "token_compression_ratio" in stats
        assert "original_tokens" in stats
        assert "compressed_tokens" in stats
        assert "steps" in stats
    
    @pytest.mark.asyncio
    async def test_compress_different_levels(self):
        """Test compression with different levels."""
        text = "Please explain machine learning concepts in detail."
        
        results = {}
        for level in [1, 2, 3]:
            payload = {
                "text": text,
                "compression_level": level,
                "include_stats": False
            }
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post("/compress", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            results[level] = data["compressed"]
        
        # Higher levels should generally produce more compression
        assert len(results[3]) <= len(results[2])
    
    @pytest.mark.asyncio
    async def test_decompress_endpoint(self):
        """Test text decompression endpoint."""
        # First compress something
        compress_payload = {
            "text": "machine learning concepts",
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            compress_response = await ac.post("/compress", json=compress_payload)
        
        assert compress_response.status_code == 200
        compressed_text = compress_response.json()["compressed"]
        
        # Now decompress
        decompress_payload = {
            "compressed_text": compressed_text,
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            decompress_response = await ac.post("/decompress", json=decompress_payload)
        
        assert decompress_response.status_code == 200
        data = decompress_response.json()
        assert "compressed" in data
        assert "decompressed" in data
        assert data["compressed"] == compressed_text
        assert len(data["decompressed"]) >= len(compressed_text)
    
    @pytest.mark.asyncio
    async def test_compress_empty_text(self):
        """Test compression of empty text."""
        payload = {
            "text": "",
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        # Should return 422 due to validation (min_length=1)
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_compress_invalid_level(self):
        """Test compression with invalid level."""
        payload = {
            "text": "test text",
            "compression_level": 5  # Invalid level
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        # Should return 422 due to validation
        assert response.status_code == 422


class TestLLMQueryEndpoints:
    """Test LLM query endpoints."""
    
    @pytest.mark.asyncio
    async def test_compress_and_query_mock(self):
        """Test compress and query endpoint (with mock response)."""
        payload = {
            "prompt": "Explain machine learning concepts",
            "model": "gpt-3.5-turbo",
            "use_compression": True,
            "compression_level": 2,
            "include_compression_stats": True
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress-and-query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "original_prompt" in data
        assert "compressed_prompt" in data
        assert "llm_response" in data
        assert "model_used" in data
        assert "compression_stats" in data
        assert "usage_stats" in data
        
        assert data["original_prompt"] == payload["prompt"]
        assert data["compressed_prompt"] is not None
        assert len(data["compressed_prompt"]) <= len(data["original_prompt"])
    
    @pytest.mark.asyncio
    async def test_compress_and_query_without_compression(self):
        """Test query endpoint without compression."""
        payload = {
            "prompt": "Explain machine learning",
            "use_compression": False
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress-and-query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["original_prompt"] == payload["prompt"]
        assert data["compressed_prompt"] is None
        assert data["compression_stats"] is None
    
    @pytest.mark.asyncio
    async def test_compress_and_query_parameters(self):
        """Test query endpoint with various parameters."""
        payload = {
            "prompt": "Explain AI",
            "model": "gpt-4",
            "max_tokens": 100,
            "temperature": 0.5,
            "use_compression": True
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress-and-query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "llm_response" in data


class TestEvaluationEndpoints:
    """Test evaluation endpoints."""
    
    @pytest.mark.asyncio
    async def test_evaluate_endpoint(self):
        """Test compression evaluation endpoint."""
        payload = {
            "texts": [
                "Explain machine learning concepts",
                "How to implement neural networks",
                "What are AI best practices"
            ],
            "compression_level": 2,
            "cost_per_token": 0.00002
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/evaluate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_texts" in data
        assert "average_compression_ratio" in data
        assert "average_token_compression_ratio" in data
        assert "total_original_tokens" in data
        assert "total_compressed_tokens" in data
        assert "estimated_cost_savings" in data
        assert "savings_percent" in data
        assert "detailed_results" in data
        
        assert data["total_texts"] == len(payload["texts"])
        assert len(data["detailed_results"]) == len(payload["texts"])
    
    @pytest.mark.asyncio
    async def test_evaluate_single_text(self):
        """Test evaluation with single text."""
        payload = {
            "texts": ["Single text for evaluation"],
            "compression_level": 1
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/evaluate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_texts"] == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_empty_list(self):
        """Test evaluation with empty text list."""
        payload = {
            "texts": [],
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/evaluate", json=payload)
        
        # Should return 422 due to validation (min_items=1)
        assert response.status_code == 422


class TestStatsEndpoints:
    """Test statistics endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting pipeline statistics."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should contain stats structure
        assert isinstance(data, dict)
        assert "total_compressions" in data
    
    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test resetting pipeline statistics."""
        # First do some compression to generate stats
        payload = {
            "text": "Test text for stats",
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            await ac.post("/compress", json=payload)
            
            # Reset stats
            reset_response = await ac.post("/reset-stats")
            assert reset_response.status_code == 200
            
            # Check stats are reset
            stats_response = await ac.get("/stats")
            assert stats_response.status_code == 200
            data = stats_response.json()
            assert data["total_compressions"] == 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/compress",
                content="invalid json",
                headers={"Content-Type": "application/json"}
            )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        payload = {
            "compression_level": 2
            # Missing required "text" field
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_text_too_long(self):
        """Test handling of text that's too long."""
        payload = {
            "text": "a" * 20000,  # Exceeds max_length=10000
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self):
        """Test accessing nonexistent endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/nonexistent")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_wrong_http_method(self):
        """Test using wrong HTTP method."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/compress")  # Should be POST
        
        assert response.status_code == 405


class TestInputValidation:
    """Test input validation edge cases."""
    
    @pytest.mark.asyncio
    async def test_boundary_values(self):
        """Test boundary values for parameters."""
        # Test minimum compression level
        payload = {
            "text": "test",
            "compression_level": 1
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 200
        
        # Test maximum compression level
        payload["compression_level"] = 3
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test handling of unicode characters."""
        payload = {
            "text": "Hello 世界 🌍 café naïve résumé",
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["compressed"]) > 0
    
    @pytest.mark.asyncio
    async def test_whitespace_text(self):
        """Test handling of whitespace-only text."""
        payload = {
            "text": "   \n\t   ",  # Whitespace only
            "compression_level": 2
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/compress", json=payload)
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 