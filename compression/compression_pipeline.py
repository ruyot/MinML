"""
Main compression pipeline that orchestrates multiple compression techniques.
"""

import os
from typing import Dict, Any, List, Optional, Union

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .base import BaseCompressor
from .stop_word_remover import StopWordRemover
from .keyword_extractor import KeywordExtractor
from .shorthand_compressor import ShorthandCompressor


class CompressionPipeline:
    """
    Main compression pipeline that combines multiple compression techniques.
    """
    
    def __init__(self, 
                 tokenizer: Optional[Union[Tokenizer, AutoTokenizer]] = None,
                 compression_level: int = 2,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize compression pipeline.
        
        Args:
            tokenizer: Tokenizer for token counting
            compression_level: Compression aggressiveness (1-3)
            config: Configuration for individual compressors
        """
        self.tokenizer = tokenizer
        self.compression_level = compression_level
        self.config = config or {}
        
        # Initialize compressors based on level
        self.compressors = self._setup_compressors()
        
        # Track compression statistics
        self.stats = {
            'total_compressions': 0,
            'total_original_length': 0,
            'total_compressed_length': 0,
            'total_original_tokens': 0,
            'total_compressed_tokens': 0
        }
    
    def _setup_compressors(self) -> List[BaseCompressor]:
        """Setup compressors based on compression level."""
        compressors = []
        
        # Level 1: Stop word removal only
        if self.compression_level >= 1:
            stop_word_config = self.config.get('stop_word_remover', {})
            compressors.append(StopWordRemover(stop_word_config))
        
        # Level 2: Add keyword extraction
        if self.compression_level >= 2:
            keyword_config = self.config.get('keyword_extractor', {})
            compressors.append(KeywordExtractor(keyword_config))
        
        # Level 3: Add aggressive shorthand
        if self.compression_level >= 3:
            shorthand_config = self.config.get('shorthand_compressor', {})
            compressors.append(ShorthandCompressor(shorthand_config))
        
        return compressors
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text."""
        if not self.tokenizer:
            # Fallback to word count approximation
            return len(text.split())
        
        if hasattr(self.tokenizer, 'encode'):
            # Hugging Face tokenizer
            if hasattr(self.tokenizer, 'tokenize'):
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            else:
                # Custom tokenizer
                encoded = self.tokenizer.encode(text)
                return len(encoded.ids)
        
        return len(text.split())
    
    def compress(self, text: str) -> str:
        """
        Compress text using the configured pipeline.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed text
        """
        if not text.strip():
            return text
        
        result = text
        compression_steps = []
        
        # Apply compressors in sequence
        for compressor in self.compressors:
            step_input = result
            result = compressor.compress(result)
            
            step_stats = compressor.get_stats(step_input, result)
            compression_steps.append(step_stats)
        
        # Update overall statistics
        self._update_stats(text, result)
        
        return result
    
    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to decompress text (best effort approximation).
        
        Args:
            compressed_text: Compressed text
            
        Returns:
            Decompressed text approximation
        """
        if not compressed_text.strip():
            return compressed_text
        
        result = compressed_text
        
        # Apply decompressors in reverse order
        for compressor in reversed(self.compressors):
            result = compressor.decompress(result)
        
        return result
    
    def compress_with_stats(self, text: str) -> Dict[str, Any]:
        """
        Compress text and return detailed statistics.
        
        Args:
            text: Input text to compress
            
        Returns:
            Dictionary with compressed text and statistics
        """
        if not text.strip():
            return {
                'original': text,
                'compressed': text,
                'stats': {'compression_ratio': 1.0, 'steps': []}
            }
        
        result = text
        compression_steps = []
        
        # Track original metrics
        original_length = len(text)
        original_tokens = self._get_token_count(text)
        
        # Apply compressors and track each step
        for compressor in self.compressors:
            step_input = result
            result = compressor.compress(result)
            
            step_stats = compressor.get_stats(step_input, result)
            step_stats['tokens_before'] = self._get_token_count(step_input)
            step_stats['tokens_after'] = self._get_token_count(result)
            compression_steps.append(step_stats)
        
        # Calculate final metrics
        final_length = len(result)
        final_tokens = self._get_token_count(result)
        
        stats = {
            'original_length': original_length,
            'compressed_length': final_length,
            'original_tokens': original_tokens,
            'compressed_tokens': final_tokens,
            'compression_ratio': final_length / original_length if original_length > 0 else 1.0,
            'token_compression_ratio': final_tokens / original_tokens if original_tokens > 0 else 1.0,
            'space_saved': original_length - final_length,
            'tokens_saved': original_tokens - final_tokens,
            'space_saved_percent': ((original_length - final_length) / original_length * 100) if original_length > 0 else 0,
            'tokens_saved_percent': ((original_tokens - final_tokens) / original_tokens * 100) if original_tokens > 0 else 0,
            'steps': compression_steps
        }
        
        self._update_stats(text, result)
        
        return {
            'original': text,
            'compressed': result,
            'stats': stats
        }
    
    def _update_stats(self, original: str, compressed: str):
        """Update pipeline statistics."""
        self.stats['total_compressions'] += 1
        self.stats['total_original_length'] += len(original)
        self.stats['total_compressed_length'] += len(compressed)
        
        if self.tokenizer:
            self.stats['total_original_tokens'] += self._get_token_count(original)
            self.stats['total_compressed_tokens'] += self._get_token_count(compressed)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        if self.stats['total_compressions'] == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats['average_compression_ratio'] = (
            stats['total_compressed_length'] / stats['total_original_length']
            if stats['total_original_length'] > 0 else 1.0
        )
        
        if stats['total_original_tokens'] > 0:
            stats['average_token_compression_ratio'] = (
                stats['total_compressed_tokens'] / stats['total_original_tokens']
            )
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self.stats = {
            'total_compressions': 0,
            'total_original_length': 0,
            'total_compressed_length': 0,
            'total_original_tokens': 0,
            'total_compressed_tokens': 0
        }
    
    def estimate_cost_savings(self, text: str, cost_per_token: float = 0.00002) -> Dict[str, float]:
        """
        Estimate cost savings from compression.
        
        Args:
            text: Input text
            cost_per_token: Cost per token (default for GPT-3.5-turbo)
            
        Returns:
            Dictionary with cost estimates
        """
        original_tokens = self._get_token_count(text)
        compressed_tokens = self._get_token_count(self.compress(text))
        
        original_cost = original_tokens * cost_per_token
        compressed_cost = compressed_tokens * cost_per_token
        savings = original_cost - compressed_cost
        
        return {
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'original_cost': original_cost,
            'compressed_cost': compressed_cost,
            'cost_savings': savings,
            'savings_percent': (savings / original_cost * 100) if original_cost > 0 else 0
        } 