"""
Main compression pipeline that orchestrates multiple compression techniques.
"""

import re
from typing import Dict, Any, List, Optional, Union

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .base import BaseCompressor
from .stop_word_remover import StopWordRemover
from .keyword_extractor import KeywordExtractor
from .shorthand_compressor import ShorthandCompressor

# Helper functions for deduplication and normalization
_DUP_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.I)

def _collapse_dupes(s: str) -> str:
    return _DUP_WORD_RE.sub(r"\1", s).strip()

def _token_count(tokenizer, s: str) -> int:
    if not tokenizer:
        return len(s.split())
    if hasattr(tokenizer, 'encode'):
        if hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.encode(s, add_special_tokens=True)
            return len(tokens)
        else:
            encoded = tokenizer.encode(s)
            return len(encoded.ids)
    return len(s.split())


class CompressionPipeline:
    """
    Main compression pipeline that combines multiple compression techniques.
    Uses replace-then-transform pattern with deduplication and monotonic guarantees.
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
        self.level = int(compression_level)
        self.config = config or {}

        # Initialize compressors
        self.stopper = StopWordRemover(self.config.get('stop_word_remover', {}))
        self.keywords = KeywordExtractor(self.config.get('keyword_extractor', {'max_keywords': 3}))
        self.shorthand = ShorthandCompressor(self.config.get('shorthand_compressor', {}))

        # Track compression statistics
        self._stats = {
            "total_compressions": 0,
            "total_original_length": 0,
            "total_compressed_length": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
        }

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

        orig = text
        orig_tokens = _token_count(self.tokenizer, orig)

        # Stage 1 (transform): Stop word removal
        s1 = _collapse_dupes(self.stopper.compress(orig))

        # Stage 2 (REPLACE): Keyword extraction replaces previous text
        s2 = _collapse_dupes(self.keywords.compress(s1))

        # Stage 3 (transform) only if level >=3: Shorthand compression
        s3 = s2
        if self.level >= 3:
            s3 = _collapse_dupes(self.shorthand.compress(s2))

        # Choose by level and apply never-longer guard
        if self.level == 1:
            out = s1
        elif self.level == 2:
            out = s2
        else:
            out = s3

        # Never-longer guard (chars & tokens)
        if len(out) >= len(orig) or _token_count(self.tokenizer, out) >= orig_tokens:
            # Back off one stage
            if self.level == 3:
                out = s2
            elif self.level == 2:
                out = s1
            else:
                out = s1

        # Update stats
        self._stats["total_compressions"] += 1
        self._stats["total_original_length"] += len(orig)
        self._stats["total_compressed_length"] += len(out)
        self._stats["total_original_tokens"] += orig_tokens
        self._stats["total_compressed_tokens"] += _token_count(self.tokenizer, out)

        return out

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

        # Only keyword decompression is implemented with templates
        return self.keywords.decompress(compressed_text)

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

        orig = text
        orig_tokens = _token_count(self.tokenizer, orig)

        # Stage 1 (transform): Stop word removal
        s1 = _collapse_dupes(self.stopper.compress(orig))

        # Stage 2 (REPLACE): Keyword extraction replaces previous text
        s2 = _collapse_dupes(self.keywords.compress(s1))

        # Stage 3 (transform) only if level >=3: Shorthand compression
        s3 = s2
        if self.level >= 3:
            s3 = _collapse_dupes(self.shorthand.compress(s2))

        # Choose by level and apply never-longer guard
        if self.level == 1:
            out = s1
        elif self.level == 2:
            out = s2
        else:
            out = s3

        # Never-longer guard (chars & tokens)
        if len(out) >= len(orig) or _token_count(self.tokenizer, out) >= orig_tokens:
            # Back off one stage
            if self.level == 3:
                out = s2
            elif self.level == 2:
                out = s1
            else:
                out = s1

        final_length = len(out)
        final_tokens = _token_count(self.tokenizer, out)

        stats = {
            'original_length': len(orig),
            'compressed_length': final_length,
            'original_tokens': orig_tokens,
            'compressed_tokens': final_tokens,
            'compression_ratio': final_length / len(orig) if len(orig) > 0 else 1.0,
            'token_compression_ratio': final_tokens / orig_tokens if orig_tokens > 0 else 1.0,
            'space_saved': len(orig) - final_length,
            'tokens_saved': orig_tokens - final_tokens,
            'space_saved_percent': ((len(orig) - final_length) / len(orig) * 100) if len(orig) > 0 else 0,
            'tokens_saved_percent': ((orig_tokens - final_tokens) / orig_tokens * 100) if orig_tokens > 0 else 0,
            'steps': [
                {'stage': 1, 'name': 'StopWordRemover', 'input': orig, 'output': s1},
                {'stage': 2, 'name': 'KeywordExtractor', 'input': s1, 'output': s2},
                {'stage': 3, 'name': 'ShorthandCompressor', 'input': s2, 'output': s3},
                {'final': out}
            ]
        }

        # Update stats
        self._stats["total_compressions"] += 1
        self._stats["total_original_length"] += len(orig)
        self._stats["total_compressed_length"] += final_length
        self._stats["total_original_tokens"] += orig_tokens
        self._stats["total_compressed_tokens"] += final_tokens

        return {
            'original': orig,
            'compressed': out,
            'stats': stats
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        if self._stats['total_compressions'] == 0:
            return self._stats

        stats = self._stats.copy()
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
        self._stats = {
            "total_compressions": 0,
            "total_original_length": 0,
            "total_compressed_length": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0
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
        original_tokens = _token_count(self.tokenizer, text)
        compressed_tokens = _token_count(self.tokenizer, self.compress(text))

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