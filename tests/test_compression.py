#!/usr/bin/env python3
"""
Tests for compression modules.
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.stop_word_remover import StopWordRemover
from compression.keyword_extractor import KeywordExtractor
from compression.shorthand_compressor import ShorthandCompressor
from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import get_default_tokenizer


class TestStopWordRemover:
    """Test stop word removal compression."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.compressor = StopWordRemover()
    
    def test_basic_compression(self):
        """Test basic stop word removal."""
        text = "Please explain the concept of machine learning in simple terms."
        compressed = self.compressor.compress(text)
        
        # Should remove stop words like "the", "of", "in"
        assert "the" not in compressed.lower()
        assert "concept" in compressed
        assert "machine" in compressed
        assert "learning" in compressed
    
    def test_question_preservation(self):
        """Test that question words are preserved."""
        text = "How do I implement neural networks?"
        compressed = self.compressor.compress(text)
        
        assert "how" in compressed.lower()
        assert "implement" in compressed
        assert "neural" in compressed
        assert "networks" in compressed
    
    def test_empty_text(self):
        """Test handling of empty text."""
        assert self.compressor.compress("") == ""
        assert self.compressor.compress("   ") == "   "
    
    def test_decompression(self):
        """Test basic decompression."""
        text = "machine learning concepts"
        decompressed = self.compressor.decompress(text)
        
        # Should add some basic words back
        assert len(decompressed) >= len(text)
    
    def test_compression_stats(self):
        """Test compression statistics."""
        original = "Please explain the concept of machine learning."
        compressed = self.compressor.compress(original)
        stats = self.compressor.get_stats(original, compressed)
        
        assert stats['compressor'] == 'StopWordRemover'
        assert stats['original_length'] == len(original)
        assert stats['compressed_length'] == len(compressed)
        assert stats['compression_ratio'] <= 1.0
        assert stats['space_saved'] >= 0


class TestKeywordExtractor:
    """Test keyword extraction compression."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.compressor = KeywordExtractor()
    
    def test_keyword_extraction(self):
        """Test basic keyword extraction."""
        text = "Please explain machine learning algorithms and neural networks."
        compressed = self.compressor.compress(text)
        
        # Should extract important technical terms
        assert "machine" in compressed.lower()
        assert "learning" in compressed.lower()
        assert "neural" in compressed.lower()
        assert "networks" in compressed.lower()
    
    def test_action_word_preservation(self):
        """Test that action words are preserved."""
        text = "Please implement a neural network using Python."
        compressed = self.compressor.compress(text)
        
        assert "implement" in compressed.lower()
        assert "neural" in compressed.lower()
        assert "network" in compressed.lower()
    
    def test_technical_keywords(self):
        """Test preservation of technical keywords."""
        text = "Use the API to train the model with the dataset."
        compressed = self.compressor.compress(text)
        
        assert "api" in compressed.lower()
        assert "train" in compressed.lower()
        assert "model" in compressed.lower()
    
    def test_decompression_templates(self):
        """Test decompression template application."""
        compressed = "explain machine learning"
        decompressed = self.compressor.decompress(compressed)
        
        assert "explain" in decompressed.lower()
        assert "machine learning" in decompressed.lower()
        assert len(decompressed) > len(compressed)
    
    def test_max_keywords_limit(self):
        """Test that keyword extraction respects max limit."""
        config = {'max_keywords': 3}
        compressor = KeywordExtractor(config)
        
        text = "machine learning deep learning neural networks artificial intelligence algorithms optimization"
        compressed = compressor.compress(text)
        
        # Should not exceed max keywords significantly
        assert len(compressed.split()) <= 5  # Some flexibility for phrases


class TestShorthandCompressor:
    """Test aggressive shorthand compression."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.compressor = ShorthandCompressor()
    
    def test_abbreviation_replacement(self):
        """Test common abbreviation replacement."""
        text = "machine learning and artificial intelligence"
        compressed = self.compressor.compress(text)
        
        assert "ML" in compressed or "machine learning" in compressed
        assert "AI" in compressed or "artificial intelligence" in compressed
        assert "&" in compressed or "and" not in compressed
    
    def test_vowel_removal(self):
        """Test vowel removal."""
        text = "implementation"
        compressed = self.compressor.compress(text)
        
        # Should remove some vowels while maintaining readability
        assert len(compressed) < len(text)
        assert compressed[0] == text[0]  # First character preserved
    
    def test_space_removal(self):
        """Test space removal."""
        config = {'remove_spaces': True}
        compressor = ShorthandCompressor(config)
        
        text = "hello world test"
        compressed = compressor.compress(text)
        
        # Should remove or reduce spaces
        assert compressed.count(' ') <= text.count(' ')
    
    def test_decompression_restoration(self):
        """Test basic decompression restoration."""
        text = "ML algorithms"
        decompressed = self.compressor.decompress(text)
        
        # Should expand some abbreviations
        assert len(decompressed) >= len(text)
    
    def test_readability_preservation(self):
        """Test that compression maintains minimum readability."""
        text = "test"
        compressed = self.compressor.compress(text)
        
        # Very short words should be mostly preserved
        assert len(compressed) >= 2


class TestCompressionPipeline:
    """Test the main compression pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = get_default_tokenizer()
        self.pipeline = CompressionPipeline(self.tokenizer, compression_level=2)
    
    def test_pipeline_compression(self):
        """Test full pipeline compression."""
        text = "Please explain the concept of machine learning algorithms in detail."
        compressed = self.pipeline.compress(text)
        
        assert len(compressed) < len(text)
        assert compressed != text
    
    def test_compression_levels(self):
        """Test different compression levels."""
        text = "Please explain machine learning concepts in simple terms."
        
        results = {}
        for level in [1, 2, 3]:
            pipeline = CompressionPipeline(self.tokenizer, compression_level=level)
            results[level] = pipeline.compress(text)
        
        # Higher levels should generally produce more compression
        assert len(results[3]) <= len(results[2])
        assert len(results[2]) <= len(results[1])
    
    def test_compression_with_stats(self):
        """Test compression with detailed statistics."""
        text = "Please explain the concept of machine learning."
        result = self.pipeline.compress_with_stats(text)
        
        assert 'original' in result
        assert 'compressed' in result
        assert 'stats' in result
        
        stats = result['stats']
        assert 'compression_ratio' in stats
        assert 'token_compression_ratio' in stats
        assert 'steps' in stats
        assert len(stats['steps']) > 0
    
    def test_decompression(self):
        """Test pipeline decompression."""
        text = "machine learning concepts"
        compressed = self.pipeline.compress(text)
        decompressed = self.pipeline.decompress(compressed)
        
        # Decompression should expand the text
        assert len(decompressed) >= len(compressed)
    
    def test_cost_estimation(self):
        """Test cost savings estimation."""
        text = "Please explain machine learning algorithms in detail."
        cost_savings = self.pipeline.estimate_cost_savings(text)
        
        assert 'original_tokens' in cost_savings
        assert 'compressed_tokens' in cost_savings
        assert 'cost_savings' in cost_savings
        assert 'savings_percent' in cost_savings
        assert cost_savings['cost_savings'] >= 0
    
    def test_pipeline_stats_tracking(self):
        """Test pipeline statistics tracking."""
        text1 = "First test text for compression."
        text2 = "Second test text for evaluation."
        
        self.pipeline.compress(text1)
        self.pipeline.compress(text2)
        
        stats = self.pipeline.get_pipeline_stats()
        assert stats['total_compressions'] == 2
        assert stats['total_original_length'] > 0
        assert stats['total_compressed_length'] > 0
    
    def test_stats_reset(self):
        """Test pipeline statistics reset."""
        text = "Test text for compression."
        self.pipeline.compress(text)
        
        # Should have some stats
        stats_before = self.pipeline.get_pipeline_stats()
        assert stats_before['total_compressions'] > 0
        
        # Reset and check
        self.pipeline.reset_stats()
        stats_after = self.pipeline.get_pipeline_stats()
        assert stats_after['total_compressions'] == 0
    
    def test_empty_input_handling(self):
        """Test handling of empty and whitespace inputs."""
        assert self.pipeline.compress("") == ""
        assert self.pipeline.compress("   ") == "   "
        assert self.pipeline.decompress("") == ""
    
    def test_roundtrip_similarity(self):
        """Test that compress->decompress maintains some similarity."""
        text = "Explain machine learning concepts"
        compressed = self.pipeline.compress(text)
        decompressed = self.pipeline.decompress(compressed)
        
        # Should contain key concepts
        key_words = ["machine", "learning", "concept"]
        for word in key_words:
            if word in text.lower():
                assert word in decompressed.lower() or word[:4] in decompressed.lower()


class TestCompressionIntegration:
    """Integration tests for compression system."""
    
    def test_multiple_texts_consistency(self):
        """Test that compression is consistent across multiple texts."""
        tokenizer = get_default_tokenizer()
        pipeline = CompressionPipeline(tokenizer, compression_level=2)
        
        texts = [
            "Explain machine learning algorithms",
            "How to implement neural networks",
            "What are the best practices for AI development"
        ]
        
        for text in texts:
            # Compress the same text twice
            result1 = pipeline.compress(text)
            result2 = pipeline.compress(text)
            
            # Should be deterministic
            assert result1 == result2
    
    def test_compression_effectiveness(self):
        """Test that compression is actually effective."""
        tokenizer = get_default_tokenizer()
        pipeline = CompressionPipeline(tokenizer, compression_level=2)
        
        # Long text should show significant compression
        long_text = (
            "Please provide a comprehensive explanation of machine learning algorithms "
            "including the fundamental concepts, mathematical foundations, and practical "
            "applications in real-world scenarios. I would like to understand the "
            "differences between supervised and unsupervised learning approaches."
        )
        
        result = pipeline.compress_with_stats(long_text)
        stats = result['stats']
        
        # Should achieve some compression
        assert stats['compression_ratio'] < 0.9  # At least 10% compression
        assert stats['tokens_saved'] > 0
    
    def test_quality_preservation(self):
        """Test that compression preserves essential meaning."""
        tokenizer = get_default_tokenizer()
        pipeline = CompressionPipeline(tokenizer, compression_level=1)  # Conservative level
        
        text = "How do I implement a neural network using Python?"
        compressed = pipeline.compress(text)
        
        # Essential keywords should be preserved
        essential_words = ["implement", "neural", "network", "python"]
        for word in essential_words:
            assert word.lower() in compressed.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 