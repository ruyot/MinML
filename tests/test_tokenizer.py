#!/usr/bin/env python3
"""
Tests for tokenizer functionality.
"""

import pytest
import os
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer_utils import (
    load_tokenizer, get_token_count, tokenize_text, 
    encode_text, decode_tokens, compare_tokenizers,
    get_default_tokenizer
)
from tokenizer.trainer import train_tokenizer, load_corpus


class TestTokenizerUtils:
    """Test tokenizer utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = get_default_tokenizer()
    
    def test_get_default_tokenizer(self):
        """Test default tokenizer loading."""
        tokenizer = get_default_tokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, 'encode') or hasattr(tokenizer, 'tokenize')
    
    def test_get_token_count(self):
        """Test token counting functionality."""
        text = "Hello world, this is a test."
        count = get_token_count(text, self.tokenizer)
        
        assert isinstance(count, int)
        assert count > 0
        assert count <= len(text.split()) + 10  # Allow for special tokens
    
    def test_tokenize_text(self):
        """Test text tokenization."""
        text = "Hello world"
        tokens = tokenize_text(text, self.tokenizer)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        text = "Hello world, this is a test."
        
        # Encode
        token_ids = encode_text(text, self.tokenizer)
        assert isinstance(token_ids, list)
        assert all(isinstance(id_, int) for id_ in token_ids)
        
        # Decode
        decoded = decode_tokens(token_ids, self.tokenizer)
        assert isinstance(decoded, str)
        
        # Should be similar (not exact due to tokenization)
        assert len(decoded) > 0
    
    def test_compare_tokenizers(self):
        """Test tokenizer comparison."""
        text = "Machine learning is fascinating."
        tokenizer1 = get_default_tokenizer()
        tokenizer2 = get_default_tokenizer()  # Same tokenizer for test
        
        comparison = compare_tokenizers(text, tokenizer1, tokenizer2)
        
        assert 'text' in comparison
        assert 'tokenizer1' in comparison
        assert 'tokenizer2' in comparison
        assert 'difference' in comparison
        assert 'compression_ratio' in comparison
        
        # Same tokenizers should have no difference
        assert comparison['difference'] == 0
        assert comparison['compression_ratio'] == 0
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        assert get_token_count("", self.tokenizer) >= 0
        
        tokens = tokenize_text("", self.tokenizer)
        assert isinstance(tokens, list)
        
        token_ids = encode_text("", self.tokenizer)
        assert isinstance(token_ids, list)


class TestTokenizerTrainer:
    """Test tokenizer training functionality."""
    
    def test_load_corpus(self):
        """Test corpus loading from file."""
        # Create temporary corpus file
        corpus_content = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks."
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for line in corpus_content:
                f.write(line + '\n')
            corpus_path = f.name
        
        try:
            loaded_corpus = load_corpus(corpus_path)
            assert len(loaded_corpus) == len(corpus_content)
            assert loaded_corpus == corpus_content
        finally:
            os.unlink(corpus_path)
    
    def test_train_small_tokenizer(self):
        """Test training a small tokenizer."""
        # Create a minimal corpus
        corpus = [
            "machine learning algorithms",
            "neural network training",
            "deep learning models",
            "artificial intelligence systems",
            "natural language processing",
            "computer vision applications"
        ]
        
        # Create temporary corpus file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for line in corpus:
                f.write(line + '\n')
            corpus_path = f.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Train a small tokenizer
                tokenizer = train_tokenizer(
                    corpus_path=corpus_path,
                    output_dir=output_dir,
                    vocab_size=100,  # Very small for testing
                    min_frequency=1
                )
                
                assert tokenizer is not None
                
                # Test the trained tokenizer
                test_text = "machine learning"
                tokens = tokenizer.encode(test_text)
                assert len(tokens.ids) > 0
                assert len(tokens.tokens) > 0
                
                # Check vocabulary
                vocab = tokenizer.get_vocab()
                assert len(vocab) <= 100
                assert "[UNK]" in vocab
                
                # Test tokenizer files were created
                tokenizer_file = os.path.join(output_dir, "tokenizer.json")
                vocab_file = os.path.join(output_dir, "vocab.txt")
                assert os.path.exists(tokenizer_file)
                assert os.path.exists(vocab_file)
                
            finally:
                os.unlink(corpus_path)
    
    def test_train_tokenizer_with_special_tokens(self):
        """Test training with custom special tokens."""
        corpus = ["hello world", "machine learning", "neural networks"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for line in corpus:
                f.write(line + '\n')
            corpus_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[CUSTOM]"]
                
                tokenizer = train_tokenizer(
                    corpus_path=corpus_path,
                    output_dir=output_dir,
                    vocab_size=50,
                    special_tokens=special_tokens
                )
                
                vocab = tokenizer.get_vocab()
                
                # Special tokens should be in vocabulary
                for token in special_tokens:
                    assert token in vocab
                    
            finally:
                os.unlink(corpus_path)
    
    def test_load_trained_tokenizer(self):
        """Test loading a trained tokenizer."""
        # First train a tokenizer
        corpus = ["test corpus for tokenizer training"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(corpus[0])
            corpus_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Train
                train_tokenizer(
                    corpus_path=corpus_path,
                    output_dir=output_dir,
                    vocab_size=30
                )
                
                # Load the trained tokenizer
                loaded_tokenizer = load_tokenizer(output_dir)
                assert loaded_tokenizer is not None
                
                # Test functionality
                test_text = "test tokenizer"
                tokens = loaded_tokenizer.encode(test_text)
                assert len(tokens.ids) > 0
                
            finally:
                os.unlink(corpus_path)


class TestTokenizerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_load_nonexistent_tokenizer(self):
        """Test loading from nonexistent path."""
        with pytest.raises(ValueError):
            load_tokenizer("/nonexistent/path")
    
    def test_very_short_corpus(self):
        """Test training with very short corpus."""
        corpus = ["a", "b"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for line in corpus:
                f.write(line + '\n')
            corpus_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Should handle gracefully
                tokenizer = train_tokenizer(
                    corpus_path=corpus_path,
                    output_dir=output_dir,
                    vocab_size=10,
                    min_frequency=1
                )
                assert tokenizer is not None
                
            finally:
                os.unlink(corpus_path)
    
    def test_large_vocab_size(self):
        """Test training with vocabulary larger than corpus."""
        corpus = ["small corpus"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(corpus[0])
            corpus_path = f.name
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Request vocab larger than possible
                tokenizer = train_tokenizer(
                    corpus_path=corpus_path,
                    output_dir=output_dir,
                    vocab_size=1000,  # Much larger than corpus
                    min_frequency=1
                )
                
                # Should work but vocab will be smaller
                vocab = tokenizer.get_vocab()
                assert len(vocab) < 1000
                
            finally:
                os.unlink(corpus_path)
    
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        tokenizer = get_default_tokenizer()
        
        unicode_text = "Hello 世界 🌍 café naïve"
        
        # Should handle unicode gracefully
        count = get_token_count(unicode_text, tokenizer)
        assert count > 0
        
        tokens = tokenize_text(unicode_text, tokenizer)
        assert len(tokens) > 0
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        tokenizer = get_default_tokenizer()
        
        # Create very long text
        long_text = "word " * 1000
        
        count = get_token_count(long_text, tokenizer)
        assert count > 0
        
        tokens = tokenize_text(long_text, tokenizer)
        assert len(tokens) > 0


class TestTokenizerPerformance:
    """Test tokenizer performance characteristics."""
    
    def test_tokenization_speed(self):
        """Test that tokenization completes in reasonable time."""
        import time
        
        tokenizer = get_default_tokenizer()
        text = "This is a test sentence for performance measurement. " * 100
        
        start_time = time.time()
        for _ in range(10):
            get_token_count(text, tokenizer)
        end_time = time.time()
        
        # Should complete quickly (less than 5 seconds for 10 iterations)
        assert (end_time - start_time) < 5.0
    
    def test_memory_efficiency(self):
        """Test that tokenizer doesn't consume excessive memory."""
        tokenizer = get_default_tokenizer()
        
        # Process multiple texts
        texts = [f"Test text number {i} for memory testing." for i in range(100)]
        
        for text in texts:
            get_token_count(text, tokenizer)
            tokenize_text(text, tokenizer)
        
        # If we get here without memory errors, test passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 