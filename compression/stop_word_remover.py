"""
Stop word removal compression technique.
"""

import re
from typing import Dict, Any, Set, Optional

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from .base import BaseCompressor


class StopWordRemover(BaseCompressor):
    """
    Compressor that removes stop words while preserving sentence structure.
    """
    
    # Default English stop words for fallback
    DEFAULT_STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
        'up', 'out', 'so', 'can', 'her', 'would', 'make', 'like', 'him',
        'into', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than',
        'first', 'been', 'call', 'who', 'oil', 'its', 'now', 'find',
        'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize stop word remover.
        
        Args:
            config: Configuration with optional 'language' and 'preserve_structure' settings
        """
        super().__init__(config)
        self.language = self.config.get('language', 'english')
        self.preserve_structure = self.config.get('preserve_structure', True)
        self.min_word_length = self.config.get('min_word_length', 2)
        
        # Load stop words
        self.stop_words = self._load_stop_words()
        
        # Patterns for sentence structure preservation
        self.sentence_starters = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'whose',
            'please', 'can', 'could', 'would', 'should', 'explain', 'describe',
            'implement', 'create', 'build', 'write', 'help', 'show', 'tell'
        }
        
        # Question words to always preserve
        self.question_words = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'whose'
        }
    
    def _load_stop_words(self) -> Set[str]:
        """Load stop words from NLTK or use default set."""
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                return set(stopwords.words(self.language))
            except Exception:
                pass
        
        # Fallback to default English stop words
        return self.DEFAULT_STOP_WORDS
    
    def _should_preserve_word(self, word: str, position: int, is_question: bool) -> bool:
        """
        Determine if a word should be preserved based on context.
        
        Args:
            word: The word to check
            position: Position in the sentence (0-based)
            is_question: Whether the sentence is a question
            
        Returns:
            True if the word should be preserved
        """
        word_lower = word.lower()
        
        # Always preserve question words
        if word_lower in self.question_words:
            return True
        
        # Preserve sentence starters
        if position == 0 and word_lower in self.sentence_starters:
            return True
        
        # Preserve words that are too short to be meaningful stop words
        if len(word) <= self.min_word_length:
            return word_lower not in self.stop_words
        
        # Regular stop word filtering
        return word_lower not in self.stop_words
    
    def compress(self, text: str) -> str:
        """
        Remove stop words while preserving sentence structure.
        
        Args:
            text: Input text to compress
            
        Returns:
            Text with stop words removed
        """
        if not text.strip():
            return text
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        compressed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if it's a question
            is_question = sentence.endswith('?') or any(
                sentence.lower().startswith(qw) for qw in self.question_words
            )
            
            # Tokenize words (simple approach)
            words = re.findall(r'\b\w+\b', sentence)
            
            if not words:
                compressed_sentences.append(sentence)
                continue
            
            # Filter words
            filtered_words = []
            for i, word in enumerate(words):
                if self._should_preserve_word(word, i, is_question):
                    filtered_words.append(word)
            
            # Preserve at least some words for readability
            if len(filtered_words) < 2 and len(words) >= 2:
                # Keep first and last words at minimum
                filtered_words = [words[0], words[-1]]
            
            if filtered_words:
                compressed_sentence = ' '.join(filtered_words)
                compressed_sentences.append(compressed_sentence)
        
        return '. '.join(compressed_sentences) + ('.' if not text.endswith('.') else '')
    
    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to restore stop words (basic approximation).
        
        Args:
            compressed_text: Compressed text
            
        Returns:
            Text with basic stop words restored
        """
        if not compressed_text.strip():
            return compressed_text
        
        # Basic decompression by adding common connectors
        text = compressed_text
        
        # Add articles before nouns (simple heuristic)
        text = re.sub(r'\b(machine learning|neural network|deep learning|artificial intelligence)\b', 
                     r'the \1', text, flags=re.IGNORECASE)
        
        # Add "to" before infinitive verbs
        text = re.sub(r'\b(implement|create|build|explain|understand|optimize)\b', 
                     r'to \1', text, flags=re.IGNORECASE)
        
        # Add "is" for simple sentences
        text = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]+ing|important|useful|necessary)\b', 
                     r'\1 is \2', text)
        
        return text 