"""
Keyword extraction compression technique.
"""

import re
from typing import Dict, Any, List, Set, Optional, Tuple
from collections import Counter

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from .base import BaseCompressor


class KeywordExtractor(BaseCompressor):
    """
    Compressor that extracts and preserves only the most important keywords.
    This implements "Ultra-Compressed Code" style compression.
    """
    
    # Technical keywords that should be preserved
    TECHNICAL_KEYWORDS = {
        'api', 'algorithm', 'array', 'async', 'await', 'boolean', 'cache', 'class',
        'code', 'compile', 'component', 'config', 'database', 'debug', 'deploy',
        'docker', 'endpoint', 'error', 'exception', 'function', 'git', 'http',
        'json', 'library', 'method', 'model', 'module', 'object', 'package',
        'parameter', 'performance', 'python', 'query', 'response', 'server',
        'service', 'test', 'token', 'type', 'variable', 'version', 'framework',
        'machine learning', 'neural network', 'deep learning', 'transformer',
        'attention', 'embedding', 'tokenizer', 'prompt', 'llm', 'gpt', 'ai',
        'artificial intelligence', 'natural language', 'nlp', 'optimization',
        'gradient', 'training', 'inference', 'fine-tuning', 'compression'
    }
    
    # Action words that indicate intent
    ACTION_WORDS = {
        'explain', 'implement', 'create', 'build', 'write', 'develop', 'design',
        'optimize', 'debug', 'fix', 'solve', 'analyze', 'evaluate', 'compare',
        'review', 'test', 'deploy', 'configure', 'setup', 'install', 'train',
        'learn', 'understand', 'help', 'show', 'demonstrate', 'tutorial',
        'guide', 'handle', 'process', 'manage', 'monitor', 'maintain'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize keyword extractor.
        
        Args:
            config: Configuration with extraction parameters
        """
        super().__init__(config)
        self.max_keywords = self.config.get('max_keywords', 10)
        self.min_word_length = self.config.get('min_word_length', 3)
        self.preserve_actions = self.config.get('preserve_actions', True)
        self.preserve_technical = self.config.get('preserve_technical', True)
        self.preserve_numbers = self.config.get('preserve_numbers', True)
        
        # Combined important keywords
        self.important_keywords = set()
        if self.preserve_technical:
            self.important_keywords.update(self.TECHNICAL_KEYWORDS)
        if self.preserve_actions:
            self.important_keywords.update(self.ACTION_WORDS)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                pass
        
        # Fallback tokenization
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract important phrases (bigrams and trigrams)."""
        words = self._tokenize_text(text)
        phrases = []
        
        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.important_keywords:
                phrases.append(bigram)
        
        # Trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if trigram in self.important_keywords:
                phrases.append(trigram)
        
        return phrases
    
    def _calculate_word_scores(self, text: str) -> Dict[str, float]:
        """Calculate importance scores for words."""
        words = self._tokenize_text(text)
        word_freq = Counter(words)
        scores = {}
        
        for word in words:
            if len(word) < self.min_word_length:
                continue
            
            score = 1.0
            
            # Boost score for technical keywords
            if word in self.important_keywords:
                score *= 3.0
            
            # Boost score for action words
            if word in self.ACTION_WORDS:
                score *= 2.5
            
            # Boost score for capitalized words (likely proper nouns)
            if word[0].isupper():
                score *= 1.5
            
            # Boost score for longer words
            if len(word) > 6:
                score *= 1.3
            
            # Consider frequency (but not too much)
            score *= (1 + word_freq[word] * 0.1)
            
            # Penalize very common words
            if word_freq[word] > len(words) * 0.1:
                score *= 0.5
            
            scores[word] = score
        
        return scores
    
    def _extract_numbers_and_codes(self, text: str) -> List[str]:
        """Extract numbers, versions, and code-like patterns."""
        if not self.preserve_numbers:
            return []
        
        patterns = [
            r'\b\d+\.?\d*\b',  # Numbers
            r'\bv?\d+\.\d+(?:\.\d+)?\b',  # Version numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w*\d+\w*\b',  # Words with numbers
        ]
        
        extracted = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted.extend(matches)
        
        return extracted
    
    def compress(self, text: str) -> str:
        """
        Extract keywords and important phrases from text.
        
        Args:
            text: Input text to compress
            
        Returns:
            Keywords and phrases separated by spaces
        """
        if not text.strip():
            return text
        
        # Extract important phrases first
        phrases = self._extract_phrases(text)
        
        # Extract numbers and codes
        numbers_codes = self._extract_numbers_and_codes(text)
        
        # Calculate word importance scores
        word_scores = self._calculate_word_scores(text)
        
        # Get top keywords by score
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, score in sorted_words[:self.max_keywords]]
        
        # Combine all important elements
        result_parts = phrases + numbers_codes + top_keywords
        
        # Remove duplicates while preserving order
        seen = set()
        final_keywords = []
        for item in result_parts:
            if item.lower() not in seen:
                final_keywords.append(item)
                seen.add(item.lower())
        
        # Limit total output length
        if len(final_keywords) > self.max_keywords:
            final_keywords = final_keywords[:self.max_keywords]
        
        return ' '.join(final_keywords) if final_keywords else text
    
    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to reconstruct readable text from keywords.
        
        Args:
            compressed_text: Space-separated keywords
            
        Returns:
            Reconstructed text (best effort)
        """
        if not compressed_text.strip():
            return compressed_text
        
        keywords = compressed_text.split()
        
        if not keywords:
            return compressed_text
        
        # Try to identify the main action/intent
        action = None
        for word in keywords:
            if word.lower() in self.ACTION_WORDS:
                action = word.lower()
                break
        
        # Try to identify the main topic
        topic_words = [word for word in keywords if word.lower() in self.TECHNICAL_KEYWORDS]
        
        # Basic reconstruction templates
        if action and topic_words:
            if action in ['explain', 'describe']:
                return f"Please {action} {' '.join(topic_words)} and related concepts."
            elif action in ['implement', 'create', 'build']:
                return f"How to {action} {' '.join(topic_words)} system."
            elif action in ['optimize', 'improve']:
                return f"How to {action} {' '.join(topic_words)} performance."
            else:
                return f"{action.capitalize()} {' '.join(topic_words)} effectively."
        elif topic_words:
            return f"Information about {' '.join(topic_words)} concepts."
        elif action:
            return f"How to {action} this effectively."
        else:
            # Fallback: just clean up the keywords
            return ' '.join(keywords) + '.' 