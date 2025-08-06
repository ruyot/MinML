"""
Aggressive shorthand compression technique.
"""

import re
from typing import Dict, Any, Optional

from .base import BaseCompressor


class ShorthandCompressor(BaseCompressor):
    """
    Aggressive compressor that removes spaces and vowels for maximum compression.
    """
    
    # Common abbreviations
    ABBREVIATIONS = {
        'and': '&',
        'the': '',
        'that': 'tht',
        'this': 'ths',
        'with': 'w/',
        'without': 'w/o',
        'you': 'u',
        'your': 'ur',
        'are': 'r',
        'to': '2',
        'too': '2',
        'for': '4',
        'before': 'b4',
        'after': 'aftr',
        'because': 'bc',
        'between': 'btwn',
        'through': 'thru',
        'about': 'abt',
        'around': 'arnd',
        'should': 'shld',
        'would': 'wld',
        'could': 'cld',
        'example': 'eg',
        'examples': 'egs',
        'please': 'pls',
        'thanks': 'thx',
        'information': 'info',
        'implementation': 'impl',
        'function': 'func',
        'variable': 'var',
        'parameter': 'param',
        'parameters': 'params',
        'return': 'ret',
        'returns': 'rets',
        'value': 'val',
        'values': 'vals',
        'object': 'obj',
        'objects': 'objs',
        'method': 'mthd',
        'methods': 'mthds',
        'class': 'cls',
        'classes': 'clss',
        'module': 'mod',
        'modules': 'mods',
        'library': 'lib',
        'libraries': 'libs',
        'framework': 'frmwrk',
        'database': 'db',
        'configuration': 'config',
        'environment': 'env',
        'development': 'dev',
        'production': 'prod',
        'machine learning': 'ML',
        'artificial intelligence': 'AI',
        'neural network': 'NN',
        'deep learning': 'DL',
        'natural language': 'NL',
        'processing': 'proc',
        'application': 'app',
        'programming': 'prog',
        'algorithm': 'algo',
        'algorithms': 'algos'
    }
    
    # Vowels to remove (but keep y)
    VOWELS = 'aeiouAEIOU'
    
    # Characters to preserve
    PRESERVE_CHARS = {' ', '.', '?', '!', ',', ':', ';', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}'}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize shorthand compressor.
        
        Args:
            config: Configuration with compression settings
        """
        super().__init__(config)
        self.remove_spaces = self.config.get('remove_spaces', True)
        self.remove_vowels = self.config.get('remove_vowels', True)
        self.use_abbreviations = self.config.get('use_abbreviations', True)
        self.preserve_first_vowel = self.config.get('preserve_first_vowel', True)
        self.min_word_length = self.config.get('min_word_length', 4)
    
    def _apply_abbreviations(self, text: str) -> str:
        """Apply common abbreviations."""
        if not self.use_abbreviations:
            return text
        
        # Apply abbreviations (case insensitive but preserve case of replacement)
        result = text
        for full, abbrev in self.ABBREVIATIONS.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(full), re.IGNORECASE)
            result = pattern.sub(abbrev, result)
        
        return result
    
    def _remove_vowels(self, word: str) -> str:
        """Remove vowels from a word while preserving readability."""
        if not self.remove_vowels or len(word) < self.min_word_length:
            return word
        
        # Don't remove vowels from very short words
        if len(word) <= 2:
            return word
        
        # Preserve first character if it's a vowel (for readability)
        result = word[0] if self.preserve_first_vowel else ''
        start_idx = 1 if self.preserve_first_vowel else 0
        
        for i in range(start_idx, len(word)):
            char = word[i]
            # Keep consonants and some special cases
            if char not in self.VOWELS or self._should_preserve_vowel(word, i):
                result += char
        
        # Ensure we don't make the word too short to be readable
        if len(result) < 2 and len(word) >= 3:
            # Keep first and last character at minimum
            result = word[0] + word[-1]
        
        return result
    
    def _should_preserve_vowel(self, word: str, position: int) -> bool:
        """Determine if a vowel should be preserved for readability."""
        char = word[position]
        
        # Preserve vowels at the end of words
        if position == len(word) - 1:
            return True
        
        # Preserve vowels followed by certain consonants
        if position < len(word) - 1:
            next_char = word[position + 1]
            if next_char in 'rlnm':  # Liquid consonants
                return True
        
        # Preserve vowels in certain patterns
        if position > 0:
            prev_char = word[position - 1]
            if prev_char in 'qg':  # qu, gu patterns
                return True
        
        return False
    
    def compress(self, text: str) -> str:
        """
        Apply aggressive shorthand compression.
        
        Args:
            text: Input text to compress
            
        Returns:
            Aggressively compressed text
        """
        if not text.strip():
            return text
        
        result = text
        
        # Step 1: Apply abbreviations
        result = self._apply_abbreviations(result)
        
        # Step 2: Process words for vowel removal
        if self.remove_vowels:
            words = result.split()
            compressed_words = []
            
            for word in words:
                # Split on punctuation but preserve it
                parts = re.split(r'([^\w])', word)
                compressed_parts = []
                
                for part in parts:
                    if part.isalpha():
                        compressed_parts.append(self._remove_vowels(part))
                    else:
                        compressed_parts.append(part)
                
                compressed_words.append(''.join(compressed_parts))
            
            result = ' '.join(compressed_words)
        
        # Step 3: Remove extra spaces
        if self.remove_spaces:
            # Remove multiple spaces but preserve single spaces around punctuation
            result = re.sub(r'\s+', ' ', result)
            # Remove spaces around certain punctuation
            result = re.sub(r'\s*([,.!?;:])\s*', r'\1', result)
            # Remove spaces around parentheses and brackets
            result = re.sub(r'\s*([()[\]{}])\s*', r'\1', result)
        
        # Step 4: Final cleanup
        result = result.strip()
        
        return result
    
    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to restore readable text from shorthand.
        
        Args:
            compressed_text: Compressed shorthand text
            
        Returns:
            Expanded text (best effort approximation)
        """
        if not compressed_text.strip():
            return compressed_text
        
        result = compressed_text
        
        # Step 1: Add spaces around punctuation
        result = re.sub(r'([.!?])', r' \1 ', result)
        result = re.sub(r'([,;:])', r'\1 ', result)
        
        # Step 2: Reverse common abbreviations
        reverse_abbrevs = {v: k for k, v in self.ABBREVIATIONS.items() if v}
        for abbrev, full in reverse_abbrevs.items():
            if abbrev in ['&', '2', '4', 'w/', 'w/o']:  # Only reverse obvious ones
                result = result.replace(abbrev, full)
        
        # Step 3: Add spaces between compressed words (simple heuristic)
        # This is very basic and won't be perfect
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)
        
        # Step 4: Try to restore some common vowel patterns
        # This is very limited and heuristic-based
        common_patterns = {
            'tht': 'that',
            'ths': 'this',
            'wth': 'with',
            'frm': 'from',
            'ntrl': 'natural',
            'lrng': 'learning',
            'mchne': 'machine',
            'ntwrk': 'network',
            'prgrmmng': 'programming',
            'lgrtm': 'algorithm'
        }
        
        for compressed, expanded in common_patterns.items():
            result = re.sub(r'\b' + compressed + r'\b', expanded, result, flags=re.IGNORECASE)
        
        # Step 5: Clean up extra spaces
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
        
        return result 