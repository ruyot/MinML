"""
Aggressive shorthand compression technique.
"""

from typing import Dict, Any, Optional
from minml.rs_bridge import shorthand as _sh

from .base import BaseCompressor


class ShorthandCompressor(BaseCompressor):
    """
    Aggressive compressor that removes spaces and vowels for maximum compression.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize shorthand compressor.

        Args:
            config: Configuration with compression settings
        """
        super().__init__(config)
        # Default to level 3 for aggressive compression
        self.level = 3

    def compress(self, text: str) -> str:
        """
        Apply aggressive shorthand compression.

        Args:
            text: Input text to compress

        Returns:
            Aggressively compressed text
        """
        return _sh(text, self.level)

    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to restore readable text from shorthand.

        Args:
            compressed_text: Compressed shorthand text

        Returns:
            Expanded text (best effort approximation)
        """
        # For now, return as-is since decompression is not the focus
        return compressed_text 