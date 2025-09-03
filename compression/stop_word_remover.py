"""
Stop word removal compression technique.
"""

from typing import Dict, Any, Optional
from minml.rs_bridge import stopword_remove as _stop

from .base import BaseCompressor


class StopWordRemover(BaseCompressor):
    """
    Compressor that removes stop words while preserving sentence structure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize stop word remover.

        Args:
            config: Configuration with optional settings
        """
        super().__init__(config)

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
        return _stop(text)

    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to restore stop words (basic approximation).

        Args:
            compressed_text: Compressed text

        Returns:
            Text with basic stop words restored
        """
        # For now, return as-is since decompression is handled by KeywordExtractor
        return compressed_text 