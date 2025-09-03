"""
Keyword extraction compression technique.
"""

from typing import Dict, Any, Optional
from minml.rs_bridge import keywords_compress, keywords_decompress

from .base import BaseCompressor


class KeywordExtractor(BaseCompressor):
    """
    Compressor that extracts and preserves only the most important keywords.
    This implements "Ultra-Compressed Code" style compression.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize keyword extractor.

        Args:
            config: Configuration with extraction parameters
        """
        super().__init__(config)
        self.max_keywords = self.config.get('max_keywords', 3)

    def compress(self, text: str) -> str:
        """
        Extract keywords and important phrases from text.

        Args:
            text: Input text to compress

        Returns:
            Keywords and phrases separated by spaces
        """
        return keywords_compress(text, self.max_keywords)

    def decompress(self, compressed_text: str) -> str:
        """
        Attempt to reconstruct readable text from keywords.

        Args:
            compressed_text: Space-separated keywords

        Returns:
            Reconstructed text (best effort)
        """
        return keywords_decompress(compressed_text) 