"""
Base interface for compression modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseCompressor(ABC):
    """
    Abstract base class for text compression techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the compressor with configuration.
        
        Args:
            config: Configuration dictionary for the compressor
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compress(self, text: str) -> str:
        """
        Compress the input text.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed text
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed_text: str) -> str:
        """
        Decompress the compressed text (best effort approximation).
        
        Args:
            compressed_text: Compressed text to decompress
            
        Returns:
            Decompressed text approximation
        """
        pass
    
    def get_compression_ratio(self, original: str, compressed: str) -> float:
        """
        Calculate the compression ratio.
        
        Args:
            original: Original text
            compressed: Compressed text
            
        Returns:
            Compression ratio (0.0 to 1.0, where 1.0 is no compression)
        """
        if len(original) == 0:
            return 1.0
        return len(compressed) / len(original)
    
    def get_stats(self, original: str, compressed: str) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Args:
            original: Original text
            compressed: Compressed text
            
        Returns:
            Dictionary with compression statistics
        """
        return {
            "compressor": self.name,
            "original_length": len(original),
            "compressed_length": len(compressed),
            "compression_ratio": self.get_compression_ratio(original, compressed),
            "space_saved": len(original) - len(compressed),
            "space_saved_percent": (1 - self.get_compression_ratio(original, compressed)) * 100
        } 