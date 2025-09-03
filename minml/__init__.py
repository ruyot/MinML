"""
Minimal ML compression library with Rust acceleration.
"""

from .rs_bridge import (
    stopword_remove,
    keywords_compress,
    keywords_decompress,
    shorthand
)

__all__ = [
    "stopword_remove",
    "keywords_compress",
    "keywords_decompress",
    "shorthand"
]
