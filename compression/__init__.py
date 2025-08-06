"""
Compression module for semantic text compression techniques.
"""

from .base import BaseCompressor
from .stop_word_remover import StopWordRemover
from .keyword_extractor import KeywordExtractor
from .shorthand_compressor import ShorthandCompressor
from .compression_pipeline import CompressionPipeline
from .registry import (
    compressor_registry, 
    register_compressor,
    CompressorRegistry,
    CompressorPipelineBuilder,
    create_compression_pipeline_from_config
)

__all__ = [
    'BaseCompressor',
    'StopWordRemover', 
    'KeywordExtractor',
    'ShorthandCompressor',
    'CompressionPipeline',
    'compressor_registry',
    'register_compressor',
    'CompressorRegistry',
    'CompressorPipelineBuilder',
    'create_compression_pipeline_from_config'
] 