"""
Extensible plugin registry for compression techniques.
"""

import importlib
import inspect
from typing import Dict, Type, List, Optional
from abc import ABC

from .base import BaseCompressor


class CompressorRegistry:
    """Registry for compression plugins with automatic discovery."""
    
    def __init__(self):
        self._compressors: Dict[str, Type[BaseCompressor]] = {}
        self._register_built_in_compressors()
    
    def _register_built_in_compressors(self):
        """Register built-in compressors."""
        from .stop_word_remover import StopWordRemover
        from .keyword_extractor import KeywordExtractor
        from .shorthand_compressor import ShorthandCompressor
        
        self.register("stop_word_remover", StopWordRemover)
        self.register("keyword_extractor", KeywordExtractor)
        self.register("shorthand_compressor", ShorthandCompressor)
    
    def register(self, name: str, compressor_class: Type[BaseCompressor]):
        """Register a compressor class."""
        if not issubclass(compressor_class, BaseCompressor):
            raise ValueError(f"Compressor {compressor_class} must inherit from BaseCompressor")
        
        if not inspect.isclass(compressor_class):
            raise ValueError(f"Expected class, got {type(compressor_class)}")
        
        self._compressors[name] = compressor_class
        print(f"Registered compressor: {name}")
    
    def unregister(self, name: str):
        """Unregister a compressor."""
        if name in self._compressors:
            del self._compressors[name]
            print(f"Unregistered compressor: {name}")
    
    def get_compressor(self, name: str) -> Type[BaseCompressor]:
        """Get a compressor class by name."""
        if name not in self._compressors:
            raise ValueError(f"Unknown compressor: {name}. Available: {list(self._compressors.keys())}")
        return self._compressors[name]
    
    def create_compressor(self, name: str, config: Optional[dict] = None) -> BaseCompressor:
        """Create a compressor instance."""
        compressor_class = self.get_compressor(name)
        return compressor_class(config)
    
    def list_compressors(self) -> List[str]:
        """List all registered compressor names."""
        return list(self._compressors.keys())
    
    def get_compressor_info(self, name: str) -> Dict[str, str]:
        """Get information about a compressor."""
        compressor_class = self.get_compressor(name)
        return {
            "name": name,
            "class": compressor_class.__name__,
            "module": compressor_class.__module__,
            "docstring": compressor_class.__doc__ or "No description available"
        }
    
    def discover_plugins(self, module_path: str):
        """Discover and register plugins from a module path."""
        try:
            module = importlib.import_module(module_path)
            
            # Find all classes that inherit from BaseCompressor
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (inspect.isclass(attr) and 
                    issubclass(attr, BaseCompressor) and 
                    attr != BaseCompressor):
                    
                    # Use class name in snake_case as the plugin name
                    plugin_name = self._camel_to_snake(attr.__name__)
                    self.register(plugin_name, attr)
        
        except ImportError as e:
            print(f"Failed to import plugin module {module_path}: {e}")
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class CompressorPipelineBuilder:
    """Builder for creating compression pipelines with registered compressors."""
    
    def __init__(self, registry: CompressorRegistry):
        self.registry = registry
        self.compressors: List[BaseCompressor] = []
    
    def add_compressor(self, name: str, config: Optional[dict] = None) -> "CompressorPipelineBuilder":
        """Add a compressor to the pipeline."""
        compressor = self.registry.create_compressor(name, config)
        self.compressors.append(compressor)
        return self
    
    def add_compressor_instance(self, compressor: BaseCompressor) -> "CompressorPipelineBuilder":
        """Add a compressor instance to the pipeline."""
        if not isinstance(compressor, BaseCompressor):
            raise ValueError("Compressor must inherit from BaseCompressor")
        self.compressors.append(compressor)
        return self
    
    def build(self):
        """Build the compression pipeline."""
        from .compression_pipeline import CompressionPipeline
        
        # Create a custom pipeline that uses the registered compressors
        class CustomCompressionPipeline(CompressionPipeline):
            def __init__(self, compressors, tokenizer=None):
                # Initialize without calling parent's _setup_compressors
                self.tokenizer = tokenizer
                self.compression_level = None  # Custom pipeline
                self.config = {}
                self.compressors = compressors
                self.stats = {
                    'total_compressions': 0,
                    'total_original_length': 0,
                    'total_compressed_length': 0,
                    'total_original_tokens': 0,
                    'total_compressed_tokens': 0
                }
        
        return CustomCompressionPipeline(self.compressors.copy())
    
    def clear(self) -> "CompressorPipelineBuilder":
        """Clear all compressors from the builder."""
        self.compressors.clear()
        return self


# Global registry instance
compressor_registry = CompressorRegistry()


def register_compressor(name: str):
    """Decorator to register a compressor class."""
    def decorator(cls):
        compressor_registry.register(name, cls)
        return cls
    return decorator


def create_compression_pipeline_from_config(config: dict, tokenizer=None):
    """Create a compression pipeline from configuration."""
    builder = CompressorPipelineBuilder(compressor_registry)
    
    # Support both list of compressors and compression levels
    if "compressors" in config:
        for comp_config in config["compressors"]:
            if isinstance(comp_config, str):
                # Just the name
                builder.add_compressor(comp_config)
            elif isinstance(comp_config, dict):
                # Name with config
                name = comp_config.pop("name")
                builder.add_compressor(name, comp_config)
    elif "compression_level" in config:
        # Use traditional compression levels
        level = config["compression_level"]
        if level >= 1:
            builder.add_compressor("stop_word_remover", config.get("stop_word_remover", {}))
        if level >= 2:
            builder.add_compressor("keyword_extractor", config.get("keyword_extractor", {}))
        if level >= 3:
            builder.add_compressor("shorthand_compressor", config.get("shorthand_compressor", {}))
    
    pipeline = builder.build()
    pipeline.tokenizer = tokenizer
    return pipeline


# Example custom compressor using the decorator
@register_compressor("example_custom")
class ExampleCustomCompressor(BaseCompressor):
    """Example custom compressor for demonstration."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.prefix = self.config.get("prefix", "[CUSTOM]")
    
    def compress(self, text: str) -> str:
        """Add a custom prefix to demonstrate plugin functionality."""
        return f"{self.prefix} {text}"
    
    def decompress(self, compressed_text: str) -> str:
        """Remove the custom prefix."""
        if compressed_text.startswith(self.prefix):
            return compressed_text[len(self.prefix):].strip()
        return compressed_text 