"""
GPT-OSS Integration for Intelligent Compression

This module integrates local GPT-OSS models to provide:
1. Semantic quality assessment of compression
2. Intelligent compression strategy selection
3. Context-aware compression optimization
4. AI dictionary generation and management
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Local model imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

from .base import BaseCompressor


class CompressionStrategy(Enum):
    """Available compression strategies."""
    CONSERVATIVE = "conservative"  # Preserve maximum meaning
    BALANCED = "balanced"         # Balance compression vs quality
    AGGRESSIVE = "aggressive"     # Maximum compression (Level 3)
    SEMANTIC = "semantic"         # GPT-OSS optimized


@dataclass
class SemanticAnalysis:
    """Results of GPT-OSS semantic analysis."""
    original_meaning: str
    compressed_meaning: str
    semantic_similarity: float  # 0.0 to 1.0
    key_concepts_preserved: List[str]
    lost_concepts: List[str]
    compression_quality_score: float  # 0.0 to 1.0
    recommendations: List[str]


@dataclass
class CompressionComparison:
    """Comparison of compressed vs uncompressed text."""
    original_text: str
    compressed_text: str
    compression_stats: Dict[str, Any]
    semantic_analysis: SemanticAnalysis
    recommendation: str


@dataclass
class AIDictionaryEntry:
    """Entry in the AI-generated compression dictionary."""
    term: str
    compressed_form: str
    context: str
    confidence: float
    usage_count: int
    last_updated: str


class GPTOSSIntegration:
    """
    Integration layer for GPT-OSS to enhance compression quality.
    """
    
    def __init__(self, 
                 local_model_path: str = "models/gpt-oss/gpt-oss-20b",
                 enable_analysis: bool = True,
                 enable_optimization: bool = True):
        """
        Initialize GPT-OSS integration with local model.
        
        Args:
            local_model_path: Path to local GPT-OSS model
            enable_analysis: Enable semantic quality analysis
            enable_optimization: Enable compression optimization
        """
        self.local_model_path = local_model_path
        self.enable_analysis = enable_analysis and LOCAL_MODELS_AVAILABLE
        self.enable_optimization = enable_optimization and self.enable_analysis
        
        # Initialize local model
        self.local_model = None
        self.local_tokenizer = None
        if self.enable_analysis:
            self._load_local_model()
        
        # AI Dictionary for storing learned compression patterns
        self.ai_dictionary: Dict[str, AIDictionaryEntry] = {}
        self.dictionary_path = "models/ai_compression_dictionary.json"
        self._load_ai_dictionary()
    
    def _load_local_model(self):
        """Load local GPT-OSS model from disk."""
        try:
            print(f"Loading local GPT-OSS model from {self.local_model_path}")
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ Local GPT-OSS model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            self.local_model = None
            self.local_tokenizer = None
    
    def _load_ai_dictionary(self):
        """Load AI dictionary from file if it exists."""
        try:
            if os.path.exists(self.dictionary_path):
                with open(self.dictionary_path, 'r') as f:
                    data = json.load(f)
                    self.ai_dictionary = {
                        k: AIDictionaryEntry(**v) for k, v in data.items()
                    }
        except Exception as e:
            print(f"Warning: Could not load AI dictionary: {e}")
    
    def _save_ai_dictionary(self):
        """Save AI dictionary to file."""
        try:
            os.makedirs(os.path.dirname(self.dictionary_path), exist_ok=True)
            with open(self.dictionary_path, 'w') as f:
                json.dump({
                    k: {
                        'term': v.term,
                        'compressed_form': v.compressed_form,
                        'context': v.context,
                        'confidence': v.confidence,
                        'usage_count': v.usage_count,
                        'last_updated': v.last_updated
                    } for k, v in self.ai_dictionary.items()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save AI dictionary: {e}")
    
    async def _query_local_model(self, prompt: str, max_tokens: int = 500) -> str:
        """Query local GPT-OSS model with prompt."""
        if not self.local_model or not self.local_tokenizer:
            raise Exception("Local GPT-OSS model not loaded")
        
        try:
            # Format prompt for the model
            if "llama" in self.local_model_path.lower():
                formatted_prompt = f"[INST] {prompt} [/INST]"
            elif "mistral" in self.local_model_path.lower():
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            # Tokenize and generate
            inputs = self.local_tokenizer(formatted_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            response = response.replace(formatted_prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Local model query failed: {e}")
            raise
    
    async def analyze_compression_quality(self, 
                                        original: str, 
                                        compressed: str,
                                        context: Optional[str] = None) -> SemanticAnalysis:
        """
        Use GPT-OSS to analyze compression quality and semantic preservation.
        
        Args:
            original: Original text
            compressed: Compressed text
            context: Optional context about the compression purpose
            
        Returns:
            SemanticAnalysis with quality assessment
        """
        if not self.enable_analysis:
            return self._default_analysis(original, compressed)
        
        try:
            prompt = f"""
            Analyze the semantic quality of text compression.
            
            Original text: "{original}"
            Compressed text: "{compressed}"
            
            Context: {context or "General text compression"}
            
            Please provide:
            1. Brief summary of original meaning
            2. Brief summary of compressed meaning  
            3. Semantic similarity score (0.0 to 1.0)
            4. Key concepts that were preserved
            5. Important concepts that were lost
            6. Overall compression quality score (0.0 to 1.0)
            7. Specific recommendations for improvement
            
            Format as JSON:
            {{
                "original_meaning": "...",
                "compressed_meaning": "...",
                "semantic_similarity": 0.85,
                "key_concepts_preserved": ["concept1", "concept2"],
                "lost_concepts": ["concept3"],
                "compression_quality_score": 0.82,
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            response = await self._query_local_model(prompt, max_tokens=500)
            
            # Parse JSON response
            try:
                analysis_data = json.loads(response)
                return SemanticAnalysis(**analysis_data)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                print(f"Failed to parse JSON response: {response}")
                return self._default_analysis(original, compressed)
            
        except Exception as e:
            print(f"GPT-OSS analysis failed: {e}")
            return self._default_analysis(original, compressed)
    
    def _default_analysis(self, original: str, compressed: str) -> SemanticAnalysis:
        """Fallback analysis when GPT-OSS is unavailable."""
        # Simple heuristic-based analysis
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        preserved = original_words.intersection(compressed_words)
        lost = original_words - compressed_words
        
        similarity = len(preserved) / len(original_words) if original_words else 0.0
        quality_score = min(similarity * 1.2, 1.0)  # Boost for compression
        
        return SemanticAnalysis(
            original_meaning="Original text meaning",
            compressed_meaning="Compressed text meaning", 
            semantic_similarity=similarity,
            key_concepts_preserved=list(preserved),
            lost_concepts=list(lost),
            compression_quality_score=quality_score,
            recommendations=["Enable GPT-OSS for detailed analysis"]
        )
    
    async def compare_compression_vs_uncompressed(self,
                                                text: str,
                                                context: Optional[str] = None) -> CompressionComparison:
        """
        Compare compressed vs uncompressed text using GPT-OSS analysis.
        
        Args:
            text: Original text to analyze
            context: Context for compression
            
        Returns:
            CompressionComparison with analysis
        """
        if not self.enable_analysis:
            return self._default_comparison(text)
        
        try:
            # Create a simple compressed version for comparison
            # This simulates what the compression pipeline would do
            words = text.split()
            compressed_words = [word for word in words if len(word) > 3]  # Simple compression
            compressed_text = " ".join(compressed_words)
            
            # Analyze compression quality
            analysis = await self.analyze_compression_quality(text, compressed_text, context)
            
            # Generate recommendation
            if analysis.compression_quality_score > 0.8:
                recommendation = "Compression recommended - high quality preservation"
            elif analysis.compression_quality_score > 0.6:
                recommendation = "Compression acceptable - moderate quality preservation"
            else:
                recommendation = "Compression not recommended - significant quality loss"
            
            # Basic compression stats
            compression_stats = {
                "original_length": len(text),
                "compressed_length": len(compressed_text),
                "compression_ratio": len(compressed_text) / len(text),
                "words_removed": len(words) - len(compressed_words)
            }
            
            return CompressionComparison(
                original_text=text,
                compressed_text=compressed_text,
                compression_stats=compression_stats,
                semantic_analysis=analysis,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"Compression comparison failed: {e}")
            return self._default_comparison(text)
    
    def _default_comparison(self, text: str) -> CompressionComparison:
        """Default comparison when GPT-OSS is unavailable."""
        return CompressionComparison(
            original_text=text,
            compressed_text=text,
            compression_stats={"error": "GPT-OSS not available"},
            semantic_analysis=self._default_analysis(text, text),
            recommendation="Enable GPT-OSS for detailed analysis"
        )
    
    async def optimize_compression_strategy(self,
                                          text: str,
                                          target_compression_ratio: float = 0.7,
                                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use GPT-OSS to determine optimal compression strategy.
        
        Args:
            text: Text to compress
            target_compression_ratio: Desired compression ratio
            context: Context about compression purpose
            
        Returns:
            Dictionary with recommended compression settings
        """
        if not self.enable_optimization:
            return self._default_strategy()
        
        try:
            prompt = f"""
            Analyze this text and recommend optimal compression strategy.
            
            Text: "{text}"
            Target compression ratio: {target_compression_ratio}
            Context: {context or "General text compression"}
            
            Consider:
            - Text complexity and technical content
            - Importance of preserving specific terms
            - Balance between compression and readability
            
            Recommend:
            1. Compression level (1-3)
            2. Specific compressor settings
            3. Terms to preserve
            4. Terms that can be aggressively compressed
            
            Format as JSON:
            {{
                "compression_level": 2,
                "compressor_settings": {{
                    "stop_word_remover": {{"preserve_structure": true}},
                    "keyword_extractor": {{"max_keywords": 8}},
                    "shorthand_compressor": {{"remove_spaces": false}}
                }},
                "preserve_terms": ["term1", "term2"],
                "compress_terms": ["term3", "term4"],
                "reasoning": "explanation"
            }}
            """
            
            response = await self._query_local_model(prompt, max_tokens=400)
            
            try:
                strategy_data = json.loads(response)
                return strategy_data
            except json.JSONDecodeError:
                print(f"Failed to parse strategy JSON: {response}")
                return self._default_strategy()
            
        except Exception as e:
            print(f"GPT-OSS optimization failed: {e}")
            return self._default_strategy()
    
    def _default_strategy(self) -> Dict[str, Any]:
        """Default compression strategy when GPT-OSS is unavailable."""
        return {
            "compression_level": 3,  # Default to Level 3 for analysis
            "compressor_settings": {
                "stop_word_remover": {"preserve_structure": True},
                "keyword_extractor": {"max_keywords": 8},
                "shorthand_compressor": {"remove_spaces": True}
            },
            "preserve_terms": [],
            "compress_terms": [],
            "reasoning": "Default Level 3 strategy for maximum compression"
        }
    
    async def generate_ai_dictionary_entry(self,
                                         term: str,
                                         context: str,
                                         compressed_form: str) -> AIDictionaryEntry:
        """
        Use GPT-OSS to generate intelligent compression dictionary entries.
        
        Args:
            term: Original term
            context: Context where term is used
            compressed_form: Proposed compressed form
            
        Returns:
            AI-generated dictionary entry
        """
        if not self.enable_optimization:
            return self._create_basic_entry(term, context, compressed_form)
        
        try:
            prompt = f"""
            Analyze this compression mapping and provide feedback.
            
            Original term: "{term}"
            Compressed form: "{compressed_form}"
            Context: {context}
            
            Evaluate:
            1. Is the compression appropriate?
            2. What's the confidence level (0.0 to 1.0)?
            3. Any suggestions for improvement?
            
            Format as JSON:
            {{
                "confidence": 0.85,
                "suggestions": ["suggestion1", "suggestion2"],
                "improved_compression": "better_form"
            }}
            """
            
            response = await self._query_local_model(prompt, max_tokens=200)
            
            try:
                analysis = json.loads(response)
                
                # Use improved compression if suggested
                final_compression = analysis.get("improved_compression", compressed_form)
                confidence = analysis.get("confidence", 0.8)
                
                entry = AIDictionaryEntry(
                    term=term,
                    compressed_form=final_compression,
                    context=context,
                    confidence=confidence,
                    usage_count=1,
                    last_updated="2024-01-01"
                )
                
                # Store in dictionary
                self.ai_dictionary[term] = entry
                self._save_ai_dictionary()
                
                return entry
                
            except json.JSONDecodeError:
                print(f"Failed to parse dictionary JSON: {response}")
                return self._create_basic_entry(term, context, compressed_form)
            
        except Exception as e:
            print(f"GPT-OSS dictionary generation failed: {e}")
            return self._create_basic_entry(term, context, compressed_form)
    
    def _create_basic_entry(self, term: str, context: str, compressed_form: str) -> AIDictionaryEntry:
        """Create basic dictionary entry without GPT-OSS."""
        return AIDictionaryEntry(
            term=term,
            compressed_form=compressed_form,
            context=context,
            confidence=0.7,
            usage_count=1,
            last_updated="2024-01-01"
        )
    
    def get_ai_dictionary_stats(self) -> Dict[str, Any]:
        """Get statistics about the AI dictionary."""
        if not self.ai_dictionary:
            return {"total_entries": 0, "average_confidence": 0.0, "most_used_terms": []}
        
        total_entries = len(self.ai_dictionary)
        avg_confidence = sum(e.confidence for e in self.ai_dictionary.values()) / total_entries
        
        return {
            "total_entries": total_entries,
            "average_confidence": avg_confidence,
            "most_used_terms": sorted(
                self.ai_dictionary.values(),
                key=lambda x: x.usage_count,
                reverse=True
            )[:5]
        }


class GPTOSSEnhancedCompressor(BaseCompressor):
    """
    Enhanced compressor that uses GPT-OSS for intelligent compression.
    """
    
    def __init__(self, 
                 base_compressor: BaseCompressor,
                 gpt_oss: GPTOSSIntegration,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced compressor.
        
        Args:
            base_compressor: Base compression algorithm
            gpt_oss: GPT-OSS integration instance
            config: Configuration options
        """
        super().__init__(config)
        self.base_compressor = base_compressor
        self.gpt_oss = gpt_oss
        self.quality_threshold = config.get('quality_threshold', 0.8)
        self.max_retries = config.get('max_retries', 2)
    
    async def compress_with_analysis(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Compress text with GPT-OSS quality analysis.
        
        Args:
            text: Text to compress
            context: Context for compression
            
        Returns:
            Dictionary with compressed text and analysis
        """
        # Get optimal strategy from GPT-OSS
        strategy = await self.gpt_oss.optimize_compression_strategy(text, context=context)
        
        # Apply base compression
        compressed = self.base_compressor.compress(text)
        
        # Analyze quality
        analysis = await self.gpt_oss.analyze_compression_quality(
            text, compressed, context
        )
        
        # If quality is below threshold, try to improve
        if analysis.compression_quality_score < self.quality_threshold:
            compressed = await self._improve_compression(text, compressed, analysis, strategy)
            # Re-analyze
            analysis = await self.gpt_oss.analyze_compression_quality(
                text, compressed, context
            )
        
        return {
            'original': text,
            'compressed': compressed,
            'analysis': analysis,
            'strategy': strategy
        }
    
    async def _improve_compression(self, 
                                  original: str, 
                                  compressed: str, 
                                  analysis: SemanticAnalysis,
                                  strategy: Dict[str, Any]) -> str:
        """Attempt to improve compression based on GPT-OSS analysis."""
        # This is a simplified improvement - in practice, you'd implement
        # more sophisticated logic based on the analysis
        if analysis.lost_concepts:
            # Try to preserve lost concepts
            preserved_concepts = ' '.join(analysis.key_concepts_preserved)
            if len(preserved_concepts) < len(compressed):
                return preserved_concepts
        
        return compressed
    
    def compress(self, text: str) -> str:
        """Synchronous compression (uses base compressor)."""
        return self.base_compressor.compress(text)
    
    def decompress(self, compressed_text: str) -> str:
        """Synchronous decompression (uses base compressor)."""
        return self.base_compressor.decompress(compressed_text)
