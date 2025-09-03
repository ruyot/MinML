#!/usr/bin/env python3
"""
Test script for GPT-OSS integration with compression system.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.gpt_oss_integration import GPTOSSIntegration, GPTOSSEnhancedCompressor
from compression.stop_word_remover import StopWordRemover
from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import get_default_tokenizer


async def test_gpt_oss_integration():
    """Test the GPT-OSS integration functionality."""
    print("=" * 60)
    print("GPT-OSS INTEGRATION TESTING")
    print("=" * 60)
    
    # Initialize GPT-OSS integration with local model
    gpt_oss = GPTOSSIntegration(
        local_model_path="models/gpt-oss/gpt-oss-20b",  # Updated to use GPT-OSS 20B
        enable_analysis=True,
        enable_optimization=True
    )
    
    print(f"GPT-OSS enabled: {gpt_oss.enable_analysis}")
    print(f"Local model path: {gpt_oss.local_model_path}")
    print(f"Optimization enabled: {gpt_oss.enable_optimization}")
    
    # Test text
    test_text = "Please explain the concept of machine learning algorithms in detail for a beginner."
    
    print(f"\nTest text: {test_text}")
    print(f"Length: {len(test_text)} characters")
    
    # Test semantic analysis
    print("\n" + "=" * 40)
    print("SEMANTIC ANALYSIS TEST")
    print("=" * 40)
    
    # Create a simple compressed version
    base_compressor = StopWordRemover()
    compressed = base_compressor.compress(test_text)
    
    print(f"Compressed text: {compressed}")
    print(f"Compressed length: {len(compressed)} characters")
    
    # Analyze with GPT-OSS
    analysis = await gpt_oss.analyze_compression_quality(
        test_text, 
        compressed,
        context="Educational content for beginners"
    )
    
    print(f"\nGPT-OSS Analysis Results:")
    print(f"  Semantic similarity: {analysis.semantic_similarity:.3f}")
    print(f"  Quality score: {analysis.compression_quality_score:.3f}")
    print(f"  Key concepts preserved: {analysis.key_concepts_preserved}")
    print(f"  Lost concepts: {analysis.lost_concepts}")
    print(f"  Recommendations: {analysis.recommendations}")
    
    # Test strategy optimization
    print("\n" + "=" * 40)
    print("STRATEGY OPTIMIZATION TEST")
    print("=" * 40)
    
    strategy = await gpt_oss.optimize_compression_strategy(
        test_text,
        target_compression_ratio=0.7,
        context="Educational content for beginners"
    )
    
    print(f"GPT-OSS Strategy Recommendations:")
    print(f"  Compression level: {strategy.get('compression_level', 'N/A')}")
    print(f"  Preserve terms: {strategy.get('preserve_terms', [])}")
    print(f"  Compress terms: {strategy.get('compress_terms', [])}")
    print(f"  Reasoning: {strategy.get('reasoning', 'N/A')}")
    
    # Test AI dictionary generation
    print("\n" + "=" * 40)
    print("AI DICTIONARY GENERATION TEST")
    print("=" * 40)
    
    entry = await gpt_oss.generate_ai_dictionary_entry(
        term="machine learning",
        context="AI and data science education",
        compressed_form="ML"
    )
    
    print(f"AI Dictionary Entry:")
    print(f"  Term: {entry.term}")
    print(f"  Compressed form: {entry.compressed_form}")
    print(f"  Confidence: {entry.confidence:.3f}")
    print(f"  Context: {entry.context}")
    
    # Test enhanced compressor
    print("\n" + "=" * 40)
    print("ENHANCED COMPRESSOR TEST")
    print("=" * 40)
    
    enhanced_compressor = GPTOSSEnhancedCompressor(
        base_compressor=base_compressor,
        gpt_oss=gpt_oss,
        config={'quality_threshold': 0.8}
    )
    
    result = await enhanced_compressor.compress_with_analysis(
        test_text,
        context="Educational content for beginners"
    )
    
    print(f"Enhanced Compression Result:")
    print(f"  Original: {result['original']}")
    print(f"  Compressed: {result['compressed']}")
    print(f"  Quality score: {result['analysis'].compression_quality_score:.3f}")
    print(f"  Strategy level: {result['strategy'].get('compression_level', 'N/A')}")
    
    # Show AI dictionary stats
    print("\n" + "=" * 40)
    print("AI DICTIONARY STATISTICS")
    print("=" * 40)
    
    stats = gpt_oss.get_ai_dictionary_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    if stats['most_used_terms']:
        print("Most used terms:")
        for term in stats['most_used_terms']:
            print(f"  {term.term} -> {term.compressed_form} (confidence: {term.confidence:.3f})")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


async def test_without_gpt_oss():
    """Test the system without GPT-OSS (fallback mode)."""
    print("\n" + "=" * 60)
    print("FALLBACK MODE TESTING (No GPT-OSS)")
    print("=" * 60)
    
    # Initialize without GPT-OSS
    gpt_oss = GPTOSSIntegration(
        enable_analysis=False,
        enable_optimization=False
    )
    
    print(f"GPT-OSS enabled: {gpt_oss.enable_analysis}")
    print(f"Optimization enabled: {gpt_oss.enable_optimization}")
    
    test_text = "Please explain the concept of machine learning algorithms in detail for a beginner."
    base_compressor = StopWordRemover()
    compressed = base_compressor.compress(test_text)
    
    # Test fallback analysis
    analysis = await gpt_oss.analyze_compression_quality(
        test_text, 
        compressed,
        context="Educational content for beginners"
    )
    
    print(f"\nFallback Analysis Results:")
    print(f"  Semantic similarity: {analysis.semantic_similarity:.3f}")
    print(f"  Quality score: {analysis.compression_quality_score:.3f}")
    print(f"  Key concepts preserved: {analysis.key_concepts_preserved}")
    print(f"  Lost concepts: {analysis.lost_concepts}")
    print(f"  Recommendations: {analysis.recommendations}")
    
    # Test fallback strategy
    strategy = await gpt_oss.optimize_compression_strategy(
        test_text,
        target_compression_ratio=0.7
    )
    
    print(f"\nFallback Strategy:")
    print(f"  Compression level: {strategy.get('compression_level', 'N/A')}")
    print(f"  Reasoning: {strategy.get('reasoning', 'N/A')}")


async def main():
    """Main test function."""
    print("Starting GPT-OSS Integration Tests...")
    
    # Test with GPT-OSS if available
    await test_gpt_oss_integration()
    
    # Test fallback mode
    await test_without_gpt_oss()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
