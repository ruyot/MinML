#!/usr/bin/env python3
"""
Test script for Level 3 compression analysis with GPT-OSS.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.gpt_oss_integration import GPTOSSIntegration, GPTOSSEnhancedCompressor
from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import get_default_tokenizer


async def test_level3_compression_analysis():
    """Test Level 3 compression with GPT-OSS analysis."""
    print("=" * 60)
    print("LEVEL 3 COMPRESSION ANALYSIS WITH GPT-OSS")
    print("=" * 60)
    
    # Initialize GPT-OSS integration with local model
    gpt_oss = GPTOSSIntegration(
        local_model_path="models/gpt_oss/llama-2-7b",  # Update this path to your model
        enable_analysis=True,
        enable_optimization=True
    )
    
    print(f"GPT-OSS enabled: {gpt_oss.enable_analysis}")
    print(f"Local model path: {gpt_oss.local_model_path}")
    
    # Test texts for different scenarios
    test_cases = [
        {
            "name": "Educational Content",
            "text": "Please explain the concept of machine learning algorithms in detail for a beginner who has no prior experience with artificial intelligence or computer science.",
            "context": "Educational content for beginners"
        },
        {
            "name": "Technical Documentation",
            "text": "The API endpoint requires authentication via JWT tokens, with rate limiting of 100 requests per minute and support for both GET and POST methods.",
            "context": "Technical API documentation"
        },
        {
            "name": "Creative Writing",
            "text": "The ancient forest whispered secrets through rustling leaves as sunlight filtered through the canopy, creating dappled patterns on the moss-covered ground.",
            "context": "Creative descriptive writing"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*50}")
        
        text = test_case['text']
        context = test_case['context']
        
        print(f"Original text ({len(text)} chars):")
        print(f'"{text}"')
        
        # Test 1: Analysis WITHOUT compression
        print(f"\n--- ANALYSIS WITHOUT COMPRESSION ---")
        no_compression_analysis = await gpt_oss.analyze_compression_quality(
            text, text, context=f"{context} - no compression"
        )
        
        print(f"Semantic similarity: {no_compression_analysis.semantic_similarity:.3f}")
        print(f"Quality score: {no_compression_analysis.compression_quality_score:.3f}")
        print(f"Key concepts preserved: {no_compression_analysis.key_concepts_preserved}")
        print(f"Recommendations: {no_compression_analysis.recommendations}")
        
        # Test 2: Analysis WITH Level 3 compression
        print(f"\n--- ANALYSIS WITH LEVEL 3 COMPRESSION ---")
        
        # Initialize Level 3 compression pipeline
        tokenizer = get_default_tokenizer()
        pipeline = CompressionPipeline(tokenizer, compression_level=3)
        
        # Compress with Level 3
        compressed = pipeline.compress(text)
        print(f"Compressed text ({len(compressed)} chars):")
        print(f'"{compressed}"')
        
        # Get compression stats
        compression_result = pipeline.compress_with_stats(text)
        stats = compression_result['stats']
        
        print(f"\nCompression Statistics:")
        print(f"  Character compression: {stats['compression_ratio']:.3f} ({stats['space_saved_percent']:.1f}% saved)")
        print(f"  Token compression: {stats['token_compression_ratio']:.3f} ({stats['tokens_saved_percent']:.1f}% saved)")
        print(f"  Original tokens: {stats['original_tokens']}")
        print(f"  Compressed tokens: {stats['compressed_tokens']}")
        print(f"  Tokens saved: {stats['tokens_saved']}")
        
        # Analyze compression quality with GPT-OSS
        compression_analysis = await gpt_oss.analyze_compression_quality(
            text, compressed, context=f"{context} - Level 3 compression"
        )
        
        print(f"\nGPT-OSS Analysis Results:")
        print(f"  Semantic similarity: {compression_analysis.semantic_similarity:.3f}")
        print(f"  Quality score: {compression_analysis.compression_quality_score:.3f}")
        print(f"  Key concepts preserved: {compression_analysis.key_concepts_preserved}")
        print(f"  Lost concepts: {compression_analysis.lost_concepts}")
        print(f"  Recommendations: {compression_analysis.recommendations}")
        
        # Test 3: Compression comparison
        print(f"\n--- COMPRESSION COMPARISON ---")
        comparison = await gpt_oss.compare_compression_vs_uncompressed(text, context)
        
        print(f"Recommendation: {comparison.recommendation}")
        print(f"Compression ratio: {comparison.compression_stats.get('compression_ratio', 'N/A'):.3f}")
        print(f"Words removed: {comparison.compression_stats.get('words_removed', 'N/A')}")
        
        # Test 4: Strategy optimization
        print(f"\n--- STRATEGY OPTIMIZATION ---")
        strategy = await gpt_oss.optimize_compression_strategy(
            text, target_compression_ratio=0.6, context=context
        )
        
        print(f"Recommended compression level: {strategy.get('compression_level', 'N/A')}")
        print(f"Terms to preserve: {strategy.get('preserve_terms', [])}")
        print(f"Terms to compress: {strategy.get('compress_terms', [])}")
        print(f"Reasoning: {strategy.get('reasoning', 'N/A')}")
    
    # Show AI dictionary stats
    print(f"\n{'='*60}")
    print("AI DICTIONARY STATISTICS")
    print(f"{'='*60}")
    
    stats = gpt_oss.get_ai_dictionary_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    if stats['most_used_terms']:
        print("Most used terms:")
        for term in stats['most_used_terms']:
            print(f"  {term.term} -> {term.compressed_form} (confidence: {term.confidence:.3f})")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


async def main():
    """Main test function."""
    print("Starting Level 3 Compression Analysis...")
    
    try:
        await test_level3_compression_analysis()
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure you have downloaded a local model to models/gpt_oss/")
        print("Example: python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); tokenizer.save_pretrained('models/gpt_oss/tinyllama'); model.save_pretrained('models/gpt_oss/tinyllama')\"")


if __name__ == "__main__":
    asyncio.run(main())
