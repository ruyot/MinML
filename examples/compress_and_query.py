#!/usr/bin/env python3
"""
End-to-end example of compression and LLM querying.
"""

import os
import sys
import argparse
import asyncio
from typing import Dict, Any

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import load_tokenizer, get_default_tokenizer


def load_compression_pipeline(tokenizer_path: str = None, compression_level: int = 2) -> CompressionPipeline:
    """Load compression pipeline with tokenizer."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"Loading custom tokenizer from {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        print("Using default GPT-2 tokenizer")
        tokenizer = get_default_tokenizer()
    
    return CompressionPipeline(tokenizer=tokenizer, compression_level=compression_level)


def demonstrate_compression(pipeline: CompressionPipeline, text: str):
    """Demonstrate compression with detailed statistics."""
    print(f"\n{'='*60}")
    print("COMPRESSION DEMONSTRATION")
    print(f"{'='*60}")
    
    print(f"Original text ({len(text)} chars):")
    print(f'"{text}"')
    print()
    
    # Compress with statistics
    result = pipeline.compress_with_stats(text)
    
    print(f"Compressed text ({len(result['compressed'])} chars):")
    print(f'"{result["compressed"]}"')
    print()
    
    # Display statistics
    stats = result['stats']
    print("Compression Statistics:")
    print(f"  Character compression: {stats['compression_ratio']:.3f} ({stats['space_saved_percent']:.1f}% saved)")
    print(f"  Token compression: {stats['token_compression_ratio']:.3f} ({stats['tokens_saved_percent']:.1f}% saved)")
    print(f"  Original tokens: {stats['original_tokens']}")
    print(f"  Compressed tokens: {stats['compressed_tokens']}")
    print(f"  Tokens saved: {stats['tokens_saved']}")
    
    # Show step-by-step compression
    print("\nStep-by-step compression:")
    for i, step in enumerate(stats['steps'], 1):
        if 'name' in step:
            print(f"  Step {i} ({step['name']}):")
            if 'input' in step and 'output' in step:
                input_len = len(step['input'])
                output_len = len(step['output'])
                reduction = ((input_len - output_len) / input_len * 100) if input_len > 0 else 0
                print(f"    {reduction:.1f}% character reduction")
        elif 'final' in step:
            print(f"  Final result: {step['final']}")
        else:
            print(f"  Step {i}: {step}")
    
    # Attempt decompression
    decompressed = pipeline.decompress(result['compressed'])
    print(f"\nDecompressed text (approximation):")
    print(f'"{decompressed}"')
    
    return result


async def query_openai_api(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 150) -> Dict[str, Any]:
    """Query OpenAI API with the prompt."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found. Using mock response.")
        return {
            "response": f"Mock response to: {prompt[:50]}...",
            "model": "mock",
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 25, "total_tokens": len(prompt.split()) + 25}
        }
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return {
            "response": f"Error: {str(e)}",
            "model": "error",
            "usage": {}
        }


async def query_local_model(prompt: str, model_url: str = "http://localhost:8001") -> Dict[str, Any]:
    """Query local model server."""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{model_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.7
                },
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "response": result.get("generated_text", ""),
                        "model": result.get("model", "local"),
                        "usage": result.get("usage", {})
                    }
                else:
                    return {
                        "response": f"HTTP {response.status}: {await response.text()}",
                        "model": "error",
                        "usage": {}
                    }
    
    except Exception as e:
        print(f"Error querying local model: {e}")
        return {
            "response": f"Local model unavailable: {str(e)}",
            "model": "error", 
            "usage": {}
        }


async def demonstrate_llm_querying(pipeline: CompressionPipeline, prompt: str, use_compression: bool = True, model: str = "gpt-3.5-turbo"):
    """Demonstrate LLM querying with and without compression."""
    print(f"\n{'='*60}")
    print("LLM QUERYING DEMONSTRATION")
    print(f"{'='*60}")
    
    # Prepare prompts
    original_prompt = prompt
    
    if use_compression:
        compression_result = pipeline.compress_with_stats(original_prompt)
        compressed_prompt = compression_result['compressed']
        compression_stats = compression_result['stats']
    else:
        compressed_prompt = original_prompt
        compression_stats = None
    
    # Query with compressed prompt
    print(f"Querying LLM with {'compressed' if use_compression else 'original'} prompt...")
    
    # Try OpenAI first, then local model
    result = await query_openai_api(compressed_prompt, model=model)
    
    if result['model'] == 'mock' or result['model'] == 'error':
        print("Trying local model...")
        local_result = await query_local_model(compressed_prompt)
        if local_result['model'] != 'error':
            result = local_result
    
    print(f"\nOriginal prompt ({len(original_prompt)} chars):")
    print(f'"{original_prompt}"')
    
    if use_compression:
        print(f"\nCompressed prompt ({len(compressed_prompt)} chars):")
        print(f'"{compressed_prompt}"')
        
        if compression_stats:
            print(f"\nCompression saved {compression_stats['tokens_saved']} tokens ({compression_stats['tokens_saved_percent']:.1f}%)")
    
    print(f"\nLLM Response (using {result['model']}):")
    print(f'"{result["response"]}"')
    
    if 'usage' in result and result['usage']:
        print(f"\nToken usage:")
        for key, value in result['usage'].items():
            print(f"  {key}: {value}")
    
    # Calculate cost savings if we have usage info
    if use_compression and compression_stats and 'usage' in result:
        original_tokens = compression_stats['original_tokens']
        compressed_tokens = compression_stats['compressed_tokens']
        tokens_saved = compression_stats['tokens_saved']
        
        # Estimate cost (GPT-3.5-turbo pricing)
        cost_per_token = 0.00002
        cost_savings = tokens_saved * cost_per_token
        
        print(f"\nEstimated cost savings:")
        print(f"  Original cost: ${original_tokens * cost_per_token:.6f}")
        print(f"  Compressed cost: ${compressed_tokens * cost_per_token:.6f}")
        print(f"  Savings: ${cost_savings:.6f} ({compression_stats['tokens_saved_percent']:.1f}%)")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Demonstrate compression and LLM querying")
    parser.add_argument("--tokenizer", help="Path to custom tokenizer")
    parser.add_argument("--compression-level", type=int, default=2, choices=[1, 2, 3],
                        help="Compression level (1-3)")
    parser.add_argument("--no-compression", action="store_true", 
                        help="Skip compression demonstration")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM querying demonstration")
    parser.add_argument("--prompt", help="Custom prompt to use")
    parser.add_argument("--model", help="Model to query (e.g., gpt-5, gpt-4o)", default="gpt-3.5-turbo")
    parser.add_argument("--interactive", action="store_true", help="Prompt for input interactively")
    
    args = parser.parse_args()
    
    # Load pipeline
    print("Loading compression pipeline...")
    pipeline = load_compression_pipeline(args.tokenizer, args.compression_level)
    
    # Sample prompts for demonstration
    sample_prompts = [
        "Please explain the concept of machine learning in simple terms for a beginner who has no prior experience with artificial intelligence or computer science.",
        "How do I implement a neural network from scratch using Python? I need step-by-step instructions including the mathematical foundations, code examples, and best practices for training and optimization.",
        "What are the best practices for prompt engineering with large language models like GPT-4? Please provide specific techniques, examples, and common pitfalls to avoid."
    ]
    
    if args.interactive and not args.prompt:
        try:
            prompt = input("Enter your prompt: ")
        except EOFError:
            prompt = ""
        if not prompt:
            prompt = sample_prompts[0]
    else:
        prompt = args.prompt or sample_prompts[0]
    
    # Compression demonstration
    if not args.no_compression:
        demonstrate_compression(pipeline, prompt)
        
        # Show other compression levels
        for level in [1, 2, 3]:
            if level != args.compression_level:
                print(f"\n{'='*40}")
                print(f"COMPRESSION LEVEL {level}")
                print(f"{'='*40}")
                temp_pipeline = CompressionPipeline(pipeline.tokenizer, compression_level=level)
                result = temp_pipeline.compress_with_stats(prompt)
                print(f"Compressed: {result['compressed']}")
                print(f"Compression: {result['stats']['compression_ratio']:.3f} ({result['stats']['space_saved_percent']:.1f}% saved)")
                print(f"Token compression: {result['stats']['token_compression_ratio']:.3f} ({result['stats']['tokens_saved_percent']:.1f}% saved)")
    
    # LLM querying demonstration
    if not args.no_llm:
        await demonstrate_llm_querying(pipeline, prompt, use_compression=True, model=args.model)
        
        # Compare with uncompressed
        print(f"\n{'='*60}")
        print("COMPARISON: WITHOUT COMPRESSION")
        print(f"{'='*60}")
        await demonstrate_llm_querying(pipeline, prompt, use_compression=False, model=args.model)
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    
    # Show pipeline statistics
    stats = pipeline.get_pipeline_stats()
    if stats['total_compressions'] > 0:
        print("\nPipeline Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main()) 