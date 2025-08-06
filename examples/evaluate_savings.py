#!/usr/bin/env python3
"""
Evaluate compression savings on a corpus of prompts.
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any
import statistics

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import load_tokenizer, get_default_tokenizer


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a file."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def load_prompts_from_json(file_path: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Try common keys
        for key in ['prompts', 'texts', 'data', 'examples']:
            if key in data:
                return data[key]
    
    raise ValueError("Could not find prompts in JSON file")


def generate_sample_prompts() -> List[str]:
    """Generate sample prompts for evaluation."""
    return [
        "Please explain the concept of machine learning in simple terms for a beginner.",
        "How do I implement a neural network from scratch using Python?",
        "What are the best practices for prompt engineering with large language models?",
        "Can you help me debug this Python function that processes natural language text?",
        "Explain the differences between supervised and unsupervised learning algorithms.",
        "Write a comprehensive guide on setting up a development environment for AI projects.",
        "How can I optimize the performance of my deep learning model training pipeline?",
        "What are the latest trends in artificial intelligence and machine learning research?",
        "Please provide a step-by-step tutorial for building a chatbot using transformers.",
        "Explain the concept of attention mechanisms in transformer architectures.",
        "How do I fine-tune a pre-trained language model for my specific use case?",
        "What are the ethical considerations when deploying AI systems in production?",
        "Can you review this code and suggest improvements for better maintainability?",
        "Explain the trade-offs between different machine learning model architectures.",
        "How do I handle imbalanced datasets in classification problems?",
        "What are the best tools and frameworks for MLOps and model deployment?",
        "Please help me understand the mathematical foundations of gradient descent.",
        "How can I implement effective data preprocessing pipelines for NLP tasks?",
        "Explain the concept of transfer learning and its applications in deep learning.",
        "What are the security considerations for AI applications handling sensitive data?"
    ]


def evaluate_compression_level(prompts: List[str], tokenizer, compression_level: int, 
                             cost_per_token: float) -> Dict[str, Any]:
    """Evaluate compression for a specific level."""
    pipeline = CompressionPipeline(tokenizer=tokenizer, compression_level=compression_level)
    
    results = []
    total_original_tokens = 0
    total_compressed_tokens = 0
    compression_ratios = []
    token_compression_ratios = []
    
    print(f"Evaluating compression level {compression_level}...")
    
    for i, prompt in enumerate(prompts):
        result = pipeline.compress_with_stats(prompt)
        stats = result['stats']
        
        # Calculate cost savings
        original_cost = stats['original_tokens'] * cost_per_token
        compressed_cost = stats['compressed_tokens'] * cost_per_token
        cost_savings = original_cost - compressed_cost
        
        prompt_result = {
            'prompt_index': i,
            'original_length': len(prompt),
            'compressed_length': len(result['compressed']),
            'original_tokens': stats['original_tokens'],
            'compressed_tokens': stats['compressed_tokens'],
            'compression_ratio': stats['compression_ratio'],
            'token_compression_ratio': stats['token_compression_ratio'],
            'space_saved_percent': stats['space_saved_percent'],
            'tokens_saved_percent': stats['tokens_saved_percent'],
            'original_cost': original_cost,
            'compressed_cost': compressed_cost,
            'cost_savings': cost_savings,
            'savings_percent': (cost_savings / original_cost * 100) if original_cost > 0 else 0
        }
        
        results.append(prompt_result)
        total_original_tokens += stats['original_tokens']
        total_compressed_tokens += stats['compressed_tokens']
        compression_ratios.append(stats['compression_ratio'])
        token_compression_ratios.append(stats['token_compression_ratio'])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts...")
    
    # Calculate summary statistics
    total_original_cost = total_original_tokens * cost_per_token
    total_compressed_cost = total_compressed_tokens * cost_per_token
    total_cost_savings = total_original_cost - total_compressed_cost
    
    summary = {
        'compression_level': compression_level,
        'total_prompts': len(prompts),
        'total_original_tokens': total_original_tokens,
        'total_compressed_tokens': total_compressed_tokens,
        'tokens_saved': total_original_tokens - total_compressed_tokens,
        'average_compression_ratio': statistics.mean(compression_ratios),
        'median_compression_ratio': statistics.median(compression_ratios),
        'std_compression_ratio': statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0,
        'average_token_compression_ratio': statistics.mean(token_compression_ratios),
        'median_token_compression_ratio': statistics.median(token_compression_ratios),
        'std_token_compression_ratio': statistics.stdev(token_compression_ratios) if len(token_compression_ratios) > 1 else 0,
        'total_original_cost': total_original_cost,
        'total_compressed_cost': total_compressed_cost,
        'total_cost_savings': total_cost_savings,
        'cost_savings_percent': (total_cost_savings / total_original_cost * 100) if total_original_cost > 0 else 0,
        'best_compression_ratio': min(compression_ratios),
        'worst_compression_ratio': max(compression_ratios),
        'best_token_compression_ratio': min(token_compression_ratios),
        'worst_token_compression_ratio': max(token_compression_ratios)
    }
    
    return {
        'summary': summary,
        'detailed_results': results
    }


def print_summary(evaluation_results: Dict[str, Any]):
    """Print evaluation summary."""
    summary = evaluation_results['summary']
    
    print(f"\n{'='*60}")
    print(f"COMPRESSION EVALUATION SUMMARY - LEVEL {summary['compression_level']}")
    print(f"{'='*60}")
    
    print(f"Dataset: {summary['total_prompts']} prompts")
    print(f"Total original tokens: {summary['total_original_tokens']:,}")
    print(f"Total compressed tokens: {summary['total_compressed_tokens']:,}")
    print(f"Tokens saved: {summary['tokens_saved']:,}")
    print()
    
    print("Character Compression:")
    print(f"  Average ratio: {summary['average_compression_ratio']:.3f}")
    print(f"  Median ratio: {summary['median_compression_ratio']:.3f}")
    print(f"  Std deviation: {summary['std_compression_ratio']:.3f}")
    print(f"  Best compression: {summary['best_compression_ratio']:.3f}")
    print(f"  Worst compression: {summary['worst_compression_ratio']:.3f}")
    print()
    
    print("Token Compression:")
    print(f"  Average ratio: {summary['average_token_compression_ratio']:.3f}")
    print(f"  Median ratio: {summary['median_token_compression_ratio']:.3f}")
    print(f"  Std deviation: {summary['std_token_compression_ratio']:.3f}")
    print(f"  Best compression: {summary['best_token_compression_ratio']:.3f}")
    print(f"  Worst compression: {summary['worst_token_compression_ratio']:.3f}")
    print()
    
    print("Cost Analysis:")
    print(f"  Original cost: ${summary['total_original_cost']:.6f}")
    print(f"  Compressed cost: ${summary['total_compressed_cost']:.6f}")
    print(f"  Total savings: ${summary['total_cost_savings']:.6f}")
    print(f"  Savings percentage: {summary['cost_savings_percent']:.2f}%")
    print()
    
    # Estimate monthly savings
    monthly_factor = 30 * 24  # 30 days, assuming 24 requests per day
    monthly_savings = summary['total_cost_savings'] * monthly_factor
    print(f"Estimated monthly savings (24 requests/day): ${monthly_savings:.2f}")


def print_detailed_analysis(evaluation_results: Dict[str, Any], top_n: int = 5):
    """Print detailed analysis of best and worst performing prompts."""
    results = evaluation_results['detailed_results']
    
    # Sort by token compression ratio
    sorted_by_token_compression = sorted(results, key=lambda x: x['token_compression_ratio'])
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} BEST COMPRESSIONS (by token ratio)")
    print(f"{'='*60}")
    
    for i, result in enumerate(sorted_by_token_compression[:top_n]):
        print(f"{i+1}. Prompt {result['prompt_index']}:")
        print(f"   Token compression: {result['token_compression_ratio']:.3f} ({result['tokens_saved_percent']:.1f}% saved)")
        print(f"   Original tokens: {result['original_tokens']}, Compressed: {result['compressed_tokens']}")
        print(f"   Cost savings: ${result['cost_savings']:.6f} ({result['savings_percent']:.1f}%)")
        print()
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} WORST COMPRESSIONS (by token ratio)")
    print(f"{'='*60}")
    
    for i, result in enumerate(sorted_by_token_compression[-top_n:]):
        print(f"{i+1}. Prompt {result['prompt_index']}:")
        print(f"   Token compression: {result['token_compression_ratio']:.3f} ({result['tokens_saved_percent']:.1f}% saved)")
        print(f"   Original tokens: {result['original_tokens']}, Compressed: {result['compressed_tokens']}")
        print(f"   Cost savings: ${result['cost_savings']:.6f} ({result['savings_percent']:.1f}%)")
        print()


def save_results(evaluation_results: Dict[str, Any], output_file: str):
    """Save evaluation results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


def compare_compression_levels(prompts: List[str], tokenizer, cost_per_token: float):
    """Compare all compression levels."""
    print(f"\n{'='*60}")
    print("COMPARING ALL COMPRESSION LEVELS")
    print(f"{'='*60}")
    
    level_results = {}
    
    for level in [1, 2, 3]:
        results = evaluate_compression_level(prompts, tokenizer, level, cost_per_token)
        level_results[level] = results
    
    # Print comparison table
    print(f"\n{'Metric':<30} {'Level 1':<12} {'Level 2':<12} {'Level 3':<12}")
    print("-" * 66)
    
    metrics = [
        ('Average Token Compression', 'average_token_compression_ratio'),
        ('Median Token Compression', 'median_token_compression_ratio'),
        ('Total Tokens Saved', 'tokens_saved'),
        ('Cost Savings %', 'cost_savings_percent'),
        ('Total Cost Savings', 'total_cost_savings')
    ]
    
    for metric_name, metric_key in metrics:
        values = [level_results[level]['summary'][metric_key] for level in [1, 2, 3]]
        
        if metric_key == 'total_cost_savings':
            row = f"{metric_name:<30} ${values[0]:<11.6f} ${values[1]:<11.6f} ${values[2]:<11.6f}"
        elif metric_key in ['cost_savings_percent']:
            row = f"{metric_name:<30} {values[0]:<11.2f}% {values[1]:<11.2f}% {values[2]:<11.2f}%"
        elif metric_key == 'tokens_saved':
            row = f"{metric_name:<30} {values[0]:<12,} {values[1]:<12,} {values[2]:<12,}"
        else:
            row = f"{metric_name:<30} {values[0]:<12.3f} {values[1]:<12.3f} {values[2]:<12.3f}"
        
        print(row)
    
    return level_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate compression savings")
    parser.add_argument("--input", help="Input file with prompts (txt or json)")
    parser.add_argument("--tokenizer", help="Path to custom tokenizer")
    parser.add_argument("--compression-level", type=int, choices=[1, 2, 3], 
                        help="Compression level to evaluate (if not specified, all levels)")
    parser.add_argument("--cost-per-token", type=float, default=0.00002,
                        help="Cost per token for savings calculation (default: 0.00002 for GPT-3.5)")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of best/worst examples to show")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed analysis of individual prompts")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return 1
        
        if args.input.endswith('.json'):
            prompts = load_prompts_from_json(args.input)
        else:
            prompts = load_prompts_from_file(args.input)
        
        print(f"Loaded {len(prompts)} prompts from {args.input}")
    else:
        prompts = generate_sample_prompts()
        print(f"Using {len(prompts)} sample prompts")
    
    # Load tokenizer
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"Loading custom tokenizer from {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        print("Using default GPT-2 tokenizer")
        tokenizer = get_default_tokenizer()
    
    start_time = time.time()
    
    if args.compression_level:
        # Evaluate single compression level
        results = evaluate_compression_level(prompts, tokenizer, args.compression_level, args.cost_per_token)
        print_summary(results)
        
        if args.detailed:
            print_detailed_analysis(results, args.top_n)
        
        if args.output:
            save_results(results, args.output)
    
    else:
        # Compare all compression levels
        level_results = compare_compression_levels(prompts, tokenizer, args.cost_per_token)
        
        # Show detailed results for level 2 (default)
        if args.detailed:
            print_detailed_analysis(level_results[2], args.top_n)
        
        if args.output:
            save_results(level_results, args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    exit(main()) 