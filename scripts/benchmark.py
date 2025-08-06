#!/usr/bin/env python3
"""
Performance benchmark script for CI and monitoring.
"""

import time
import asyncio
import statistics
import sys
import os
from typing import List, Dict, Any
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.compression_pipeline import CompressionPipeline
from compression.registry import create_compression_pipeline_from_config
from tokenizer.tokenizer_utils import get_default_tokenizer


class PerformanceBenchmark:
    """Performance benchmark suite."""
    
    def __init__(self):
        self.tokenizer = get_default_tokenizer()
        self.test_texts = [
            "Please explain the concept of machine learning in simple terms.",
            "How do I implement a neural network from scratch using Python? I need detailed instructions.",
            "What are the best practices for prompt engineering with large language models like GPT-4?",
            "Can you help me debug this Python function that processes natural language text and returns a summary?",
            "Explain the differences between supervised and unsupervised learning algorithms in machine learning.",
            "Write a comprehensive guide on setting up a development environment for AI projects using Docker.",
            "How can I optimize the performance of my deep learning model training pipeline for faster convergence?",
            "What are the latest trends in artificial intelligence and machine learning research as of 2024?",
            "Please provide a step-by-step tutorial for building a chatbot using transformer models.",
            "Explain the concept of attention mechanisms in transformer architectures and their importance."
        ]
        
        # Performance thresholds (fail if exceeded)
        self.max_compression_time_ms = 200  # Per text
        self.max_memory_usage_mb = 100
        self.min_compression_ratio = 0.7  # At least 30% compression
    
    def benchmark_compression_speed(self, compression_level: int = 2, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark compression speed."""
        print(f"Benchmarking compression speed (level {compression_level})...")
        
        pipeline_config = {"compression_level": compression_level}
        pipeline = create_compression_pipeline_from_config(pipeline_config, self.tokenizer)
        
        times = []
        compression_ratios = []
        
        for i in range(iterations):
            for text in self.test_texts:
                start_time = time.time()
                result = pipeline.compress_with_stats(text)
                end_time = time.time()
                
                compression_time = (end_time - start_time) * 1000  # ms
                times.append(compression_time)
                compression_ratios.append(result['stats']['compression_ratio'])
        
        return {
            "compression_level": compression_level,
            "iterations": iterations,
            "texts_per_iteration": len(self.test_texts),
            "total_compressions": len(times),
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "p95_time_ms": statistics.quantiles(times, n=20)[18],  # 95th percentile
            "max_time_ms": max(times),
            "min_time_ms": min(times),
            "avg_compression_ratio": statistics.mean(compression_ratios),
            "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_memory_usage(self, compression_level: int = 2) -> Dict[str, Any]:
        """Benchmark memory usage."""
        print(f"Benchmarking memory usage (level {compression_level})...")
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            return {"error": "psutil not available for memory benchmarking"}
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        pipeline_config = {"compression_level": compression_level}
        pipeline = create_compression_pipeline_from_config(pipeline_config, self.tokenizer)
        
        # Memory after pipeline creation
        pipeline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process all texts
        for text in self.test_texts * 10:  # 10x for more memory pressure
            pipeline.compress(text)
        
        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "compression_level": compression_level,
            "baseline_memory_mb": baseline_memory,
            "pipeline_memory_mb": pipeline_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": peak_memory - baseline_memory,
            "pipeline_overhead_mb": pipeline_memory - baseline_memory
        }
    
    def benchmark_compression_quality(self) -> Dict[str, Any]:
        """Benchmark compression quality across levels."""
        print("Benchmarking compression quality...")
        
        results = {}
        
        for level in [1, 2, 3]:
            pipeline_config = {"compression_level": level}
            pipeline = create_compression_pipeline_from_config(pipeline_config, self.tokenizer)
            
            compression_ratios = []
            token_compression_ratios = []
            
            for text in self.test_texts:
                result = pipeline.compress_with_stats(text)
                stats = result['stats']
                compression_ratios.append(stats['compression_ratio'])
                token_compression_ratios.append(stats['token_compression_ratio'])
            
            results[f"level_{level}"] = {
                "avg_compression_ratio": statistics.mean(compression_ratios),
                "avg_token_compression_ratio": statistics.mean(token_compression_ratios),
                "min_compression_ratio": min(compression_ratios),
                "max_compression_ratio": max(compression_ratios)
            }
        
        return results
    
    async def benchmark_concurrent_load(self, concurrent_requests: int = 10, requests_per_worker: int = 5) -> Dict[str, Any]:
        """Benchmark concurrent load handling."""
        print(f"Benchmarking concurrent load ({concurrent_requests} workers, {requests_per_worker} requests each)...")
        
        async def worker_task(worker_id: int):
            """Single worker task."""
            pipeline_config = {"compression_level": 2}
            pipeline = create_compression_pipeline_from_config(pipeline_config, self.tokenizer)
            
            times = []
            for i in range(requests_per_worker):
                text = self.test_texts[i % len(self.test_texts)]
                start_time = time.time()
                pipeline.compress(text)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            return {
                "worker_id": worker_id,
                "times": times,
                "avg_time": statistics.mean(times),
                "total_requests": len(times)
            }
        
        # Run concurrent workers
        start_time = time.time()
        tasks = [worker_task(i) for i in range(concurrent_requests)]
        worker_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Aggregate results
        all_times = []
        total_requests = 0
        
        for result in worker_results:
            all_times.extend(result["times"])
            total_requests += result["total_requests"]
        
        return {
            "concurrent_workers": concurrent_requests,
            "requests_per_worker": requests_per_worker,
            "total_requests": total_requests,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time,
            "avg_response_time_ms": statistics.mean(all_times),
            "p95_response_time_ms": statistics.quantiles(all_times, n=20)[18],
            "max_response_time_ms": max(all_times)
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run full benchmark suite."""
        print("Running full performance benchmark suite...")
        print("=" * 60)
        
        results = {
            "timestamp": time.time(),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # Speed benchmarks
        for level in [1, 2, 3]:
            key = f"speed_level_{level}"
            results[key] = self.benchmark_compression_speed(level, iterations=5)
        
        # Memory benchmark
        results["memory"] = self.benchmark_memory_usage()
        
        # Quality benchmark
        results["quality"] = self.benchmark_compression_quality()
        
        # Concurrent load benchmark
        results["concurrent_load"] = asyncio.run(self.benchmark_concurrent_load(5, 3))
        
        return results
    
    def check_performance_thresholds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if performance meets thresholds."""
        issues = []
        
        # Check compression speed
        for level in [1, 2, 3]:
            speed_results = results.get(f"speed_level_{level}", {})
            avg_time = speed_results.get("avg_time_ms", 0)
            
            if avg_time > self.max_compression_time_ms:
                issues.append(f"Level {level} compression too slow: {avg_time:.1f}ms > {self.max_compression_time_ms}ms")
        
        # Check memory usage
        memory_results = results.get("memory", {})
        memory_increase = memory_results.get("memory_increase_mb", 0)
        
        if memory_increase > self.max_memory_usage_mb:
            issues.append(f"Memory usage too high: {memory_increase:.1f}MB > {self.max_memory_usage_mb}MB")
        
        # Check compression quality
        quality_results = results.get("quality", {})
        for level in [1, 2, 3]:
            level_quality = quality_results.get(f"level_{level}", {})
            avg_ratio = level_quality.get("avg_compression_ratio", 1.0)
            
            if avg_ratio > self.min_compression_ratio:
                issues.append(f"Level {level} compression ratio too low: {avg_ratio:.3f} > {self.min_compression_ratio}")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "thresholds": {
                "max_compression_time_ms": self.max_compression_time_ms,
                "max_memory_usage_mb": self.max_memory_usage_mb,
                "min_compression_ratio": self.min_compression_ratio
            }
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        # Speed results
        print("\nCOMPRESSION SPEED:")
        for level in [1, 2, 3]:
            speed = results.get(f"speed_level_{level}", {})
            print(f"  Level {level}: {speed.get('avg_time_ms', 0):.1f}ms avg, {speed.get('p95_time_ms', 0):.1f}ms p95")
        
        # Memory results
        memory = results.get("memory", {})
        print(f"\nMEMORY USAGE:")
        print(f"  Peak: {memory.get('peak_memory_mb', 0):.1f}MB")
        print(f"  Increase: {memory.get('memory_increase_mb', 0):.1f}MB")
        
        # Quality results
        quality = results.get("quality", {})
        print(f"\nCOMPRESSION QUALITY:")
        for level in [1, 2, 3]:
            q = quality.get(f"level_{level}", {})
            print(f"  Level {level}: {q.get('avg_compression_ratio', 1):.3f} char ratio, {q.get('avg_token_compression_ratio', 1):.3f} token ratio")
        
        # Concurrent load results
        concurrent = results.get("concurrent_load", {})
        print(f"\nCONCURRENT LOAD:")
        print(f"  RPS: {concurrent.get('requests_per_second', 0):.1f}")
        print(f"  Avg response: {concurrent.get('avg_response_time_ms', 0):.1f}ms")
        
        # Threshold check
        threshold_check = self.check_performance_thresholds(results)
        print(f"\nPERFORMANCE THRESHOLDS: {'PASSED' if threshold_check['passed'] else 'FAILED'}")
        if threshold_check['issues']:
            for issue in threshold_check['issues']:
                print(f"  ❌ {issue}")
        else:
            print("  ✅ All thresholds passed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--fail-on-threshold", action="store_true", 
                        help="Exit with non-zero code if thresholds not met")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark (fewer iterations)")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    if args.quick:
        # Quick benchmark for CI
        results = {
            "speed_level_2": benchmark.benchmark_compression_speed(2, iterations=2),
            "quality": benchmark.benchmark_compression_quality()
        }
    else:
        # Full benchmark
        results = benchmark.run_full_benchmark()
    
    benchmark.print_results(results)
    
    # Check thresholds
    threshold_check = benchmark.check_performance_thresholds(results)
    results["threshold_check"] = threshold_check
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    # Exit with error code if thresholds failed
    if args.fail_on_threshold and not threshold_check['passed']:
        print("\n❌ Performance thresholds not met!")
        sys.exit(1)
    
    print("\n✅ Benchmark completed successfully")


if __name__ == "__main__":
    main() 