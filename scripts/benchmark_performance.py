#!/usr/bin/env python3
"""
Benchmarking script for GTX 1060 performance validation.
Tests various scenarios and measures generation times, memory usage, and throughput.
"""
import asyncio
import aiohttp
import json
import time
import statistics
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """Configuration for benchmark tests."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.timeout = 120  # seconds
        self.warmup_requests = 2
        self.test_scenarios = [
            # (width, height, steps, description)
            (512, 512, 20, "Standard 512x512, 20 steps"),
            (512, 512, 30, "Standard 512x512, 30 steps"),
            (512, 512, 50, "Standard 512x512, 50 steps"),
            (768, 768, 20, "Large 768x768, 20 steps"),
            (768, 768, 30, "Large 768x768, 30 steps"),
            (256, 256, 20, "Small 256x256, 20 steps"),
            (1024, 1024, 20, "Maximum 1024x1024, 20 steps"),
        ]
        self.test_prompts = [
            "A beautiful landscape with mountains and lakes",
            "A futuristic city with flying cars",
            "A portrait of a cat wearing a hat",
            "Abstract art with vibrant colors",
            "A peaceful garden with flowers",
        ]
        self.concurrent_tests = [1, 2, 3]  # Number of concurrent requests to test


class BenchmarkResult:
    """Results from a benchmark test."""
    
    def __init__(self, scenario: str, prompt: str, width: int, height: int, steps: int):
        self.scenario = scenario
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.generation_times: List[float] = []
        self.queue_wait_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.errors: List[str] = []
        self.memory_usage: List[float] = []
        
    def add_result(self, generation_time: float, queue_wait_time: float = 0.0, 
                   success: bool = True, error: str = None, memory_usage: float = 0.0):
        """Add a result to this benchmark."""
        if success:
            self.generation_times.append(generation_time)
            self.queue_wait_times.append(queue_wait_time)
            self.memory_usage.append(memory_usage)
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.errors.append(error)
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of results."""
        if not self.generation_times:
            return {
                "scenario": self.scenario,
                "success_rate": 0.0,
                "error_count": self.error_count,
                "errors": self.errors
            }
        
        return {
            "scenario": self.scenario,
            "prompt": self.prompt,
            "dimensions": f"{self.width}x{self.height}",
            "steps": self.steps,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / (self.success_count + self.error_count)) * 100,
            "generation_time": {
                "mean": statistics.mean(self.generation_times),
                "median": statistics.median(self.generation_times),
                "min": min(self.generation_times),
                "max": max(self.generation_times),
                "stdev": statistics.stdev(self.generation_times) if len(self.generation_times) > 1 else 0.0
            },
            "queue_wait_time": {
                "mean": statistics.mean(self.queue_wait_times) if self.queue_wait_times else 0.0,
                "max": max(self.queue_wait_times) if self.queue_wait_times else 0.0
            },
            "memory_usage": {
                "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0.0,
                "max": max(self.memory_usage) if self.memory_usage else 0.0
            },
            "errors": list(set(self.errors))  # Unique errors
        }


class PerformanceBenchmark:
    """Main benchmark runner."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info("Starting GTX 1060 performance benchmarks")
        
        # Check API availability
        if not await self._check_api_health():
            raise RuntimeError("API is not available or healthy")
        
        # Run warmup
        await self._run_warmup()
        
        # Run single request benchmarks
        await self._run_single_request_benchmarks()
        
        # Run concurrent request benchmarks
        await self._run_concurrent_benchmarks()
        
        # Generate report
        report = self._generate_report()
        
        logger.info("Benchmarks completed")
        return report
        
    async def _check_api_health(self) -> bool:
        """Check if the API is healthy and ready."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.api_base_url}/health", timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        logger.info(f"API health check passed: {health_data}")
                        return True
                    else:
                        logger.error(f"API health check failed with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
            
    async def _run_warmup(self):
        """Run warmup requests to initialize the model."""
        logger.info(f"Running {self.config.warmup_requests} warmup requests")
        
        for i in range(self.config.warmup_requests):
            try:
                await self._make_generation_request(
                    prompt="Warmup request",
                    width=512,
                    height=512,
                    steps=20
                )
                logger.info(f"Warmup request {i+1} completed")
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
                
    async def _run_single_request_benchmarks(self):
        """Run benchmarks for single requests."""
        logger.info("Running single request benchmarks")
        
        for width, height, steps, description in self.config.test_scenarios:
            for prompt in self.config.test_prompts:
                result = BenchmarkResult(description, prompt, width, height, steps)
                
                logger.info(f"Testing: {description} with prompt: '{prompt[:50]}...'")
                
                # Run multiple iterations for statistical significance
                for iteration in range(3):
                    try:
                        start_time = time.time()
                        response_data = await self._make_generation_request(
                            prompt=prompt,
                            width=width,
                            height=height,
                            steps=steps
                        )
                        generation_time = time.time() - start_time
                        
                        # Extract metrics from response
                        metadata = response_data.get("metadata", {})
                        actual_generation_time = metadata.get("generation_time_seconds", generation_time)
                        
                        result.add_result(
                            generation_time=actual_generation_time,
                            success=True
                        )
                        
                        logger.info(f"  Iteration {iteration+1}: {actual_generation_time:.2f}s")
                        
                    except Exception as e:
                        logger.warning(f"  Iteration {iteration+1} failed: {e}")
                        result.add_result(
                            generation_time=0.0,
                            success=False,
                            error=str(e)
                        )
                
                self.results.append(result)
                
                # Small delay between tests
                await asyncio.sleep(2)
                
    async def _run_concurrent_benchmarks(self):
        """Run benchmarks for concurrent requests."""
        logger.info("Running concurrent request benchmarks")
        
        # Test with standard parameters
        width, height, steps = 512, 512, 20
        prompt = "Concurrent test request"
        
        for concurrent_count in self.config.concurrent_tests:
            logger.info(f"Testing {concurrent_count} concurrent requests")
            
            result = BenchmarkResult(
                f"Concurrent {concurrent_count}x requests",
                prompt,
                width,
                height,
                steps
            )
            
            # Run concurrent requests
            tasks = []
            start_time = time.time()
            
            for i in range(concurrent_count):
                task = asyncio.create_task(
                    self._make_generation_request(
                        prompt=f"{prompt} #{i+1}",
                        width=width,
                        height=height,
                        steps=steps
                    )
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results
            for i, task_result in enumerate(results):
                if isinstance(task_result, Exception):
                    logger.warning(f"Concurrent request {i+1} failed: {task_result}")
                    result.add_result(
                        generation_time=0.0,
                        success=False,
                        error=str(task_result)
                    )
                else:
                    metadata = task_result.get("metadata", {})
                    generation_time = metadata.get("generation_time_seconds", 0.0)
                    result.add_result(
                        generation_time=generation_time,
                        success=True
                    )
            
            logger.info(f"Concurrent test completed in {total_time:.2f}s")
            self.results.append(result)
            
            # Longer delay between concurrent tests
            await asyncio.sleep(5)
            
    async def _make_generation_request(self, prompt: str, width: int, height: int, steps: int) -> Dict[str, Any]:
        """Make a single generation request to the API."""
        request_data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.api_base_url}/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"API request failed with status {response.status}: {error_text}")
                    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "api_base_url": self.config.api_base_url,
                "total_tests": len(self.results),
                "test_scenarios": len(self.config.test_scenarios),
                "test_prompts": len(self.config.test_prompts)
            },
            "summary": self._generate_summary(),
            "detailed_results": [result.get_statistics() for result in self.results],
            "recommendations": self._generate_recommendations()
        }
        
        return report
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all tests."""
        all_generation_times = []
        total_success = 0
        total_tests = 0
        error_types = {}
        
        for result in self.results:
            all_generation_times.extend(result.generation_times)
            total_success += result.success_count
            total_tests += result.success_count + result.error_count
            
            for error in result.errors:
                error_types[error] = error_types.get(error, 0) + 1
        
        if not all_generation_times:
            return {"error": "No successful generations to analyze"}
        
        return {
            "overall_success_rate": (total_success / total_tests) * 100 if total_tests > 0 else 0.0,
            "total_successful_generations": total_success,
            "total_failed_generations": total_tests - total_success,
            "generation_time_stats": {
                "mean": statistics.mean(all_generation_times),
                "median": statistics.median(all_generation_times),
                "min": min(all_generation_times),
                "max": max(all_generation_times),
                "p95": sorted(all_generation_times)[int(len(all_generation_times) * 0.95)] if len(all_generation_times) > 20 else max(all_generation_times)
            },
            "most_common_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        # Analyze generation times
        all_times = []
        for result in self.results:
            all_times.extend(result.generation_times)
        
        if all_times:
            avg_time = statistics.mean(all_times)
            max_time = max(all_times)
            
            if avg_time > 60:
                recommendations.append("Average generation time is high (>60s). Consider reducing default steps or image size.")
            
            if max_time > 120:
                recommendations.append("Some generations are very slow (>120s). Check for memory constraints or increase timeout.")
        
        # Analyze error rates
        total_errors = sum(result.error_count for result in self.results)
        total_tests = sum(result.success_count + result.error_count for result in self.results)
        
        if total_tests > 0:
            error_rate = (total_errors / total_tests) * 100
            if error_rate > 10:
                recommendations.append(f"High error rate ({error_rate:.1f}%). Check system resources and model initialization.")
        
        # Analyze concurrent performance
        concurrent_results = [r for r in self.results if "Concurrent" in r.scenario]
        if concurrent_results:
            for result in concurrent_results:
                if result.error_count > 0:
                    recommendations.append(f"Concurrent requests failing. Consider implementing request queuing or reducing concurrency.")
        
        # GTX 1060 specific recommendations
        recommendations.extend([
            "For GTX 1060 6GB: Keep image sizes ≤768x768 for optimal performance",
            "Use 20-30 steps for best quality/speed balance",
            "Enable attention slicing and CPU offloading for memory efficiency",
            "Consider request queuing for concurrent requests to prevent OOM errors"
        ])
        
        return recommendations
        
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save benchmark report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Benchmark report saved to {filepath}")


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="GTX 1060 Performance Benchmark")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="benchmark_report.json", help="Output report file")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup requests")
    
    args = parser.parse_args()
    
    # Configure benchmark
    config = BenchmarkConfig()
    config.api_base_url = args.api_url
    config.timeout = args.timeout
    config.warmup_requests = args.warmup
    
    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    
    try:
        report = await benchmark.run_all_benchmarks()
        
        # Save report
        benchmark.save_report(report, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        summary = report["summary"]
        if "error" not in summary:
            print(f"Success Rate: {summary['overall_success_rate']:.1f}%")
            print(f"Successful Generations: {summary['total_successful_generations']}")
            print(f"Failed Generations: {summary['total_failed_generations']}")
            print(f"Average Generation Time: {summary['generation_time_stats']['mean']:.2f}s")
            print(f"Median Generation Time: {summary['generation_time_stats']['median']:.2f}s")
            print(f"95th Percentile: {summary['generation_time_stats']['p95']:.2f}s")
            
            if summary['most_common_errors']:
                print("\nMost Common Errors:")
                for error, count in summary['most_common_errors']:
                    print(f"  - {error}: {count} occurrences")
        else:
            print(f"Error: {summary['error']}")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
        
        print(f"\nFull report saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))