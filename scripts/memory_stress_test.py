#!/usr/bin/env python3
"""
Memory stress test for GTX 1060 optimization validation.
Tests memory limits, queue behavior, and recovery mechanisms.
"""
import asyncio
import aiohttp
import json
import time
import logging
from typing import List, Dict, Any
from datetime import datetime
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryStressTest:
    """Memory stress testing for GTX 1060 constraints."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all memory stress tests."""
        logger.info("Starting GTX 1060 memory stress tests")
        
        # Check API health
        if not await self._check_api_health():
            raise RuntimeError("API is not available")
        
        test_results = {}
        
        # Test 1: Progressive memory load
        logger.info("Test 1: Progressive memory load")
        test_results["progressive_load"] = await self._test_progressive_memory_load()
        
        # Test 2: Concurrent request handling
        logger.info("Test 2: Concurrent request handling")
        test_results["concurrent_requests"] = await self._test_concurrent_requests()
        
        # Test 3: Large image generation
        logger.info("Test 3: Large image generation")
        test_results["large_images"] = await self._test_large_image_generation()
        
        # Test 4: Queue behavior under load
        logger.info("Test 4: Queue behavior under load")
        test_results["queue_behavior"] = await self._test_queue_behavior()
        
        # Test 5: Memory recovery after errors
        logger.info("Test 5: Memory recovery after errors")
        test_results["memory_recovery"] = await self._test_memory_recovery()
        
        # Generate final report
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "api_base_url": self.api_base_url
            },
            "test_results": test_results,
            "summary": self._generate_summary(test_results),
            "recommendations": self._generate_recommendations(test_results)
        }
        
        return report
        
    async def _check_api_health(self) -> bool:
        """Check API health and get system info."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/health", timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        logger.info(f"API health: {health_data}")
                        return True
                    return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def _test_progressive_memory_load(self) -> Dict[str, Any]:
        """Test progressive memory loading with increasing image sizes."""
        test_sizes = [
            (256, 256, "Small"),
            (512, 512, "Medium"),
            (768, 768, "Large"),
            (1024, 1024, "Extra Large")
        ]
        
        results = []
        
        for width, height, size_name in test_sizes:
            logger.info(f"Testing {size_name} images ({width}x{height})")
            
            test_result = {
                "size": size_name,
                "dimensions": f"{width}x{height}",
                "attempts": [],
                "success_count": 0,
                "error_count": 0
            }
            
            # Try multiple generations at this size
            for attempt in range(3):
                try:
                    start_time = time.time()
                    response = await self._make_request({
                        "prompt": f"Memory test {size_name} image #{attempt+1}",
                        "width": width,
                        "height": height,
                        "steps": 20
                    })
                    
                    generation_time = time.time() - start_time
                    metadata = response.get("metadata", {})
                    
                    attempt_result = {
                        "attempt": attempt + 1,
                        "success": True,
                        "generation_time": generation_time,
                        "reported_time": metadata.get("generation_time_seconds", 0),
                        "error": None
                    }
                    
                    test_result["success_count"] += 1
                    logger.info(f"  Attempt {attempt+1}: Success in {generation_time:.2f}s")
                    
                except Exception as e:
                    attempt_result = {
                        "attempt": attempt + 1,
                        "success": False,
                        "generation_time": 0,
                        "reported_time": 0,
                        "error": str(e)
                    }
                    
                    test_result["error_count"] += 1
                    logger.warning(f"  Attempt {attempt+1}: Failed - {e}")
                
                test_result["attempts"].append(attempt_result)
                
                # Wait between attempts
                await asyncio.sleep(2)
            
            results.append(test_result)
            
            # Longer wait between size changes
            await asyncio.sleep(5)
        
        return {
            "test_name": "Progressive Memory Load",
            "results": results,
            "analysis": self._analyze_progressive_load(results)
        }
        
    async def _test_concurrent_requests(self) -> Dict[str, Any]:
        """Test concurrent request handling and queuing."""
        concurrent_levels = [2, 3, 5, 8]
        results = []
        
        for concurrent_count in concurrent_levels:
            logger.info(f"Testing {concurrent_count} concurrent requests")
            
            # Create concurrent requests
            tasks = []
            start_time = time.time()
            
            for i in range(concurrent_count):
                task = asyncio.create_task(
                    self._make_request({
                        "prompt": f"Concurrent test request #{i+1}",
                        "width": 512,
                        "height": 512,
                        "steps": 20
                    })
                )
                tasks.append(task)
            
            # Wait for all to complete
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful = 0
            failed = 0
            errors = []
            generation_times = []
            
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    failed += 1
                    errors.append(str(result))
                    logger.warning(f"  Request {i+1}: Failed - {result}")
                else:
                    successful += 1
                    metadata = result.get("metadata", {})
                    gen_time = metadata.get("generation_time_seconds", 0)
                    generation_times.append(gen_time)
                    logger.info(f"  Request {i+1}: Success in {gen_time:.2f}s")
            
            test_result = {
                "concurrent_count": concurrent_count,
                "total_time": total_time,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / concurrent_count) * 100,
                "average_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
                "max_generation_time": max(generation_times) if generation_times else 0,
                "errors": list(set(errors))
            }
            
            results.append(test_result)
            
            # Wait between tests
            await asyncio.sleep(10)
        
        return {
            "test_name": "Concurrent Requests",
            "results": results,
            "analysis": self._analyze_concurrent_requests(results)
        }
        
    async def _test_large_image_generation(self) -> Dict[str, Any]:
        """Test generation of large images that push memory limits."""
        large_sizes = [
            (768, 768, 30),
            (1024, 1024, 20),
            (1024, 1024, 30),
        ]
        
        results = []
        
        for width, height, steps in large_sizes:
            logger.info(f"Testing large image: {width}x{height}, {steps} steps")
            
            try:
                start_time = time.time()
                response = await self._make_request({
                    "prompt": f"Large image test {width}x{height}",
                    "width": width,
                    "height": height,
                    "steps": steps
                }, timeout=180)  # Longer timeout for large images
                
                generation_time = time.time() - start_time
                metadata = response.get("metadata", {})
                
                result = {
                    "dimensions": f"{width}x{height}",
                    "steps": steps,
                    "success": True,
                    "generation_time": generation_time,
                    "reported_time": metadata.get("generation_time_seconds", 0),
                    "error": None
                }
                
                logger.info(f"  Success: {generation_time:.2f}s")
                
            except Exception as e:
                result = {
                    "dimensions": f"{width}x{height}",
                    "steps": steps,
                    "success": False,
                    "generation_time": 0,
                    "reported_time": 0,
                    "error": str(e)
                }
                
                logger.warning(f"  Failed: {e}")
            
            results.append(result)
            
            # Wait between large image tests
            await asyncio.sleep(10)
        
        return {
            "test_name": "Large Image Generation",
            "results": results,
            "analysis": self._analyze_large_images(results)
        }
        
    async def _test_queue_behavior(self) -> Dict[str, Any]:
        """Test queue behavior under heavy load."""
        logger.info("Flooding API with requests to test queue behavior")
        
        # Send many requests quickly
        request_count = 10
        tasks = []
        
        start_time = time.time()
        
        for i in range(request_count):
            task = asyncio.create_task(
                self._make_request({
                    "prompt": f"Queue test request #{i+1}",
                    "width": 512,
                    "height": 512,
                    "steps": 20
                }, timeout=300)  # Long timeout for queued requests
            )
            tasks.append(task)
            
            # Small delay to simulate rapid requests
            await asyncio.sleep(0.1)
        
        # Wait for all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze queue behavior
        successful = 0
        failed = 0
        queue_errors = 0
        timeout_errors = 0
        generation_times = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                error_str = str(result)
                if "queue" in error_str.lower():
                    queue_errors += 1
                elif "timeout" in error_str.lower():
                    timeout_errors += 1
                logger.warning(f"Request {i+1}: {error_str}")
            else:
                successful += 1
                metadata = result.get("metadata", {})
                gen_time = metadata.get("generation_time_seconds", 0)
                generation_times.append(gen_time)
        
        return {
            "test_name": "Queue Behavior",
            "request_count": request_count,
            "total_time": total_time,
            "successful": successful,
            "failed": failed,
            "queue_errors": queue_errors,
            "timeout_errors": timeout_errors,
            "success_rate": (successful / request_count) * 100,
            "average_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
            "analysis": {
                "queue_handling": "Good" if queue_errors < request_count * 0.2 else "Poor",
                "timeout_handling": "Good" if timeout_errors < request_count * 0.1 else "Poor",
                "overall_throughput": successful / (total_time / 60)  # requests per minute
            }
        }
        
    async def _test_memory_recovery(self) -> Dict[str, Any]:
        """Test memory recovery after intentional errors."""
        logger.info("Testing memory recovery after errors")
        
        # First, try to cause a memory error with very large image
        try:
            await self._make_request({
                "prompt": "Memory error test",
                "width": 1536,  # Very large, likely to fail
                "height": 1536,
                "steps": 50
            }, timeout=60)
            memory_error_occurred = False
        except Exception as e:
            memory_error_occurred = True
            logger.info(f"Expected memory error occurred: {e}")
        
        # Wait for recovery
        await asyncio.sleep(5)
        
        # Test if system recovered with normal request
        recovery_attempts = []
        
        for attempt in range(3):
            try:
                start_time = time.time()
                response = await self._make_request({
                    "prompt": f"Recovery test #{attempt+1}",
                    "width": 512,
                    "height": 512,
                    "steps": 20
                })
                
                generation_time = time.time() - start_time
                recovery_attempts.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "generation_time": generation_time,
                    "error": None
                })
                
                logger.info(f"Recovery attempt {attempt+1}: Success")
                
            except Exception as e:
                recovery_attempts.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "generation_time": 0,
                    "error": str(e)
                })
                
                logger.warning(f"Recovery attempt {attempt+1}: Failed - {e}")
            
            await asyncio.sleep(2)
        
        successful_recoveries = sum(1 for attempt in recovery_attempts if attempt["success"])
        
        return {
            "test_name": "Memory Recovery",
            "memory_error_triggered": memory_error_occurred,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": (successful_recoveries / len(recovery_attempts)) * 100,
            "analysis": {
                "recovery_capability": "Good" if successful_recoveries >= 2 else "Poor",
                "system_stability": "Stable" if successful_recoveries == 3 else "Unstable"
            }
        }
        
    async def _make_request(self, data: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Make a request to the API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")
                    
    def _analyze_progressive_load(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze progressive load test results."""
        analysis = {
            "memory_limit_reached": False,
            "max_successful_size": None,
            "failure_pattern": []
        }
        
        for result in results:
            if result["success_count"] > 0:
                analysis["max_successful_size"] = result["size"]
            
            if result["error_count"] > 0:
                analysis["failure_pattern"].append({
                    "size": result["size"],
                    "error_rate": (result["error_count"] / (result["success_count"] + result["error_count"])) * 100
                })
                
                if result["error_count"] == len(result["attempts"]):
                    analysis["memory_limit_reached"] = True
        
        return analysis
        
    def _analyze_concurrent_requests(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze concurrent request test results."""
        analysis = {
            "max_concurrent_supported": 0,
            "performance_degradation": [],
            "queue_effectiveness": "Unknown"
        }
        
        for result in results:
            if result["success_rate"] >= 80:  # 80% success rate threshold
                analysis["max_concurrent_supported"] = result["concurrent_count"]
            
            analysis["performance_degradation"].append({
                "concurrent_count": result["concurrent_count"],
                "success_rate": result["success_rate"],
                "avg_time": result["average_generation_time"]
            })
        
        # Determine queue effectiveness
        high_concurrency_results = [r for r in results if r["concurrent_count"] >= 5]
        if high_concurrency_results:
            avg_success_rate = sum(r["success_rate"] for r in high_concurrency_results) / len(high_concurrency_results)
            analysis["queue_effectiveness"] = "Good" if avg_success_rate >= 60 else "Poor"
        
        return analysis
        
    def _analyze_large_images(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze large image test results."""
        successful_large = [r for r in results if r["success"]]
        failed_large = [r for r in results if not r["success"]]
        
        return {
            "large_image_support": len(successful_large) > 0,
            "max_supported_resolution": max([r["dimensions"] for r in successful_large]) if successful_large else None,
            "failure_rate": (len(failed_large) / len(results)) * 100,
            "performance_impact": {
                "avg_time": sum(r["generation_time"] for r in successful_large) / len(successful_large) if successful_large else 0,
                "max_time": max(r["generation_time"] for r in successful_large) if successful_large else 0
            }
        }
        
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary."""
        summary = {
            "overall_stability": "Good",
            "memory_management": "Good",
            "queue_performance": "Good",
            "recovery_capability": "Good",
            "issues_found": []
        }
        
        # Check each test for issues
        if "progressive_load" in test_results:
            prog_result = test_results["progressive_load"]
            if prog_result["analysis"]["memory_limit_reached"]:
                summary["issues_found"].append("Memory limits reached with large images")
                summary["memory_management"] = "Fair"
        
        if "concurrent_requests" in test_results:
            conc_result = test_results["concurrent_requests"]
            max_concurrent = conc_result["analysis"]["max_concurrent_supported"]
            if max_concurrent < 3:
                summary["issues_found"].append("Low concurrent request capacity")
                summary["queue_performance"] = "Poor"
        
        if "memory_recovery" in test_results:
            recovery_result = test_results["memory_recovery"]
            if recovery_result["recovery_rate"] < 80:
                summary["issues_found"].append("Poor memory recovery after errors")
                summary["recovery_capability"] = "Poor"
                summary["overall_stability"] = "Fair"
        
        return summary
        
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze results and provide specific recommendations
        if "progressive_load" in test_results:
            prog_result = test_results["progressive_load"]["analysis"]
            if prog_result["memory_limit_reached"]:
                recommendations.append("Implement stricter memory limits for large images")
                recommendations.append("Consider reducing maximum allowed image dimensions")
        
        if "concurrent_requests" in test_results:
            conc_result = test_results["concurrent_requests"]["analysis"]
            if conc_result["max_concurrent_supported"] < 3:
                recommendations.append("Implement request queuing to handle concurrent requests")
                recommendations.append("Consider reducing memory usage per request")
        
        if "queue_behavior" in test_results:
            queue_result = test_results["queue_behavior"]
            if queue_result["success_rate"] < 70:
                recommendations.append("Optimize queue management and timeout handling")
        
        # General GTX 1060 recommendations
        recommendations.extend([
            "Monitor GPU memory usage continuously",
            "Implement aggressive memory cleanup between requests",
            "Use attention slicing and CPU offloading for memory efficiency",
            "Set appropriate request timeouts based on image size",
            "Consider implementing request prioritization"
        ])
        
        return recommendations


async def main():
    """Main stress test execution."""
    parser = argparse.ArgumentParser(description="GTX 1060 Memory Stress Test")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="memory_stress_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Run stress test
    stress_test = MemoryStressTest(args.api_url)
    
    try:
        report = await stress_test.run_all_tests()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("MEMORY STRESS TEST SUMMARY")
        print("="*60)
        
        summary = report["summary"]
        print(f"Overall Stability: {summary['overall_stability']}")
        print(f"Memory Management: {summary['memory_management']}")
        print(f"Queue Performance: {summary['queue_performance']}")
        print(f"Recovery Capability: {summary['recovery_capability']}")
        
        if summary["issues_found"]:
            print("\nIssues Found:")
            for issue in summary["issues_found"]:
                print(f"  - {issue}")
        else:
            print("\nNo major issues found!")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
        
        print(f"\nFull report saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))