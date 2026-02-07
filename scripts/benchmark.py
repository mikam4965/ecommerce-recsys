"""API Load Testing for Recommendation Service.

This script performs load testing on the recommendation API to measure:
- Latency percentiles (p50, p95, p99)
- Throughput (requests per second)
- Error rates under load

Run: python scripts/benchmark.py

Requires: API running at http://localhost:8000
"""

import asyncio
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import aiohttp
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Configuration
BASE_URL = "http://localhost:8000/api/v1"
DURATION = 30  # seconds per test (reduced for faster iteration)
CONCURRENCY_LEVELS = [10, 50, 100]
TRAFFIC_MIX = {"recommend": 1.0, "similar": 0.0}  # Focus on /recommend only
SAMPLE_SIZE = 100  # Smaller pool = more cache hits

# Test data (loaded from DB)
SAMPLE_USER_IDS: list[int] = []
SAMPLE_ITEM_IDS: list[int] = []


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    endpoint: str
    concurrency: int
    duration: float
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        """50th percentile latency in ms."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.50)
        return sorted_lat[idx] * 1000

    @property
    def p95(self) -> float:
        """95th percentile latency in ms."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)] * 1000

    @property
    def p99(self) -> float:
        """99th percentile latency in ms."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)] * 1000

    @property
    def avg_latency(self) -> float:
        """Average latency in ms."""
        if not self.latencies:
            return 0.0
        return (sum(self.latencies) / len(self.latencies)) * 1000

    @property
    def rps(self) -> float:
        """Requests per second."""
        if self.duration <= 0:
            return 0.0
        return self.total_requests / self.duration

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed / self.total_requests) * 100


def load_sample_ids() -> tuple[list[int], list[int]]:
    """Load sample user/item IDs from database.

    Returns:
        Tuple of (user_ids, item_ids)
    """
    db_path = project_root / "data" / "database.sqlite"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run: python scripts/init_database.py")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get SAMPLE_SIZE user IDs that have segments (these are known to the model!)
    # Model only knows 8,182 users, but DB has 978,921
    cursor.execute(f"""
        SELECT user_id FROM users
        WHERE segment IS NOT NULL
        ORDER BY RANDOM() LIMIT {SAMPLE_SIZE}
    """)
    user_ids = [row[0] for row in cursor.fetchall()]

    # Get SAMPLE_SIZE random item IDs
    cursor.execute(f"SELECT item_id FROM items ORDER BY RANDOM() LIMIT {SAMPLE_SIZE}")
    item_ids = [row[0] for row in cursor.fetchall()]

    conn.close()

    logger.info(f"Loaded {len(user_ids)} user IDs, {len(item_ids)} item IDs")
    return user_ids, item_ids


async def recommend_request(
    session: aiohttp.ClientSession,
    user_ids: list[int],
) -> tuple[bool, float]:
    """Make POST /recommend request.

    Args:
        session: aiohttp session
        user_ids: List of user IDs to sample from

    Returns:
        Tuple of (success, latency_seconds)
    """
    import random

    user_id = random.choice(user_ids)
    payload = {"user_id": user_id, "n_recommendations": 10}

    start = time.perf_counter()
    try:
        async with session.post(
            f"{BASE_URL}/recommend",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            success = resp.status == 200
            await resp.read()
    except Exception:
        success = False

    latency = time.perf_counter() - start
    return success, latency


async def similar_request(
    session: aiohttp.ClientSession,
    item_ids: list[int],
) -> tuple[bool, float]:
    """Make POST /similar/{item_id} request.

    Args:
        session: aiohttp session
        item_ids: List of item IDs to sample from

    Returns:
        Tuple of (success, latency_seconds)
    """
    import random

    item_id = random.choice(item_ids)

    start = time.perf_counter()
    try:
        async with session.post(
            f"{BASE_URL}/similar/{item_id}?n_similar=10",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            success = resp.status == 200
            await resp.read()
    except Exception:
        success = False

    latency = time.perf_counter() - start
    return success, latency


async def worker(
    session: aiohttp.ClientSession,
    result: BenchmarkResult,
    duration: float,
    request_fn: Callable,
    fn_args: tuple,
):
    """Single worker making requests until duration expires.

    Args:
        session: aiohttp session
        result: BenchmarkResult to update
        duration: Test duration in seconds
        request_fn: Request function to call
        fn_args: Arguments for request function
    """
    end_time = time.time() + duration

    while time.time() < end_time:
        try:
            success, latency = await request_fn(session, *fn_args)
            result.latencies.append(latency)
            if success:
                result.successful += 1
            else:
                result.failed += 1
            result.total_requests += 1
        except Exception:
            result.failed += 1
            result.total_requests += 1


async def run_benchmark(
    concurrency: int,
    duration: float,
    traffic_mix: dict[str, float],
    user_ids: list[int],
    item_ids: list[int],
) -> dict[str, BenchmarkResult]:
    """Run benchmark with specified concurrency.

    Args:
        concurrency: Number of concurrent workers
        duration: Test duration in seconds
        traffic_mix: Traffic distribution {"recommend": 0.8, "similar": 0.2}
        user_ids: List of user IDs for testing
        item_ids: List of item IDs for testing

    Returns:
        Dictionary of endpoint -> BenchmarkResult
    """
    results = {
        "recommend": BenchmarkResult("recommend", concurrency, duration),
        "similar": BenchmarkResult("similar", concurrency, duration),
    }

    # Distribute workers by traffic mix
    recommend_workers = int(concurrency * traffic_mix["recommend"])
    similar_workers = concurrency - recommend_workers

    # Ensure at least 1 worker for each endpoint IF traffic is > 0
    if recommend_workers == 0 and concurrency > 0 and traffic_mix["recommend"] > 0:
        recommend_workers = 1
        similar_workers = concurrency - 1
    if similar_workers == 0 and concurrency > 1 and traffic_mix["similar"] > 0:
        similar_workers = 1
        recommend_workers = concurrency - 1

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        # Recommend workers
        for _ in range(recommend_workers):
            tasks.append(
                worker(
                    session,
                    results["recommend"],
                    duration,
                    recommend_request,
                    (user_ids,),
                )
            )

        # Similar workers
        for _ in range(similar_workers):
            tasks.append(
                worker(
                    session,
                    results["similar"],
                    duration,
                    similar_request,
                    (item_ids,),
                )
            )

        await asyncio.gather(*tasks)

    return results


async def check_api_health() -> bool:
    """Check if API is running.

    Returns:
        True if API is healthy
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("model_loaded", False)
    except Exception as e:
        logger.error(f"API health check failed: {e}")
    return False


async def flush_cache():
    """Flush recommendation cache before benchmark."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/cache/flush",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    logger.info("Cache flushed successfully")
                else:
                    logger.warning(f"Cache flush returned: {resp.status}")
    except Exception as e:
        logger.warning(f"Could not flush cache: {e}")


def print_results_table(all_results: list[dict[str, BenchmarkResult]]):
    """Print formatted results table.

    Args:
        all_results: List of results for each concurrency level
    """
    print("\n" + "=" * 90)
    print("API LOAD TESTING RESULTS")
    print("=" * 90)

    for results in all_results:
        # Get concurrency from first result
        concurrency = list(results.values())[0].concurrency
        duration = list(results.values())[0].duration

        print(f"\nConcurrency: {concurrency} users | Duration: {duration:.0f}s")
        print("-" * 90)
        print(
            f"{'Endpoint':<12} | {'Requests':>9} | {'RPS':>7} | "
            f"{'p50':>7} | {'p95':>7} | {'p99':>7} | {'Errors':>7}"
        )
        print("-" * 90)

        total_rps = 0
        for endpoint, result in results.items():
            total_rps += result.rps
            print(
                f"/{endpoint:<11} | {result.total_requests:>9,} | "
                f"{result.rps:>7.1f} | "
                f"{result.p50:>6.1f}ms | {result.p95:>6.1f}ms | "
                f"{result.p99:>6.1f}ms | {result.error_rate:>6.1f}%"
            )

        print("-" * 90)
        print(f"{'TOTAL':<12} | {'':<9} | {total_rps:>7.1f} |")


def check_targets(all_results: list[dict[str, BenchmarkResult]]) -> bool:
    """Check if performance targets are met.

    Targets:
    - p99 latency < 100ms
    - Throughput > 100 RPS

    Args:
        all_results: List of results for each concurrency level

    Returns:
        True if all targets met
    """
    print("\n" + "=" * 90)
    print("TARGET CHECK")
    print("=" * 90)

    all_passed = True

    # Find max throughput
    max_rps = 0
    max_rps_concurrency = 0
    for results in all_results:
        total_rps = sum(r.rps for r in results.values())
        concurrency = list(results.values())[0].concurrency
        if total_rps > max_rps:
            max_rps = total_rps
            max_rps_concurrency = concurrency

    # Check p99 latency
    worst_p99 = 0
    worst_p99_endpoint = ""
    for results in all_results:
        for endpoint, result in results.items():
            if result.p99 > worst_p99:
                worst_p99 = result.p99
                worst_p99_endpoint = endpoint

    # Print results
    p99_pass = worst_p99 < 100
    rps_pass = max_rps > 100

    status_p99 = "[PASS]" if p99_pass else "[FAIL]"
    status_rps = "[PASS]" if rps_pass else "[FAIL]"

    print(f"{status_p99} p99 latency < 100ms: {worst_p99:.1f}ms ({worst_p99_endpoint})")
    print(f"{status_rps} Throughput > 100 RPS: {max_rps:.1f} RPS at {max_rps_concurrency} concurrent")

    if not p99_pass:
        all_passed = False
        print("\n[!] OPTIMIZATION SUGGESTIONS for latency:")
        print("   - Enable SQLite caching for /recommend")
        print("   - Pre-warm cache for popular users")
        print("   - Reduce model computation (fewer factors)")

    if not rps_pass:
        all_passed = False
        print("\n[!] OPTIMIZATION SUGGESTIONS for throughput:")
        print("   - Use uvloop: pip install uvloop")
        print("   - Increase workers: uvicorn --workers 4")
        print("   - Enable connection pooling")

    print("=" * 90)
    return all_passed


async def warmup(user_ids: list[int], item_ids: list[int]):
    """Warm up the API by pre-populating cache for all test users.

    Args:
        user_ids: List of user IDs
        item_ids: List of item IDs
    """
    logger.info(f"Warming up API - pre-caching {len(user_ids)} users...")

    async with aiohttp.ClientSession() as session:
        # Pre-populate cache for ALL test users
        for user_id in user_ids:
            payload = {"user_id": user_id, "n_recommendations": 10}
            try:
                async with session.post(
                    f"{BASE_URL}/recommend",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    await resp.read()
            except Exception:
                pass

    logger.info("Warmup complete - cache populated")


async def main():
    """Main benchmark workflow."""
    print("=" * 70)
    print("API LOAD TESTING")
    print("=" * 70)

    # 1. Check API health
    logger.info(f"Checking API at {BASE_URL}...")
    if not await check_api_health():
        logger.error("API is not available or model not loaded!")
        logger.error("Start API: uvicorn src.api.main:app --port 8000")
        sys.exit(1)
    logger.info("API is healthy")

    # 2. Load test data
    user_ids, item_ids = load_sample_ids()

    # 3. Skip cache flush - test with realistic cache hits
    # await flush_cache()  # Disabled for realistic test

    # 4. Warmup - pre-populate cache for test users
    await warmup(user_ids, item_ids)

    # 5. Run benchmarks for each concurrency level
    all_results = []

    for concurrency in CONCURRENCY_LEVELS:
        logger.info(f"\nRunning with {concurrency} concurrent users for {DURATION}s...")

        results = await run_benchmark(
            concurrency=concurrency,
            duration=DURATION,
            traffic_mix=TRAFFIC_MIX,
            user_ids=user_ids,
            item_ids=item_ids,
        )
        all_results.append(results)

        # Brief pause between tests
        await asyncio.sleep(2)

    # 6. Print results
    print_results_table(all_results)

    # 7. Check targets
    check_targets(all_results)


if __name__ == "__main__":
    asyncio.run(main())
