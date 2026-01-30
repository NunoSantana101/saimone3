"""
Simple RNG Utility for Monte Carlo Simulations
Designed for ChatGPT 4.1 Code Interpreter Integration
================================================
v1.2 - Added traceability for lognormal validation
"""

import random
import math
from typing import List, Tuple, Optional, Union

class MCRandom:
    """
    Lightweight RNG wrapper for Monte Carlo simulations.
    Provides reproducible, seedable random generation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        self.rng = random.Random(seed)
        self._seed = seed
    
    def set_seed(self, seed: int) -> None:
        """Reset RNG with new seed."""
        self._seed = seed
        self.rng.seed(seed)
    
    def get_seed(self) -> Optional[int]:
        """Return the seed used for initialization."""
        return self._seed
    
    # === Core Distributions ===
    
    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Uniform distribution U(low, high)."""
        return self.rng.uniform(low, high)
    
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Normal/Gaussian distribution N(mu, sigma)."""
        return self.rng.gauss(mu, sigma)
    
    def triangular(self, low: float, mode: float, high: float) -> float:
        """Triangular distribution - useful for expert estimates."""
        return self.rng.triangular(low, high, mode)
    
    def lognormal(self, mu: float, sigma: float) -> float:
        """
        Log-normal distribution.
        
        Returns exp(normal(mu, sigma)).
        For validation: use lognormal_traced() to get underlying normal.
        """
        return self.rng.lognormvariate(mu, sigma)
    
    def lognormal_traced(self, mu: float, sigma: float) -> Tuple[float, float]:
        """
        Log-normal with traceability - returns (lognormal_value, underlying_normal).
        
        Use this for audit validation:
            ln_val, norm_val = rng.lognormal_traced(2.08, 0.44)
            assert abs(ln_val - math.exp(norm_val)) < 1e-12
        """
        underlying_normal = self.rng.gauss(mu, sigma)
        lognormal_value = math.exp(underlying_normal)
        return lognormal_value, underlying_normal
    
    def beta(self, alpha: float, beta: float) -> float:
        """Beta distribution - useful for probabilities/proportions."""
        return self.rng.betavariate(alpha, beta)
    
    # === Discrete Distributions ===
    
    def bernoulli(self, p: float) -> int:
        """Bernoulli trial with probability p. Returns 0 or 1."""
        return 1 if self.rng.random() < p else 0
    
    def binomial(self, n: int, p: float, size: Optional[int] = None) -> Union[int, List[int]]:
        """
        Binomial distribution - number of successes in n trials with probability p.
        
        Args:
            n: Number of trials
            p: Probability of success per trial
            size: Number of samples to generate (optional)
        
        Returns:
            Single int if size is None, else list of ints
        """
        def single_binomial():
            successes = 0
            for _ in range(n):
                if self.rng.random() < p:
                    successes += 1
            return successes
        
        if size is None:
            return single_binomial()
        else:
            return [single_binomial() for _ in range(size)]
    
    def poisson(self, lam: float, size: Optional[int] = None) -> Union[int, List[int]]:
        """
        Poisson distribution with rate parameter lambda.
        
        Args:
            lam: Rate parameter (expected number of events)
            size: Number of samples to generate (optional)
        
        Returns:
            Single int if size is None, else list of ints
        """
        def single_poisson():
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= self.rng.random()
            return k - 1
        
        if size is None:
            return single_poisson()
        else:
            return [single_poisson() for _ in range(size)]
    
    # === MC Simulation Helpers ===
    
    def sample_n(self, distribution: str, n: int, **params) -> List[float]:
        """
        Generate n samples from specified distribution.
        
        Args:
            distribution: 'uniform', 'normal', 'triangular', 'lognormal', 'beta', 'binomial', 'poisson'
            n: number of samples
            **params: distribution parameters
        
        Returns:
            List of n random samples
        """
        dist_map = {
            'uniform': lambda: self.uniform(params.get('low', 0), params.get('high', 1)),
            'normal': lambda: self.normal(params.get('mu', 0), params.get('sigma', 1)),
            'triangular': lambda: self.triangular(params['low'], params['mode'], params['high']),
            'lognormal': lambda: self.lognormal(params.get('mu', 0), params.get('sigma', 1)),
            'beta': lambda: self.beta(params['alpha'], params['beta']),
            'binomial': lambda: self.binomial(params['n'], params['p']),
            'poisson': lambda: self.poisson(params['lam']),
        }
        
        sampler = dist_map.get(distribution)
        if not sampler:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return [sampler() for _ in range(n)]
    
    def choice(self, items: list, weights: Optional[List[float]] = None):
        """Weighted random choice from list."""
        if weights:
            return self.rng.choices(items, weights=weights, k=1)[0]
        return self.rng.choice(items)
    
    def shuffle(self, items: list) -> list:
        """Return shuffled copy of list."""
        result = items.copy()
        self.rng.shuffle(result)
        return result
    
    # === Statistics Helpers ===
    
    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        """Calculate p-th percentile (0-100) of data."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)
    
    @staticmethod
    def confidence_interval(data: List[float], ci: float = 95) -> Tuple[float, float]:
        """Calculate confidence interval bounds."""
        lower_p = (100 - ci) / 2
        upper_p = 100 - lower_p
        return (MCRandom.percentile(data, lower_p), MCRandom.percentile(data, upper_p))
    
    @staticmethod
    def summary_stats(data: List[float]) -> dict:
        """Return summary statistics for MC results."""
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1) if n > 1 else 0
        
        return {
            'n': n,
            'mean': mean,
            'std': math.sqrt(variance),
            'min': min(data),
            'max': max(data),
            'p5': MCRandom.percentile(data, 5),
            'p25': MCRandom.percentile(data, 25),
            'p50': MCRandom.percentile(data, 50),
            'p75': MCRandom.percentile(data, 75),
            'p95': MCRandom.percentile(data, 95),
        }
    
    @staticmethod
    def count_frequency(data: List, value) -> Tuple[int, float]:
        """Count occurrences and frequency of a value in data."""
        count = sum(1 for x in data if x == value)
        return count, count / len(data) if data else 0


# === Validation Suite ===
def validate_lognormal_consistency(seed: int = 2026, n: int = 20) -> Tuple[bool, List[dict]]:
    """
    Validate that lognormal = exp(normal) for audit purposes.
    
    Returns (all_match, results_list).
    """
    rng = MCRandom(seed=seed)
    
    results = []
    for i in range(n):
        ln_val, norm_val = rng.lognormal_traced(2.08, 0.44)
        reconstructed = math.exp(norm_val)
        diff = abs(ln_val - reconstructed)
        match = diff < 1e-12
        results.append({
            'index': i + 1,
            'lognormal': ln_val,
            'underlying_normal': norm_val,
            'exp(normal)': reconstructed,
            'diff': diff,
            'match': match
        })
    
    all_match = all(r['match'] for r in results)
    return all_match, results


# === Quick Usage Example ===
if __name__ == "__main__":
    print("=== mc_rng.py v1.2 Validation ===\n")
    
    # Test lognormal traceability
    print("1. Lognormal Traceability Test (seed=2026):")
    all_match, results = validate_lognormal_consistency(seed=2026, n=5)
    
    for r in results:
        print(f"   [{r['index']}] lognormal={r['lognormal']:.6f} | "
              f"exp(normal)={r['exp(normal)']:.6f} | "
              f"match={r['match']}")
    
    print(f"\n   All values match: {all_match}\n")
    
    # Test other distributions
    rng = MCRandom(seed=2026)
    
    print("2. Distribution Tests:")
    print(f"   binomial(10, 0.2, size=5): {rng.binomial(10, 0.2, size=5)}")
    print(f"   poisson(3.5, size=5): {rng.poisson(3.5, size=5)}")
    print(f"   beta(2, 5): {rng.beta(2, 5):.4f}")
    print(f"   triangular(10, 15, 20): {rng.triangular(10, 15, 20):.4f}")
