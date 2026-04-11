"""
Random Physics Module.
Generates stochastic tensors strictly bounded by physical dimensions.
"""
from .core.random import (
    seed, uniform, normal, poisson, exponential, randint, choice, shuffle, permutation
)

__all__ = [
    "seed", "uniform", "normal", "poisson", "exponential", 
    "randint", "choice", "shuffle", "permutation"
]