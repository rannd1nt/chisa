"""
Random Physics Module.
Generates stochastic tensors strictly bounded by physical dimensions.
"""
from __future__ import annotations
from typing import Any

from .compat import HAS_NUMPY
from .registry import ureg
from .base import BaseUnit

if HAS_NUMPY:
    import numpy as np

def _resolve_unit(unit: Any) -> type[BaseUnit]:
    if isinstance(unit, str):
        return ureg().get_unit_class(unit)
    if isinstance(unit, type) and issubclass(unit, BaseUnit):
        return unit
    raise TypeError("The 'unit' argument must be a string alias or a BaseUnit class.")

class RandomState:
    def __init__(self, seed: int | None = None) -> None:
        if HAS_NUMPY:
            self._rng = np.random.default_rng(seed)

    def seed(self, seed: int | None = None) -> None:
        """
        Reseeds the isolated physics random number generator.
        
        Crucial for ensuring absolute reproducibility in stochastic physical models,
        thermodynamic simulations, or machine learning cross-validations.
        
        Args:
            seed: An integer to initialize the internal BitGenerator. 
                    If None, fresh, unpredictable entropy will be pulled from the OS.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        self._rng = np.random.default_rng(seed)

    def uniform(self, low=0.0, high=1.0, size=None, unit=None) -> BaseUnit:
        """
        Draws samples from a uniform distribution and injects physical DNA.
        
        Args:
            low: Lower boundary of the output interval.
            high: Upper boundary of the output interval.
            size: Output shape (e.g., (2, 3)).
            unit: The physical dimension to attach (Class or alias string).
            
        Returns:
            A BaseUnit tensor containing uniformly distributed physical values.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if unit is None: raise ValueError("A physical unit must be specified.")
            
        raw_arr = self._rng.uniform(low, high, size)
        UnitClass = _resolve_unit(unit)
        return UnitClass(raw_arr)

    def normal(self, loc=0.0, scale=1.0, size=None, unit=None) -> BaseUnit:
        """
        Draws random samples from a normal (Gaussian) distribution.
        
        Args:
            loc: Mean ("centre") of the distribution.
            scale: Standard deviation (spread or "width").
            size: Output shape.
            unit: The physical dimension to attach.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if unit is None: raise ValueError("A physical unit must be specified.")
            
        raw_arr = self._rng.normal(loc, scale, size)
        UnitClass = _resolve_unit(unit)
        return UnitClass(raw_arr)

    def poisson(self, lam=1.0, size=None, unit=None) -> BaseUnit:
        """
        Draws samples from a Poisson distribution.
        
        Extremely useful in Phaethon for modeling discrete physical events over 
        a continuous interval, such as radioactive decays (u.Becquerel) or 
        photon strikes (u.Photon).
        
        Args:
            lam: Expected number of events occurring in a fixed-time interval.
            size: Output shape.
            unit: The physical dimension to attach.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if unit is None: raise ValueError("A physical unit must be specified.")
            
        raw_arr = self._rng.poisson(lam, size)
        UnitClass = _resolve_unit(unit)
        return UnitClass(raw_arr)

    def exponential(self, scale=1.0, size=None, unit=None) -> BaseUnit:
        """
        Draws samples from an exponential distribution.
        
        Ideal for simulating the time between independent physics events, 
        such as the decay time of radioactive isotopes or thermodynamic 
        relaxation times.
        
        Args:
            scale: The scale parameter, β = 1/λ. Must be non-negative.
            size: Output shape.
            unit: The physical dimension to attach (typically u.Second).
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if unit is None: raise ValueError("A physical unit must be specified.")
            
        raw_arr = self._rng.exponential(scale, size)
        UnitClass = _resolve_unit(unit)
        return UnitClass(raw_arr)
    
    def randint(self, low: int, high: int | None = None, size=None, unit=None) -> BaseUnit:
        """
        Draws random integers from a discrete uniform distribution.
        
        Crucial for quantum mechanics, statistical grids, or any physical domain
        where magnitudes are strictly quantized.
        
        Args:
            low: Lowest (signed) integer to be drawn from the distribution.
            high: One above the largest (signed) integer to be drawn.
            size: Output shape.
            unit: The physical dimension to attach.
            
        Returns:
            A BaseUnit tensor containing discrete, uniformly distributed integers.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if unit is None: raise ValueError("A physical unit must be specified.")
            
        raw_arr = self._rng.integers(low, high, size=size)
        UnitClass = _resolve_unit(unit)
        return UnitClass(raw_arr)

    def choice(self, a, size=None, replace=True, p=None, unit=None) -> BaseUnit:
        """
        Generates a random sample from a given 1-D array of physical states.
        
        Allows physical entities to randomly collapse into a predefined set of 
        allowed states. Ideal for simulating quantum state measurements or 
        drawing specific velocity vectors in a Monte Carlo gas simulation.
        
        Args:
            a: A 1-D array-like of allowed magnitudes.
            size: Output shape.
            replace: Whether the sample is with or without replacement.
            p: The probabilities associated with each entry in 'a'.
            unit: The physical dimension to attach.
            
        Returns:
            A BaseUnit tensor representing the collapsed random states.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if unit is None: raise ValueError("A physical unit must be specified.")
            
        raw_arr = self._rng.choice(a, size=size, replace=replace, p=p)
        UnitClass = _resolve_unit(unit)
        return UnitClass(raw_arr)
    
    def shuffle(self, x: BaseUnit) -> None:
        """
        Modifies a physical tensor sequence in-place by shuffling its contents.
        
        Randomizes the distribution of physical magnitudes along the first axis
        without destroying or altering the underlying dimensional DNA. 
        Vital for dataset splitting or cross-validation in `phaethon.ml`.
        
        Args:
            x: The Phaethon BaseUnit array to be shuffled in-place.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        if not isinstance(x, BaseUnit):
            raise TypeError("The input to shuffle must be a Phaethon BaseUnit.")
        
        self._rng.shuffle(x._value)

    def permutation(self, x: BaseUnit | int) -> BaseUnit | Any:
        """
        Randomly permutes a physical tensor, returning a completely NEW copy.
        
        Unlike `shuffle`, this method does not mutate the original tensor. 
        If an integer is passed, it returns a permuted range of dimensionless integers.
        
        Args:
            x: A Phaethon BaseUnit array, or an integer to define a range.
            
        Returns:
            A new BaseUnit tensor with randomly permuted elements along the first axis.
        """
        if not HAS_NUMPY: raise ImportError("NumPy is required.")
        
        if isinstance(x, int):
            from .units.scalar import Dimensionless
            return Dimensionless(self._rng.permutation(x))
            
        if not isinstance(x, BaseUnit):
            raise TypeError("The input must be an integer or a Phaethon BaseUnit.")
            
        raw_perm = self._rng.permutation(x.mag)
        return x.__class__(raw_perm, context=x.context)

_rand = RandomState()

seed = _rand.seed
uniform = _rand.uniform
normal = _rand.normal
poisson = _rand.poisson
exponential = _rand.exponential
randint = _rand.randint
choice = _rand.choice
shuffle = _rand.shuffle
permutation = _rand.permutation