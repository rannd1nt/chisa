"""
Dimensional Linear Algebra Module.
Provides strict, physics-aware matrix operations preserving 
dimensional integrity and isotropic scaling.
"""
from .core.linalg import inv, det, solve, norm

__all__ = ["inv", "det", "solve", "norm"]