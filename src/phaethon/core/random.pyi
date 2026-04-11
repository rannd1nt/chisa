import numpy as np
import numpy.typing as npt
from typing import Any, overload
from .._typing import _UnitT
from .base import BaseUnit

class RandomState:
    def __init__(self, seed: int | None = ...) -> None: ...
    
    def seed(self, seed: int | None = ...) -> None: ...

    @overload
    def uniform(self, low: float = ..., high: float = ..., size: Any = ..., unit: type[_UnitT] = ...) -> _UnitT: ...
    @overload
    def uniform(self, low: float = ..., high: float = ..., size: Any = ..., unit: str = ...) -> BaseUnit: ...
    
    @overload
    def normal(self, loc: float = ..., scale: float = ..., size: Any = ..., unit: type[_UnitT] = ...) -> _UnitT: ...
    @overload
    def normal(self, loc: float = ..., scale: float = ..., size: Any = ..., unit: str = ...) -> BaseUnit: ...
    
    @overload
    def poisson(self, lam: float = ..., size: Any = ..., unit: type[_UnitT] = ...) -> _UnitT: ...
    @overload
    def poisson(self, lam: float = ..., size: Any = ..., unit: str = ...) -> BaseUnit: ...
    
    @overload
    def exponential(self, scale: float = ..., size: Any = ..., unit: type[_UnitT] = ...) -> _UnitT: ...
    @overload
    def exponential(self, scale: float = ..., size: Any = ..., unit: str = ...) -> BaseUnit: ...

    @overload
    def randint(self, low: int, high: int | None = ..., size: Any = ..., unit: type[_UnitT] = ...) -> _UnitT: ...
    @overload
    def randint(self, low: int, high: int | None = ..., size: Any = ..., unit: str = ...) -> BaseUnit: ...

    @overload
    def choice(self, a: Any, size: Any = ..., replace: bool = ..., p: Any = ..., unit: type[_UnitT] = ...) -> _UnitT: ...
    @overload
    def choice(self, a: Any, size: Any = ..., replace: bool = ..., p: Any = ..., unit: str = ...) -> BaseUnit: ...

    def shuffle(self, x: BaseUnit) -> None: ...

    @overload
    def permutation(self, x: int) -> BaseUnit: ...
    @overload
    def permutation(self, x: _UnitT) -> _UnitT: ...

_rand: RandomState

seed = _rand.seed
uniform = _rand.uniform
normal = _rand.normal
poisson = _rand.poisson
exponential = _rand.exponential
randint = _rand.randint
choice = _rand.choice
shuffle = _rand.shuffle
permutation = _rand.permutation