from __future__ import annotations

from typing import Any, TYPE_CHECKING, TypeAlias, Iterable, TypeVar, Callable, Literal, Protocol, Mapping, TypedDict
from phaethon.core.compat import HAS_POLARS, HAS_PANDAS, HAS_NUMPY, HAS_TORCH

if TYPE_CHECKING:
    if HAS_POLARS: import polars as pl
    if HAS_PANDAS: import pandas as pd
    if HAS_NUMPY: import numpy as np
    if HAS_TORCH: import torch
    from .core.base import BaseUnit

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
    NumericLike: TypeAlias = int | float | str | np.ndarray | Iterable[Any]
    UnitLike: TypeAlias = str | type[BaseUnit]

    ConvertibleInput: TypeAlias = NumericLike | 'BaseUnit' | Iterable['BaseUnit']
    ColumnTarget: TypeAlias = str | Any
    Extractable: TypeAlias = BaseUnit | torch.Tensor | np.ndarray | float | int | str | list[Any] | tuple[Any, ...] | None
    UnwrappedArray: TypeAlias = np.ndarray | float | int | None

    ContextDict: TypeAlias = dict[str, NumericLike | BaseUnit]
    AliasRegistry: TypeAlias = dict[str, str | list[str]]

    ImputeMethod: TypeAlias = Literal['mean', 'median', 'mode', 'ffill', 'bfill'] | str | float
    InterpolationMethod = Literal[
        "linear", "nearest", "time", "index", "values", "pad", "zero", "slinear","quadratic",
        "cubic", "spline", "barycentric", "polynomial", "krogh","piecewise_polynomial",
        "pchip", "akima", "cubicspline"
    ]
    ErrorAction: TypeAlias = Literal['raise', 'coerce', 'clip']
    StrictnessLevel = Literal["default", "strict", "strict_warn", "loose_warn", "ignore"]
    NumDtype: TypeAlias = Literal["float64", "float32", "float16", "int64", "int32"]

    if HAS_TORCH:
        from .core.pinns.tensor import PTensor
        TensorLikeDict: TypeAlias = dict[str, PTensor | torch.Tensor]
        TensorLikeTuple: TypeAlias = tuple[PTensor | torch.Tensor, ...]
        GradTarget: TypeAlias = bool | list[str]

    class _ParquetConfig(TypedDict, total=False):
        engine: Literal["pyarrow", "fastparquet", "auto"]
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"]
        index: bool
        partition_cols: list[str]

    class _HDF5Config(TypedDict, total=False):
        compression: Literal["gzip", "lzf", "szip"]
        compression_opts: int
        chunks: bool | tuple[int, ...]

    DatasetInput: TypeAlias = Mapping[str, Any] | Iterable[Any] | Any
    DatasetStateDict: TypeAlias = dict[str, dict[str, Any]]

    class _ResponsiveTableConfig(TypedDict, total=False):
        max_col_width: int
        float_format: str
        justify: Literal["left", "right", "center"]
        
else:
    DataFrameLike: TypeAlias = Any
    NumericLike: TypeAlias = Any
    UnitLike: TypeAlias = Any
    ConvertibleInput: TypeAlias = Any
    ColumnTarget: TypeAlias = Any
    Extractable: TypeAlias = Any
    UnwrappedArray: TypeAlias = Any
    ContextDict: TypeAlias = Any
    AliasRegistry: TypeAlias = Any
    ImputeMethod: TypeAlias = Any
    InterpolationMethod: TypeAlias = Any
    ErrorAction: TypeAlias = Any
    StrictnessLevel: TypeAlias = Any
    NumDtype: TypeAlias = Any
    TensorLikeDict: TypeAlias = Any
    TensorLikeTuple: TypeAlias = Any
    GradTarget: TypeAlias = Any

_Signature: TypeAlias = frozenset[tuple[str, int]]
_DataFrameT = TypeVar("_DataFrameT", bound=DataFrameLike)
_NumericT = TypeVar("_NumericT", bound=NumericLike)
_UnitT = TypeVar("_UnitT", bound='BaseUnit')
_UnitT_co = TypeVar("_UnitT_co", bound='BaseUnit', covariant=True)
_UnitClassT = TypeVar("_UnitClassT", bound=type)
_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])
_ReturnT = TypeVar("_ReturnT")
_KeyT = TypeVar('_KeyT')

class SupportsPredict(Protocol):
    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...

class SupportsTransform(Protocol):
    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any: ...
    def transform(self, X: Any) -> Any: ...

class SupportsInverseTransform(SupportsTransform, Protocol):
    def inverse_transform(self, X: Any) -> Any: ...

_EstimatorT = TypeVar("_EstimatorT", bound=SupportsPredict)
_TransformerT = TypeVar("_TransformerT", bound=SupportsTransform)
_InvTransformerT = TypeVar("_InvTransformerT", bound=SupportsInverseTransform)