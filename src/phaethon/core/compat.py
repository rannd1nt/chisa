from __future__ import annotations

import importlib.util
import importlib.metadata
import warnings
from packaging.version import parse
from typing import Any

def _check_dep(module_name: str, package_name: str | None = None) -> tuple[bool, str | None]:
    if package_name is None:
        package_name = module_name
        
    if importlib.util.find_spec(module_name) is not None:
        try:
            return True, importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return True, "unknown"
    return False, None

def _require_min_version(pkg_name: str, current_ver: str | None, min_ver: str) -> None:
    if current_ver and current_ver != "unknown":
        if parse(current_ver) < parse(min_ver):
            warnings.warn(
                f"\033[33m[Phaethon Warning]\033[0m {pkg_name} version {current_ver} is installed, "
                f"but Phaethon recommends >= {min_ver}. Expect potential instability."
            )

HAS_NUMPY, NUMPY_VERSION = _check_dep("numpy")
HAS_PANDAS, PANDAS_VERSION = _check_dep("pandas")
HAS_POLARS, POLARS_VERSION = _check_dep("polars")
HAS_SKLEARN, SKLEARN_VERSION = _check_dep("sklearn", "scikit-learn")
HAS_TORCH, TORCH_VERSION = _check_dep("torch")
HAS_RAPIDFUZZ, RAPIDFUZZ_VERSION = _check_dep("rapidfuzz")
HAS_PYARROW, PYARROW_VERSION = _check_dep("pyarrow")
HAS_H5PY, H5PY_VERSION = _check_dep("h5py")

if not HAS_NUMPY:
    raise ImportError("Phaethon requires 'numpy' as its core engine. Please install it: pip install numpy>=1.26.0")
_require_min_version("numpy", NUMPY_VERSION, "1.26.0")

SCHEMA_COMPAT = HAS_PANDAS or HAS_POLARS

def require_dataframe_backend(feature_name: str = "This feature") -> None:
    if not SCHEMA_COMPAT:
        raise ImportError(f"{feature_name} requires Pandas or Polars. Install via: pip install 'phaethon[dataframe]'")

def require_torch(feature_name: str = "This feature") -> None:
    if not HAS_TORCH:
        raise ImportError(f"{feature_name} requires PyTorch. Install via: pip install 'phaethon[pinns]' or torch>=2.0.0")

def require_sklearn(feature_name: str = "This feature") -> None:
    if not HAS_SKLEARN:
        raise ImportError(f"{feature_name} requires Scikit-Learn. Install via: pip install 'phaethon[ml]' or scikit-learn>=1.3.0")

def require_parquet(feature_name: str = "Parquet I/O") -> None:
    if not HAS_PYARROW:
        raise ImportError(f"{feature_name} requires PyArrow. Install via: pip install 'phaethon[io]' or pyarrow>=14.0.0")

def require_h5py(feature_name: str = "HDF5 I/O") -> None:
    if not HAS_H5PY:
        raise ImportError(f"{feature_name} requires h5py. Install via: pip install 'phaethon[io]' or h5py>=3.0.0")

def is_pandas_df(df: Any) -> bool:
    return HAS_PANDAS and df.__class__.__module__.startswith('pandas')

def is_polars_df(df: Any) -> bool:
    return HAS_POLARS and df.__class__.__module__.startswith('polars')