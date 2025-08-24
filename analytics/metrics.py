from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Union
import numpy as np
import pandas as pd

Scalar = Union[float, int, str, None]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    fn: Callable[[pd.Series], Scalar]
    applies_to: str  # "numeric" | "any"


def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mean(s: pd.Series) -> Scalar: return _as_num(s).mean()
def _median(s: pd.Series) -> Scalar: return _as_num(s).median()
def _std(s: pd.Series) -> Scalar: return _as_num(s).std(ddof=1)
def _min(s: pd.Series) -> Scalar: return _as_num(s).min()
def _max(s: pd.Series) -> Scalar: return _as_num(s).max()
def _sum(s: pd.Series) -> Scalar: return _as_num(s).sum()


def _q(p: float):
    def inner(s: pd.Series) -> Scalar:
        return _as_num(s).quantile(p)
    return inner


def _iqr(s: pd.Series) -> Scalar:
    x = _as_num(s)
    return x.quantile(0.75) - x.quantile(0.25)


def _nunique(s: pd.Series) -> Scalar: return int(s.nunique(dropna=True))


def _unique_share_pct(s: pd.Series) -> Scalar:
    n = len(s)
    return float(s.nunique(dropna=True) / n * 100.0) if n else np.nan


def _missing(s: pd.Series) -> Scalar: return int(s.isna().sum())
def _missing_pct(s: pd.Series) -> Scalar: return float(s.isna().mean() * 100.0)
def _zeros(s: pd.Series) -> Scalar: return int((_as_num(s) == 0).sum())
def _negatives(s: pd.Series) -> Scalar: return int((_as_num(s) < 0).sum())
def _positives(s: pd.Series) -> Scalar: return int((_as_num(s) > 0).sum())
def _skew(s: pd.Series) -> Scalar: return _as_num(s).skew()
def _kurt(s: pd.Series) -> Scalar: return _as_num(s).kurtosis()


REGISTRY: Dict[str, Metric] = {
    "mean":              Metric("mean", "Mean", _mean, "numeric"),
    "median":            Metric("median", "Median", _median, "numeric"),
    "std":               Metric("std", "Std (ddof=1)", _std, "numeric"),
    "min":               Metric("min", "Min", _min, "numeric"),
    "q05":               Metric("q05", "Q05", _q(0.05), "numeric"),
    "q25":               Metric("q25", "Q25", _q(0.25), "numeric"),
    "q50":               Metric("q50", "Q50/Median", _q(0.50), "numeric"),
    "q75":               Metric("q75", "Q75", _q(0.75), "numeric"),
    "q95":               Metric("q95", "Q95", _q(0.95), "numeric"),
    "iqr":               Metric("iqr", "IQR", _iqr, "numeric"),
    "max":               Metric("max", "Max", _max, "numeric"),
    "sum":               Metric("sum", "Sum", _sum, "numeric"),
    "nunique":           Metric("nunique", "# unique", _nunique, "any"),
    "unique_share_pct":  Metric("unique_share_pct", "Unique %", _unique_share_pct, "any"),
    "missing":           Metric("missing", "# missing", _missing, "any"),
    "missing_pct":       Metric("missing_pct", "Missing %", _missing_pct, "any"),
    "zeros":             Metric("zeros", "# zeros", _zeros, "numeric"),
    "negatives":         Metric("negatives", "# negatives", _negatives, "numeric"),
    "positives":         Metric("positives", "# positives", _positives, "numeric"),
    "skew":              Metric("skew", "Skew", _skew, "numeric"),
    "kurtosis":          Metric("kurtosis", "Kurtosis", _kurt, "numeric"),
}


def compute_metrics(series: pd.Series, metric_keys: list[str]) -> Dict[str, Scalar]:
    out: Dict[str, Scalar] = {}
    for key in metric_keys:
        m = REGISTRY.get(key)
        if not m:
            continue
        try:
            if m.applies_to == "numeric" and not pd.api.types.is_numeric_dtype(series):
                x = _as_num(series)
                out[key] = m.fn(x) if x.notna().any() else np.nan
            else:
                out[key] = m.fn(series)
        except Exception:
            out[key] = np.nan
    return out
