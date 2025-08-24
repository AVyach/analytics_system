import pandas as pd
from analytics.metrics import compute_metrics


def test_basic_numeric_metrics():
    s = pd.Series([1, 2, 3, 4, 5])
    out = compute_metrics(
        s, ["mean", "median", "min", "max", "std", "q25", "q75", "iqr"])
    assert round(out["mean"], 3) == 3.0
    assert out["min"] == 1
    assert out["max"] == 5
    assert round(out["iqr"], 3) == round(out["q75"] - out["q25"], 3)


def test_categorical_metrics():
    s = pd.Series(["a", "a", "b", None])
    out = compute_metrics(
        s, ["nunique", "missing", "missing_pct", "unique_share_pct"])
    assert out["nunique"] == 2
    assert out["missing"] == 1
