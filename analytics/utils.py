from __future__ import annotations
import io
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_robust(raw_bytes: bytes, encoding: str | None = None) -> pd.DataFrame:
    buf = io.BytesIO(raw_bytes)
    try:
        return pd.read_csv(buf, sep=None, engine="python", encoding=encoding)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, encoding=encoding)


def corr_figure(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix (numeric)")
    fig.tight_layout()
    return fig
