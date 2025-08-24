# analytics/product_metrics.py
from __future__ import annotations
from typing import Iterable, Optional, Dict
import numpy as np
import pandas as pd

REQUIRED_COLS = {"user_id", "event_time", "event_name"}


def prep_events(df: pd.DataFrame, tz: Optional[str] = None) -> pd.DataFrame:
    if not REQUIRED_COLS.issubset(df.columns):
        missing = list(REQUIRED_COLS - set(df.columns))
        raise ValueError(f"Нет обязательных колонок: {missing}")
    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["event_name"] = out["event_name"].astype(str)

    # Время -> datetime (может быть naive или с TZ)
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")

    # Приведение TZ, если запрошено
    if tz == "UTC":
        if out["event_time"].dt.tz is None:
            out["event_time"] = out["event_time"].dt.tz_localize("UTC")
        else:
            out["event_time"] = out["event_time"].dt.tz_convert("UTC")
    elif tz is None:
        pass  # оставляем как есть
    else:
        if out["event_time"].dt.tz is None:
            out["event_time"] = out["event_time"].dt.tz_localize(tz)
        else:
            out["event_time"] = out["event_time"].dt.tz_convert(tz)

    # Вычислим day/month в НАИВНОМ времени, чтобы избежать конфликтов aware vs naive
    base_time = (out["event_time"].dt.tz_convert(None)
                 if out["event_time"].dt.tz is not None
                 else out["event_time"])
    out["event_date"] = base_time.dt.date
    out["event_month"] = base_time.dt.to_period("M").dt.to_timestamp()

    if "revenue" in out.columns:
        out["revenue"] = pd.to_numeric(
            out["revenue"], errors="coerce").fillna(0.0)
    if "is_paying" not in out.columns:
        out["is_paying"] = (out["event_name"].str.lower()
                            == "purchase").astype(int)

    for c in ("session_start", "session_end"):
        if c in out.columns:
            out[c] = pd.to_datetime(c, errors="coerce")

    return out

# --------- вспомогалки для фильтров ---------


def _normalize_series_to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s


def _normalize_ts_to_naive(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts)
    try:
        return t.tz_localize(None) if t.tzinfo is not None else t
    except Exception:
        return t


def _apply_filters(df: pd.DataFrame, date_from=None, date_to=None, where: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    x = df.copy()
    ev_time_naive = _normalize_series_to_naive(x["event_time"])
    if date_from is not None:
        t_from = _normalize_ts_to_naive(date_from)
        x = x[ev_time_naive >= t_from]
        ev_time_naive = ev_time_naive.loc[x.index]
    if date_to is not None:
        t_to = _normalize_ts_to_naive(date_to)
        x = x[ev_time_naive < t_to]
        ev_time_naive = ev_time_naive.loc[x.index]
    if where:
        for col, vals in where.items():
            if col in x.columns:
                x = x[x[col].isin(list(vals))]
    return x


def _group_keys(dimensions: Iterable[str] | None) -> list[str]:
    if not dimensions:
        return []
    allowed = {"device_type", "channel", "feature"}
    return [d for d in dimensions if d in allowed]

# --------- метрики ---------


def dau(df: pd.DataFrame, dimensions: Iterable[str] | None = None, *, date_from=None, date_to=None, where=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, where)
    keys = _group_keys(dimensions)
    g = x.groupby(keys + ["event_date"],
                  dropna=False)["user_id"].nunique().reset_index(name="DAU")
    return g.sort_values(keys + ["event_date"])


def mau(df: pd.DataFrame, dimensions: Iterable[str] | None = None, *, date_from=None, date_to=None, where=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, where)
    keys = _group_keys(dimensions)
    g = x.groupby(keys + ["event_month"],
                  dropna=False)["user_id"].nunique().reset_index(name="MAU")
    return g.sort_values(keys + ["event_month"])


def sticky_factor(dau_df: pd.DataFrame, mau_df: pd.DataFrame, dimensions: Iterable[str] | None = None) -> pd.DataFrame:
    """Sticky factor = средний DAU в месяце / MAU за этот месяц."""
    keys = _group_keys(dimensions)
    d = dau_df.copy()
    dt = pd.to_datetime(d["event_date"], errors="coerce")
    d["event_month"] = dt.dt.to_period("M").dt.to_timestamp()
    dau_m = (d.groupby(keys + ["event_month"], dropna=False)["DAU"]
             .mean()
             .reset_index(name="avg_DAU"))
    merged = dau_m.merge(mau_df, on=keys + ["event_month"], how="inner")
    merged["sticky_factor"] = merged["avg_DAU"] / \
        merged["MAU"].replace(0, np.nan)
    return merged[keys + ["event_month", "avg_DAU", "MAU", "sticky_factor"]]


def new_users(df: pd.DataFrame, dimensions: Iterable[str] | None = None, *, date_from=None, date_to=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, None)
    keys = _group_keys(dimensions)
    first_seen = x.groupby(
        keys + ["user_id"], dropna=False)["event_time"].min().reset_index()
    first_seen["first_date"] = _normalize_series_to_naive(
        first_seen["event_time"]).dt.date
    res = first_seen.groupby(
        keys + ["first_date"], dropna=False)["user_id"].nunique().reset_index(name="new_users")
    return res.sort_values(keys + ["first_date"])


def feature_usage_share(df: pd.DataFrame, feature_name: str, dimensions: Iterable[str] | None = None, *, date_from=None, date_to=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, None)
    keys = _group_keys(dimensions)
    daily = (
        x.groupby(keys + ["event_date"], dropna=False)
         .agg(users=("user_id", "nunique"),
              feature_users=("event_name", lambda s: x.loc[s.index, "user_id"][s == feature_name].nunique()))
         .reset_index()
    )
    daily["feature_usage_share"] = daily["feature_users"] / \
        daily["users"].replace(0, np.nan)
    return daily.sort_values(keys + ["event_date"])


def avg_target_actions_per_user(df: pd.DataFrame, action_name: str, dimensions: Iterable[str] | None = None, *, date_from=None, date_to=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, None)
    keys = _group_keys(dimensions)
    acts = x.assign(is_target=(x["event_name"] == action_name).astype(int))
    agg = acts.groupby(keys + ["event_date"], dropna=False).agg(
        users=("user_id", "nunique"),
        target_actions=("is_target", "sum"),
    ).reset_index()
    agg["avg_target_actions_per_user"] = agg["target_actions"] / \
        agg["users"].replace(0, np.nan)
    return agg.sort_values(keys + ["event_date"])


def retention_by_cohort(df: pd.DataFrame, freq: str = "D", horizon: int = 14) -> pd.DataFrame:
    x = df.copy()
    x["first_date"] = _normalize_series_to_naive(
        x["event_time"]).dt.floor(freq)
    x["period"] = _normalize_series_to_naive(x["event_time"]).dt.floor(freq)
    base = (x.drop_duplicates(["user_id", "first_date"])
            .groupby("first_date")["user_id"].nunique()
            .rename("cohort_size")
            .to_frame())
    visits = (x.drop_duplicates(["user_id", "period"])
              .groupby(["first_date", "period"])["user_id"].nunique()
              .to_frame("actives"))
    table = visits.join(base, on="first_date")
    denom = {"D": 1, "W": 7}.get(freq, 1)
    table["period_index"] = ((table.index.get_level_values("period") -
                              table.index.get_level_values("first_date")) /
                             np.timedelta64(1, "D")).astype(int) // denom
    table = table[table["period_index"].between(0, horizon)]
    table["retention"] = table["actives"] / \
        table["cohort_size"].replace(0, np.nan)
    pivot = (table.reset_index()
                  .pivot(index="first_date", columns="period_index", values="retention")
                  .fillna(0.0))
    pivot.columns = [f"{freq}{i}" for i in pivot.columns]
    return pivot.reset_index().sort_values("first_date")


def nps(df: pd.DataFrame, score_col: str = "nps_score") -> pd.DataFrame:
    if score_col not in df.columns:
        raise ValueError(f"Ожидаю колонку {score_col} (0..10) для NPS")
    s = pd.to_numeric(df[score_col], errors="coerce").dropna()
    if len(s) == 0:
        return pd.DataFrame({"NPS": [np.nan]})
    promoters = (s >= 9).mean()
    detractors = (s <= 6).mean()
    return pd.DataFrame({"NPS": [(promoters - detractors) * 100.0]})


def conversion_rate(df: pd.DataFrame, action_name: str, *, date_from=None, date_to=None) -> float:
    x = _apply_filters(df, date_from, date_to, None)
    users_all = x["user_id"].nunique()
    users_action = x.loc[x["event_name"] == action_name, "user_id"].nunique()
    return float(users_action / users_all) if users_all else np.nan


def purchase_conversion(df: pd.DataFrame, purchase_name: str = "purchase", **kwargs) -> float:
    return conversion_rate(df, purchase_name, **kwargs)


def arpu(df: pd.DataFrame, *, date_from=None, date_to=None) -> float:
    x = _apply_filters(df, date_from, date_to, None)
    rev = x["revenue"].sum() if "revenue" in x.columns else 0.0
    users = x["user_id"].nunique()
    return float(rev / users) if users else np.nan


def arp_dau(df: pd.DataFrame, *, date_from=None, date_to=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, None)
    daily = x.groupby("event_date", dropna=False).agg(
        revenue=("revenue", "sum") if "revenue" in x.columns else (
            "user_id", "size"),
        dau=("user_id", "nunique"),
    ).reset_index()
    daily["ARPDAU"] = daily["revenue"] / daily["dau"].replace(0, np.nan)
    return daily[["event_date", "revenue", "dau", "ARPDAU"]]


def arpp_dau(df: pd.DataFrame, *, date_from=None, date_to=None) -> pd.DataFrame:
    x = _apply_filters(df, date_from, date_to, None)
    daily = x.groupby("event_date", dropna=False).agg(
        revenue=("revenue", "sum") if "revenue" in x.columns else (
            "user_id", "size"),
        paying_dau=("is_paying", "sum") if "is_paying" in x.columns else (
            "user_id", "size"),
    ).reset_index()
    daily["ARPPDAU"] = daily["revenue"] / \
        daily["paying_dau"].replace(0, np.nan)
    return daily[["event_date", "revenue", "paying_dau", "ARPPDAU"]]


def per_user_ltv(df: pd.DataFrame) -> pd.DataFrame:
    if "revenue" not in df.columns:
        return pd.DataFrame(columns=["user_id", "ltv"])
    return df.groupby("user_id", as_index=False)["revenue"].sum().rename(columns={"revenue": "ltv"})


def mean_ltv(df: pd.DataFrame) -> float:
    p = per_user_ltv(df)
    return float(p["ltv"].mean()) if len(p) else np.nan


def avg_session_duration(df: pd.DataFrame) -> float:
    if {"session_start", "session_end"}.issubset(df.columns):
        x = df.dropna(subset=["session_start", "session_end"]).copy()
        x["session_start"] = pd.to_datetime(
            x["session_start"], errors="coerce")
        x["session_end"] = pd.to_datetime(x["session_end"], errors="coerce")
        dur = (x["session_end"] - x["session_start"]).dt.total_seconds() / 60.0
        return float(dur.mean()) if len(dur) else np.nan
    if "session_id" in df.columns:
        sess = df.groupby("session_id")["event_time"].agg(["min", "max"])
        smin = sess["min"]
        smax = sess["max"]
        try:
            smin = smin.dt.tz_convert(None)
            smax = smax.dt.tz_convert(None)
        except Exception:
            pass
        dur = (smax - smin).dt.total_seconds() / 60.0
        return float(dur.mean()) if len(dur) else np.nan
    return np.nan


def avg_sessions_per_user(df: pd.DataFrame) -> float:
    if "session_id" not in df.columns:
        return np.nan
    per_user = df.groupby("user_id")["session_id"].nunique()
    return float(per_user.mean()) if len(per_user) else np.nan


def avg_errors_per_session(df: pd.DataFrame, error_event_name: str = "error") -> float:
    if "session_id" not in df.columns:
        return np.nan
    sess_events = df.groupby("session_id").apply(lambda g: (
        g["event_name"].str.lower() == error_event_name).sum())
    return float(sess_events.mean()) if len(sess_events) else np.nan
