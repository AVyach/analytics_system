import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analytics.metrics import REGISTRY, compute_metrics
from analytics.metric_info import COLUMN_METRIC_INFO, PRODUCT_METRIC_INFO
from analytics.utils import read_csv_robust, corr_figure
from analytics.product_metrics import (
    prep_events, dau, mau, sticky_factor, new_users, feature_usage_share,
    avg_target_actions_per_user, retention_by_cohort, nps, conversion_rate,
    purchase_conversion, arpu, arp_dau, arpp_dau, per_user_ltv, mean_ltv,
    avg_session_duration, avg_sessions_per_user, avg_errors_per_session
)

st.set_page_config(
    page_title="Analytics – CSV Metrics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analytics: метрики по CSV")
st.caption(
    "Загрузите CSV, сопоставьте колонки и считайте как базовые, так и продуктовые метрики.")

# ---------- демо-датасет ----------


def make_demo_df(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=rows, freq="6H", tz="UTC")
    return pd.DataFrame({
        "user_id": rng.integers(1, 180, size=rows),
        "event_time": ts,
        "event_name": rng.choice(["open", "feature_use", "purchase", "error"], size=rows, p=[.5, .3, .15, .05]),
        "revenue": rng.choice([0, 0, 0, 1.99, 4.99, 9.99], size=rows),
        "device_type": rng.choice(["ios", "android", "web"], size=rows),
        "channel": rng.choice(["ads", "organic", "email"], size=rows),
        "feature": rng.choice(["A", "B", "C"], size=rows),
        "session_id": rng.integers(1000, 3000, size=rows),
        "nps_score": rng.integers(0, 11, size=rows),
    })


# ---------- сайдбар ----------
with st.sidebar:
    st.header("Загрузка CSV")
    demo_clicked = st.button("Загрузить демо-датасет")
    file = st.file_uploader("CSV файл", type=["csv"])
    decimal = st.selectbox("Десятичный разделитель", [".", ","], index=0)
    encoding_hint = st.text_input("Кодировка (опционально)", value="")
    show_corr = st.checkbox("Матрица корреляций", value=True)
    show_hist = st.checkbox("Гистограммы", value=False)
    bins = st.slider("Бины", 5, 100, 30) if show_hist else None

with st.sidebar.expander("Справка: Column metrics"):
    for k, v in COLUMN_METRIC_INFO.items():
        st.markdown(f"- **{k}** — {v}")
with st.sidebar.expander("Справка: Product metrics"):
    for k, v in PRODUCT_METRIC_INFO.items():
        st.markdown(f"- **{k}** — {v}")

# ---------- выбор источника данных ----------
df = None
if 'use_demo' not in st.session_state:
    st.session_state.use_demo = False
if demo_clicked:
    st.session_state.use_demo = True

if st.session_state.use_demo:
    df = make_demo_df()
elif file is not None:
    raw = file.read()
    df = read_csv_robust(raw, encoding=encoding_hint or None)

tab_stats, tab_product = st.tabs(["Column stats", "Product metrics"])

# ---------- заглушка без данных ----------
if df is None:
    with tab_stats:
        st.info(
            "Здесь появится интерфейс после загрузки данных в левом сайдбаре. "
            "Нажмите **«Загрузить демо-датасет»** или выберите свой CSV."
        )
    st.stop()

# ---------- базовая подготовка и предпросмотр ----------
if decimal == ",":
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col].astype(
                    str).str.replace(",", ".", regex=False))
            except Exception:
                pass

st.success(
    f"Загружено: {len(df):,} строк × {df.shape[1]} столбцов".replace(",", " "))
st.dataframe(df.head(20), use_container_width=True)

# ---------- сопоставление колонок ----------


def _guess(colnames, candidates):
    s = {c.lower() for c in colnames}
    for c in candidates:
        if c in s:
            return [n for n in colnames if n.lower() == c][0]
    return None


names = list(df.columns)

guess_user = _guess(names, ["user_id", "userid",
                    "user", "uid", "client_id", "account_id"])
guess_time = _guess(names, ["event_time", "timestamp",
                    "time", "ts", "datetime", "date"])
guess_event = _guess(
    names, ["event_name", "event", "action", "type", "eventtype"])
guess_rev = _guess(names, ["revenue", "amount", "value", "price", "sum"])
guess_sess = _guess(names, ["session_id", "session", "sid"])
guess_dev = _guess(names, ["device_type", "device", "platform", "os"])
guess_ch = _guess(names, ["channel", "source", "utm_source", "traffic_source"])
guess_feat = _guess(names, ["feature", "feature_name", "flag"])
guess_nps = _guess(names, ["nps", "nps_score", "npsscore", "score"])

with st.sidebar.expander("Сопоставление колонок (для Product metrics)", expanded=True):
    def pick(label, default=None, required=False):
        opts = ["<нет>"] + names
        idx = (opts.index(default) if default in opts else 0)
        val = st.selectbox(label, options=opts, index=idx)
        if required and val == "<нет>":
            st.warning(f"Требуется указать колонку: {label}")
        return None if val == "<нет>" else val

    user_col = pick("user_id (обязательно)", guess_user, required=True)
    time_col = pick("event_time (обязательно)", guess_time, required=True)
    name_col = pick("event_name (обязательно)", guess_event, required=True)
    rev_col = pick("revenue (опционально)", guess_rev)
    sess_col = pick("session_id (опционально)", guess_sess)
    dev_col = pick("device_type (опционально)", guess_dev)
    ch_col = pick("channel (опционально)", guess_ch)
    feat_col = pick("feature (опционально)", guess_feat)
    nps_col = pick("nps_score (опционально)", guess_nps)
    tz_mode = st.selectbox("Часовой пояс времени", [
                           "Auto (не трогать)", "Считать как UTC", "Сделать naive (без TZ)"], index=0)

# Сконструируем события
ev = None
if user_col and time_col and name_col:
    ev = pd.DataFrame({
        "user_id":  df[user_col],
        "event_time": df[time_col],
        "event_name": df[name_col],
    })
    opt_map: Dict[str, Optional[str]] = {
        "revenue": rev_col, "session_id": sess_col, "device_type": dev_col,
        "channel": ch_col, "feature": feat_col, "nps_score": nps_col
    }
    for k, src in opt_map.items():
        if src:
            ev[k] = df[src]

    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    if tz_mode == "Считать как UTC":
        if ev["event_time"].dt.tz is None:
            ev["event_time"] = ev["event_time"].dt.tz_localize("UTC")
        else:
            ev["event_time"] = ev["event_time"].dt.tz_convert("UTC")
    elif tz_mode == "Сделать naive (без TZ)":
        try:
            ev["event_time"] = ev["event_time"].dt.tz_convert(None)
        except Exception:
            pass

# --- безопасное сравнение дат ---


def _series_to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s


def _ts_to_naive(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts)
    try:
        return t.tz_localize(None) if t.tzinfo is not None else t
    except Exception:
        return t


# ===== Column stats =====
with tab_stats:
    st.subheader("Выбор столбцов и метрик")
    left, right = st.columns([2, 3])
    with left:
        selected_cols: List[str] = st.multiselect(
            "Столбцы",
            options=list(df.columns),
            default=list(df.select_dtypes(
                include=[np.number]).columns) or list(df.columns)[:3],
        )
    with right:
        selected_metrics: List[str] = st.multiselect(
            "Метрики",
            options=list(REGISTRY.keys()),
            default=["mean", "median", "std", "min", "q25",
                     "q50", "q75", "max", "missing_pct", "nunique"],
        )

    if selected_cols and selected_metrics:
        rows = []
        for col in selected_cols:
            metrics = compute_metrics(df[col], selected_metrics)
            for k, v in metrics.items():
                rows.append({"column": col, "metric": k, "value": v})
        res = pd.DataFrame(rows)
        pivot = res.pivot(index="column", columns="metric",
                          values="value").reset_index()

        st.subheader("Результаты")
        st.dataframe(pivot, use_container_width=True)
        st.download_button(
            "Скачать метрики (CSV)",
            data=pivot.to_csv(index=False).encode("utf-8"),
            file_name="metrics.csv",
            mime="text/csv",
        )

        st.subheader("Частоты значений (Top-10)")
        tabs_freq = st.tabs(selected_cols)
        for tab, col in zip(tabs_freq, selected_cols):
            with tab:
                vc = df[col].astype("string").value_counts(
                    dropna=False).head(10)
                freq = vc.rename("count").to_frame(
                ).reset_index(names=["value"])
                st.dataframe(freq, use_container_width=True)

        num_cols = df.select_dtypes(include=[np.number]).columns
        if show_corr and len(num_cols) >= 2:
            st.subheader("Корреляции (числовые)")
            fig = corr_figure(df)
            st.pyplot(fig)

        if show_hist:
            st.subheader("Гистограммы")
            for col in selected_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig, ax = plt.subplots()
                    ax.hist(df[col].dropna(), bins=bins)
                    ax.set_title(col)
                    st.pyplot(fig)
    else:
        st.warning("Выберите хотя бы один столбец и одну метрику.")

# ===== Product metrics =====
with tab_product:
    if ev is None:
        st.info("Для продуктовых метрик укажите в сайдбаре колонки: user_id, event_time, event_name. Остальные — опционально.")
    else:
        try:
            ev = prep_events(ev, tz=None)
        except Exception as e:
            st.error(f"Не удалось подготовить события: {e}")
            st.stop()

        st.subheader("Параметры расчёта")
        c1, c2, c3 = st.columns(3)
        with c1:
            date_from = st.date_input(
                "От даты", value=ev["event_time"].min().date())
        with c2:
            date_to = st.date_input(
                "До даты (не включительно)", value=ev["event_time"].max().date())
        with c3:
            dims = st.multiselect(
                "Разрезы", ["device_type", "channel", "feature"], default=[])

        st.markdown("Выберите метрики")
        m1, m2, m3 = st.columns(3)
        with m1:
            use_dau = st.checkbox("DAU", True)
            use_mau = st.checkbox("MAU", True)
            use_sticky = st.checkbox("Sticky factor", True)
            use_new = st.checkbox("New users", True)
            use_feat = st.checkbox("Feature usage share", False)
        with m2:
            use_avg_target = st.checkbox("Avg target actions / user", False)
            use_ret = st.checkbox("Retention (cohort)", True)
            use_nps = st.checkbox("NPS", False)
            use_conv = st.checkbox("Conversion to action", False)
            use_pconv = st.checkbox("Conversion to purchase", False)
        with m3:
            use_arpu = st.checkbox("ARPU", True)
            use_arpdau = st.checkbox("ARPDAU", True)
            use_arppdau = st.checkbox("ARPPDAU", False)
            use_ltv = st.checkbox("LTV (per user / mean)", True)
            use_sessions = st.checkbox(
                "Sessions (avg duration / per user / errors)", False)

        st.divider()
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            feature_name = st.text_input(
                "feature_name / action_name", value="feature_use")
        with pcol2:
            purchase_name = st.text_input(
                "purchase event_name", value="purchase")
        with pcol3:
            horizon = st.number_input(
                "Retention horizon (days)", min_value=7, max_value=60, value=14)

        date_from_ts = pd.to_datetime(date_from)
        date_to_ts = pd.to_datetime(date_to)

        if use_dau:
            st.subheader("DAU")
            st.dataframe(dau(ev, dims, date_from=date_from_ts,
                         date_to=date_to_ts), use_container_width=True)
        if use_mau:
            st.subheader("MAU")
            mau_df = mau(ev, dims, date_from=date_from_ts, date_to=date_to_ts)
            st.dataframe(mau_df, use_container_width=True)
        else:
            mau_df = None
        if use_sticky:
            st.subheader("Sticky factor (avg_DAU / MAU)")
            dau_df = dau(ev, dims, date_from=date_from_ts, date_to=date_to_ts)
            mau_df2 = mau_df if mau_df is not None else mau(
                ev, dims, date_from=date_from_ts, date_to=date_to_ts)
            st.dataframe(sticky_factor(dau_df, mau_df2, dims),
                         use_container_width=True)
        if use_new:
            st.subheader("New users")
            st.dataframe(new_users(ev, dims, date_from=date_from_ts,
                         date_to=date_to_ts), use_container_width=True)
        if use_feat:
            st.subheader("Feature usage share")
            st.dataframe(feature_usage_share(ev, feature_name, dims,
                         date_from=date_from_ts, date_to=date_to_ts), use_container_width=True)
        if use_avg_target:
            st.subheader("Avg target actions per user")
            st.dataframe(avg_target_actions_per_user(ev, feature_name, dims,
                         date_from=date_from_ts, date_to=date_to_ts), use_container_width=True)
        if use_ret:
            st.subheader("Retention (cohort by first activity)")
            ev_time_naive = _series_to_naive(ev["event_time"])
            t_from = _ts_to_naive(date_from_ts)
            t_to = _ts_to_naive(date_to_ts)
            filt = (ev_time_naive >= t_from) & (ev_time_naive < t_to)
            st.dataframe(retention_by_cohort(
                ev[filt], freq="D", horizon=int(horizon)), use_container_width=True)
        if use_nps and "nps_score" in ev.columns:
            st.subheader("NPS")
            ev_time_naive = _series_to_naive(ev["event_time"])
            t_from = _ts_to_naive(date_from_ts)
            t_to = _ts_to_naive(date_to_ts)
            filt = (ev_time_naive >= t_from) & (ev_time_naive < t_to)
            st.dataframe(nps(ev[filt]), use_container_width=True)
        if use_conv:
            st.subheader("Conversion to action")
            st.write(conversion_rate(ev, feature_name,
                     date_from=date_from_ts, date_to=date_to_ts))
        if use_pconv:
            st.subheader("Conversion to purchase")
            st.write(purchase_conversion(ev, purchase_name,
                     date_from=date_from_ts, date_to=date_to_ts))
        if use_arpu:
            st.subheader("ARPU")
            st.write(arpu(ev, date_from=date_from_ts, date_to=date_to_ts))
        if use_arpdau:
            st.subheader("ARPDAU")
            st.dataframe(arp_dau(ev, date_from=date_from_ts,
                         date_to=date_to_ts), use_container_width=True)
        if use_arppdau:
            st.subheader("ARPPDAU")
            st.dataframe(arpp_dau(ev, date_from=date_from_ts,
                         date_to=date_to_ts), use_container_width=True)
        if use_ltv:
            st.subheader("LTV (per user)")
            st.dataframe(per_user_ltv(ev), use_container_width=True)
            st.markdown("Mean LTV:")
            st.write(mean_ltv(ev))
        if use_sessions:
            st.subheader("Sessions")
            st.write("Average session duration (min): ",
                     avg_session_duration(ev))
            st.write("Average sessions per user: ", avg_sessions_per_user(ev))
            st.write("Average errors per session: ",
                     avg_errors_per_session(ev))
