"""Interactive EVM dashboard (Streamlit).

Reuses the canonical calculations from evm_analysis.py so the script,
notebook, and dashboard stay consistent. This file only adds UI;
it does not change existing outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evm_analysis import (
    FEATURES,
    add_evm_metrics,
    add_lag_features,
    build_dataset,
    evm_forecasts,
)

DEFAULT_SPLIT = 8
DEFAULT_N_EST = 300
DEFAULT_MAX_DEPTH = 4


def compute_full_table(base: pd.DataFrame) -> pd.DataFrame:
    """Cumulative EVM table from per-sprint PV/EV/AC."""
    df = base.copy()
    for c in ["PV", "EV", "AC"]:
        df[f"cum_{c}"] = df[c].cumsum()
    df = add_evm_metrics(df)
    df = add_lag_features(df)
    return df


def train_predict(df: pd.DataFrame, split: int, n_est: int, max_depth: int):
    """Same logic as evm_analysis.train_and_predict, with adjustable params."""
    train = df[df["sprint_id"] <= split].copy()
    test = df[df["sprint_id"] > split].copy()
    model = RandomForestRegressor(
        n_estimators=n_est, max_depth=max_depth, random_state=9
    )
    model.fit(train[FEATURES], train[["CPI", "SPI"]])
    pred = model.predict(test[FEATURES])
    test = test.copy()
    test["CPI_pred"] = pred[:, 0]
    test["SPI_pred"] = pred[:, 1]
    cpi8 = train.iloc[-1]["CPI"]
    spi8 = train.iloc[-1]["SPI"]
    test["CPI_trad"] = cpi8
    test["SPI_trad"] = spi8
    metrics = {
        "MAE_CPI_trad": mean_absolute_error(test["CPI"], test["CPI_trad"]),
        "MAE_SPI_trad": mean_absolute_error(test["SPI"], test["SPI_trad"]),
        "MAE_CPI_rf": mean_absolute_error(test["CPI"], test["CPI_pred"]),
        "MAE_SPI_rf": mean_absolute_error(test["SPI"], test["SPI_pred"]),
        "RMSE_CPI_trad": float(np.sqrt(mean_squared_error(test["CPI"], test["CPI_trad"]))),
        "RMSE_SPI_trad": float(np.sqrt(mean_squared_error(test["SPI"], test["SPI_trad"]))),
        "RMSE_CPI_rf": float(np.sqrt(mean_squared_error(test["CPI"], test["CPI_pred"]))),
        "RMSE_SPI_rf": float(np.sqrt(mean_squared_error(test["SPI"], test["SPI_pred"]))),
    }
    return train, test, metrics


st.set_page_config(page_title="EVM AI Forecasting Dashboard", layout="wide")
st.title("EVM with AI-Enhanced Forecasting — Interactive Dashboard")
st.caption(
    "Same dataset and formulas as evm_analysis.py. Edit sprints, move the "
    "checkpoint, retrain the Random Forest, and compare against flat EVM."
)

with st.sidebar:
    st.header("Forecast settings")
    split = st.slider("Checkpoint sprint (train 1–N)", 4, 13, DEFAULT_SPLIT)
    n_est = st.slider("RF n_estimators", 50, 500, DEFAULT_N_EST, step=50)
    max_depth = st.slider("RF max_depth", 2, 8, DEFAULT_MAX_DEPTH)
    st.divider()
    st.caption(
        "Baseline: flat CPI/SPI held at checkpoint values. "
        "RF features: sprint_id, cum_PV/EV/AC, CV, SV, CPI/SPI lag-1."
    )

base_default = build_dataset()[["sprint_id", "PV", "EV", "AC"]]
st.subheader("Per-sprint inputs ($m)")
edited = st.data_editor(
    base_default,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "sprint_id": st.column_config.NumberColumn(disabled=True),
        "PV": st.column_config.NumberColumn(min_value=0.0, format="%.3f"),
        "EV": st.column_config.NumberColumn(min_value=0.0, format="%.3f"),
        "AC": st.column_config.NumberColumn(min_value=0.0, format="%.3f"),
    },
)

df = compute_full_table(edited)
fc = evm_forecasts(df, at_sprint=int(split))
train, test, m = train_predict(df, int(split), int(n_est), int(max_depth))

c1, c2, c3, c4 = st.columns(4)
c1.metric("BAC (final PV)", f"${fc['BAC']:.3f}m")
c2.metric(f"CPI @ sprint {split}", f"{fc['CPI']:.3f}")
c3.metric(f"SPI @ sprint {split}", f"{fc['SPI']:.3f}")
c4.metric("Actual final AC", f"${df['cum_AC'].iloc[-1]:.3f}m")

c1, c2 = st.columns(2)
c1.metric("EAC typical (BAC/CPI)", f"${fc['EAC_cpi']:.3f}m")
c2.metric("EAC atypical (AC+BAC−EV)", f"${fc['EAC_ac']:.3f}m")

st.subheader("S-curves and performance")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(df["sprint_id"], df["cum_PV"], marker="o", label="PV")
ax[0].plot(df["sprint_id"], df["cum_EV"], marker="s", label="EV")
ax[0].plot(df["sprint_id"], df["cum_AC"], marker="^", label="AC")
ax[0].set_title("S-Curve (cumulative $m)")
ax[0].set_xlabel("Sprint")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(df["sprint_id"], df["CPI"], marker="o", label="CPI")
ax[1].plot(df["sprint_id"], df["SPI"], marker="s", label="SPI")
ax[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax[1].axvline(split + 0.5, color="gray", linestyle=":", linewidth=1)
ax[1].set_title("CPI / SPI (1.0 = on track)")
ax[1].set_xlabel("Sprint")
ax[1].legend()
ax[1].grid(alpha=0.3)

x = df["sprint_id"].to_numpy()
ax[2].bar(x - 0.2, df["CV"], width=0.4, label="CV")
ax[2].bar(x + 0.2, df["SV"], width=0.4, label="SV")
ax[2].axhline(0, color="black", linewidth=0.8)
ax[2].set_title("Variances (cumulative $m)")
ax[2].set_xlabel("Sprint")
ax[2].legend()
ax[2].grid(alpha=0.3, axis="y")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader(f"Forecast: train 1–{split} → predict {split + 1}–15")
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
for i, (col, pred, trad, title) in enumerate(
    [
        ("CPI", "CPI_pred", "CPI_trad", "Cost Performance Index (CPI)"),
        ("SPI", "SPI_pred", "SPI_trad", "Schedule Performance Index (SPI)"),
    ]
):
    a = ax2[i]
    a.plot(df["sprint_id"], df[col], "k-o", label="Actual", linewidth=2)
    a.plot(test["sprint_id"], test[trad], "r--s", label="Traditional (flat)")
    a.plot(test["sprint_id"], test[pred], "b-^", label="RF predicted")
    a.axvline(split + 0.5, color="gray", linestyle=":")
    a.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    a.set_title(title)
    a.set_xlabel("Sprint")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.write(
    f"RF vs traditional — CPI MAE {m['MAE_CPI_rf']:.4f} vs {m['MAE_CPI_trad']:.4f} | "
    f"SPI MAE {m['MAE_SPI_rf']:.4f} vs {m['MAE_SPI_trad']:.4f} | "
    f"CPI RMSE {m['RMSE_CPI_rf']:.4f} vs {m['RMSE_CPI_trad']:.4f} | "
    f"SPI RMSE {m['RMSE_SPI_rf']:.4f} vs {m['RMSE_SPI_trad']:.4f}"
)

st.subheader("Tables")
st.dataframe(df.round(4), use_container_width=True)
st.dataframe(
    test[["sprint_id", "CPI", "CPI_trad", "CPI_pred", "SPI", "SPI_trad", "SPI_pred"]].round(4),
    use_container_width=True,
)
st.download_button(
    "Download EVM table (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="evm_table.csv",
    mime="text/csv",
)
