from pathlib import Path

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

# ── Design tokens ────────────────────────────────────────────────────────────
INK = "#1E293B"       # text, actuals
MUTED = "#64748B"     # secondary text, baseline
SOFT = "#94A3B8"      # tertiary series
HAIR = "#E2E8F0"      # hairlines, grids
ACCENT = "#2F5BEA"    # the one accent: AI / model output
GOOD = "#1A7F4B"
BAD = "#B42318"


# ── Core logic (unchanged) ───────────────────────────────────────────────────
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


# ── Streamlit version compatibility (use_container_width → width="stretch") ──
def edit_df(data, **kw):
    try:
        return st.data_editor(data, width="stretch", **kw)
    except Exception:
        return st.data_editor(data, use_container_width=True, **kw)


# ── Theme bootstrap ──────────────────────────────────────────────────────────
def ensure_light_theme():
    """Streamlit follows the OS light/dark setting unless a theme is configured.
    This page is designed for light, so write a minimal config on first run.
    It takes effect after the next restart of `streamlit run`."""
    try:
        cfg = Path.cwd() / ".streamlit" / "config.toml"
        if not cfg.exists():
            cfg.parent.mkdir(exist_ok=True)
            cfg.write_text(
                '[theme]\nbase = "light"\nprimaryColor = "#2F5BEA"\n'
                'backgroundColor = "#FFFFFF"\nsecondaryBackgroundColor = "#F7F8FA"\n'
                'textColor = "#1E293B"\n\n[client]\ntoolbarMode = "minimal"\n'
            )
    except Exception:
        pass


# ── Styling ──────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {{
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: #FFFFFF; color: {INK}; }}

/* keep widget text readable even if the OS is in dark mode */
[data-testid="stWidgetLabel"] p, [data-testid="stExpander"] summary p,
[data-testid="stSliderThumbValue"], [data-testid="stSlider"] label {{ color: {INK}; }}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: {MUTED}; }}

#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}

.block-container {{ max-width: 1080px; padding-top: 3rem; padding-bottom: 5rem; }}

/* masthead */
.mast {{ display: flex; justify-content: space-between; align-items: baseline; }}
.mast-t {{ font-size: 1rem; font-weight: 600; letter-spacing: -0.005em; }}
.mast-n {{ color: {MUTED}; font-size: 0.85rem; }}

/* hero statement */
.hero-t {{ font-size: 2.35rem; font-weight: 500; letter-spacing: -0.028em; line-height: 1.18;
           max-width: 24ch; margin: 3.25rem 0 0 0; }}
.hero-s {{ color: {MUTED}; font-size: 1.02rem; line-height: 1.6; max-width: 62ch; margin: 1.1rem 0 0 0; }}
.hero-s b {{ color: {INK}; font-weight: 500; font-variant-numeric: tabular-nums; }}

.rule {{ border-top: 1px solid {HAIR}; margin: 2.25rem 0 0.9rem 0; }}

/* section heads */
.sec {{ border-top: 1px solid {HAIR}; padding-top: 1.1rem; margin: 3.25rem 0 1.4rem 0; }}
.sec-t {{ font-size: 1.3rem; font-weight: 500; letter-spacing: -0.015em; }}
.sec-n {{ color: {MUTED}; font-size: 0.88rem; margin-top: 0.3rem; max-width: 70ch; line-height: 1.5; }}
.sub-t {{ font-size: 0.9rem; font-weight: 500; margin: 1.75rem 0 0.6rem 0; }}

/* ledger */
.lg {{ color: {MUTED}; font-size: 0.8rem; margin: 1.15rem 0 0.2rem 0; }}
.lg:first-child {{ margin-top: 0; }}
.lr {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
       padding: 0.6rem 0; border-bottom: 1px solid {HAIR}; font-size: 0.9rem; }}
.lr .k {{ color: {MUTED}; }}
.lr .v {{ font-weight: 500; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.lr .v.good {{ color: {GOOD}; }}
.lr .v.bad {{ color: {BAD}; }}

/* verdict */
.verdict {{ font-size: 1rem; margin: 0 0 1.1rem 0; }}
.verdict b {{ font-weight: 600; color: {ACCENT}; }}

/* tables (custom HTML so they match the page in any OS theme) */
.tbl-wrap {{ overflow-x: auto; }}
.tbl {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; color: {INK}; }}
.tbl th {{ text-align: right; font-weight: 500; color: {MUTED}; padding: 0.55rem 0.9rem;
           border-bottom: 1px solid {HAIR}; white-space: nowrap; }}
.tbl td {{ text-align: right; padding: 0.6rem 0.9rem; border-bottom: 1px solid #EEF2F6;
           font-variant-numeric: tabular-nums; white-space: nowrap; }}
.tbl th:first-child, .tbl td:first-child {{ text-align: left; }}
.tbl th.grp {{ text-align: center; color: {INK}; font-weight: 600; border-bottom: 1px solid {HAIR}; }}
.tbl .sep {{ border-left: 1px solid {HAIR}; }}
.tbl td.best {{ color: {ACCENT}; font-weight: 600; }}
.tbl tbody tr:last-child td {{ border-bottom: 0; }}

/* widgets */
[data-testid="stExpander"] details {{ border: 1px solid {HAIR}; border-radius: 6px; background: #FFFFFF; }}
[data-testid="stExpander"] summary {{ padding: 0.65rem 0.9rem; }}
[data-testid="stExpander"] summary p {{ font-size: 0.92rem; font-weight: 500; }}
.stDownloadButton button {{
    background: #FFFFFF; color: {INK}; border: 1px solid {INK};
    border-radius: 6px; font-weight: 500; padding: 0.45rem 1rem; margin-top: 0.75rem;
}}
.stDownloadButton button:hover {{ background: {INK}; color: #FFFFFF; border-color: {INK}; }}
[data-testid="stDataEditor"] {{ border: 1px solid {HAIR}; border-radius: 6px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def section(title: str, note: str = ""):
    note_html = f'<div class="sec-n">{note}</div>' if note else ""
    st.markdown(
        f'<div class="sec"><div class="sec-t">{title}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def ledger(groups):
    """groups: list of (group_label, [(key, value, tone), ...]). `$` is passed pre-escaped."""
    html = ""
    for label, rows in groups:
        html += f'<div class="lg">{label}</div>'
        for k, v, tone in rows:
            html += f'<div class="lr"><span class="k">{k}</span><span class="v {tone}">{v}</span></div>'
    st.markdown(html, unsafe_allow_html=True)


def money(v: float) -> str:
    return f"&#36;{v:.3f}m"


# ── HTML tables ──────────────────────────────────────────────────────────────
def html_table(headers, rows, group=None, seps=()):
    """headers: list[str]. rows: list[list[(text, css_class)]].
    group: optional list[(label, colspan)] shown above headers. seps: column indexes with a left rule."""
    h = ""
    if group:
        h += "<tr>" + "".join(f'<th class="grp sep" colspan="{n}">{lab}</th>' if lab else f'<th colspan="{n}"></th>'
                               for lab, n in group) + "</tr>"
    h += "<tr>" + "".join(f'<th class="{"sep" if i in seps else ""}">{t}</th>' for i, t in enumerate(headers)) + "</tr>"
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="{c} {"sep" if i in seps else ""}">{t}</td>' for i, (t, c) in enumerate(r)
        ) + "</tr>"
    st.markdown(
        f'<div class="tbl-wrap"><table class="tbl"><thead>{h}</thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def df_table(d: pd.DataFrame):
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if pd.isna(v):
                cells.append(("—", ""))
            elif isinstance(v, (int, np.integer)):
                cells.append((str(int(v)), ""))
            else:
                cells.append((f"{float(v):.4f}", ""))
        rows.append(cells)
    html_table([str(c) for c in d.columns], rows)


# ── Charts ───────────────────────────────────────────────────────────────────
def style_ax(a, title):
    a.set_facecolor("white")
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(HAIR)
    a.tick_params(colors=MUTED, labelsize=8.5, length=0, pad=6)
    a.grid(axis="y", color=HAIR, linewidth=0.7, alpha=0.9)
    a.set_axisbelow(True)
    a.set_xlabel("Sprint", color=MUTED, fontsize=9)
    a.set_title(title, loc="left", fontsize=10.5, fontweight="medium", color=INK, pad=14)


def end_labels(a, items, x, gap_frac=0.075):
    """Direct labels at the right edge of a line chart, nudged apart so they never overlap.
    items: list of (y, text, color)."""
    lo, hi = a.get_ylim()
    gap = (hi - lo) * gap_frac
    last = None
    for y, text, color in sorted(items, key=lambda t: t[0]):
        ty = y if last is None or y - last >= gap else last + gap
        last = ty
        a.text(x, ty, text, color=color, fontsize=8.5, fontweight="medium", va="center", ha="left")


def fig_one(w, h):
    fig, a = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    return fig, a


def finish(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def forecast_chart(df, test, split, last_sprint, col, pred, trad, title):
    fig, a = fig_one(6.2, 4.0)
    a.axvspan(split + 0.5, last_sprint + 0.5, color="#F1F5F9", zorder=0)
    a.fill_between(test["sprint_id"], test[col], test[trad], color=MUTED, alpha=0.10, linewidth=0)
    a.fill_between(test["sprint_id"], test[col], test[pred], color=ACCENT, alpha=0.14, linewidth=0)
    a.plot(df["sprint_id"], df[col], color=INK, linewidth=2, marker="o", markersize=3.5)
    a.plot(test["sprint_id"], test[trad], color=MUTED, linestyle="--", linewidth=1.6)
    a.plot(test["sprint_id"], test[pred], color=ACCENT, linewidth=2, marker="o", markersize=3.5)
    a.set_xlim(0.4, last_sprint + 4.3)
    a.set_xticks(range(1, last_sprint + 1, 2 if last_sprint > 12 else 1))
    # scale to the data (not the 1.0 line) and leave headroom for the "Forecast" tag
    vals = pd.concat([df[col], test[trad], test[pred]])
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 0.02)
    a.set_ylim(lo - 0.12 * span, hi + 0.30 * span)
    a.axhline(1.0, color=SOFT, linestyle="--", linewidth=1, alpha=0.6)
    style_ax(a, title)
    a.text(split + 0.7, 0.97, "Forecast", transform=a.get_xaxis_transform(),
           fontsize=8.5, color=MUTED, va="top")
    end_labels(
        a,
        [
            (df[col].iloc[-1], f"Actual {df[col].iloc[-1]:.3f}", INK),
            (test[trad].iloc[-1], f"Traditional {test[trad].iloc[-1]:.3f}", MUTED),
            (test[pred].iloc[-1], f"RF {test[pred].iloc[-1]:.3f}", ACCENT),
        ],
        x=last_sprint + 0.75,
    )
    finish(fig)


# ── Page ─────────────────────────────────────────────────────────────────────
ensure_light_theme()
st.set_page_config(
    page_title="EVM AI Forecasting",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

st.markdown(
    '<div class="mast"><span class="mast-t">EVM forecasting</span>'
    '<span class="mast-n">Traditional EVM vs Random Forest</span></div>',
    unsafe_allow_html=True,
)

# Visual order is fixed here; every block is filled in after the data is known.
hero_box = st.container()
controls_box = st.container()
status_box = st.container()
perf_box = st.container()
forecast_box = st.container()
data_box = st.container()

with controls_box:
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        split = st.slider("Checkpoint sprint (train 1–N)", 4, 13, DEFAULT_SPLIT)
    with c2:
        n_est = st.slider("RF n_estimators", 50, 500, DEFAULT_N_EST, step=50)
    with c3:
        max_depth = st.slider("RF max_depth", 2, 8, DEFAULT_MAX_DEPTH)

base_default = build_dataset()[["sprint_id", "PV", "EV", "AC"]]
with data_box:
    section("Sprint data", "Planned value, earned value and actual cost per sprint, in &#36;m. Edit any cell and everything above updates.")
    with st.expander("Edit inputs (PV, EV, AC)"):
        edited = edit_df(
            base_default,
            num_rows="fixed",
            key="sprint_inputs",
            column_config={
                "sprint_id": st.column_config.NumberColumn("Sprint", disabled=True),
                "PV": st.column_config.NumberColumn("PV", min_value=0.0, format="%.3f"),
                "EV": st.column_config.NumberColumn("EV", min_value=0.0, format="%.3f"),
                "AC": st.column_config.NumberColumn("AC", min_value=0.0, format="%.3f"),
            },
        )

if edited[["PV", "EV", "AC"]].isnull().any().any():
    st.warning("Some PV, EV or AC cells are empty. Fill them in under “Edit inputs” at the bottom of the page to continue.")
    st.stop()

df = compute_full_table(edited)
fc = evm_forecasts(df, at_sprint=int(split))
train, test, m = train_predict(df, int(split), int(n_est), int(max_depth))

last_sprint = int(df["sprint_id"].max())
ac_to_date = float(df.loc[df["sprint_id"] == int(split), "cum_AC"].iloc[0])
etc_cpi = fc["EAC_cpi"] - ac_to_date
etc_ac = fc["EAC_ac"] - ac_to_date
final_ac = float(df["cum_AC"].iloc[-1])

if fc["CPI"] > 1.0005:
    cost_word, cost_tone = "under budget", "good"
elif fc["CPI"] < 0.9995:
    cost_word, cost_tone = "over budget", "bad"
else:
    cost_word, cost_tone = "on budget", ""
if fc["SPI"] > 1.0005:
    sched_word, sched_tone = "ahead of schedule", "good"
elif fc["SPI"] < 0.9995:
    sched_word, sched_tone = "behind schedule", "bad"
else:
    sched_word, sched_tone = "on schedule", ""

# ── Hero: the answer first, in a sentence ────────────────────────────────────
pct = (fc["EAC_cpi"] / fc["BAC"] - 1) * 100
direction = "above" if pct > 0 else "below"
with hero_box:
    st.markdown(
        f'<div class="hero-t">At sprint {split}, the project is {cost_word} and {sched_word}.</div>'
        f'<div class="hero-s">Cost performance index is <b>{fc["CPI"]:.2f}</b> and schedule performance '
        f'index is <b>{fc["SPI"]:.2f}</b>. If current cost efficiency continues, the project finishes at '
        f'<b>{money(fc["EAC_cpi"])}</b>, {abs(pct):.1f}% {direction} the <b>{money(fc["BAC"])}</b> budget.</div>',
        unsafe_allow_html=True,
    )

# ── Status: S-curve with a ledger beside it ──────────────────────────────────
with status_box:
    section("Where the project stands", f"Cumulative planned value, earned value and actual cost, with the figures at sprint {split}.")
    left, right = st.columns([5, 3], gap="large")
    with left:
        fig, a = fig_one(7.4, 4.3)
        a.plot(df["sprint_id"], df["cum_PV"], color=SOFT, linewidth=1.8, linestyle="--")
        a.plot(df["sprint_id"], df["cum_EV"], color=INK, linewidth=1.8)
        a.plot(df["sprint_id"], df["cum_AC"], color=ACCENT, linewidth=1.8)
        a.axvline(split, color=SOFT, linestyle=":", linewidth=1)
        a.set_xlim(df["sprint_id"].min() - 0.3, last_sprint + 2.2)
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.08 * (hi - lo))
        style_ax(a, "S-curve (cumulative $m)")
        a.text(split + 0.15, 0.97, "Checkpoint", transform=a.get_xaxis_transform(),
               fontsize=8.5, color=MUTED, va="top")
        end_labels(
            a,
            [
                (df["cum_PV"].iloc[-1], "PV", MUTED),
                (df["cum_EV"].iloc[-1], "EV", INK),
                (df["cum_AC"].iloc[-1], "AC", ACCENT),
            ],
            x=last_sprint + 0.3,
        )
        finish(fig)
    with right:
        ledger(
            [
                ("At the checkpoint", [
                    ("Budget at completion", money(fc["BAC"]), ""),
                    ("Cost performance index", f"{fc['CPI']:.3f}", cost_tone),
                    ("Schedule performance index", f"{fc['SPI']:.3f}", sched_tone),
                ]),
                ("Estimate at completion", [
                    ("Typical (BAC / CPI)", money(fc["EAC_cpi"]), ""),
                    ("Atypical (AC + BAC − EV)", money(fc["EAC_ac"]), ""),
                ]),
                ("Estimate to complete", [
                    ("Typical", money(etc_cpi), ""),
                    ("Atypical", money(etc_ac), ""),
                ]),
                ("Outcome", [
                    ("Actual final cost", money(final_ac), ""),
                ]),
            ]
        )

# ── Performance: indices and variances side by side ──────────────────────────
with perf_box:
    section("Cost and schedule performance", "Indices above 1.0 mean the project is doing better than plan. Variances are cumulative.")
    pl, pr = st.columns(2, gap="large")
    with pl:
        fig, a = fig_one(6.2, 3.7)
        a.plot(df["sprint_id"], df["CPI"], color=ACCENT, linewidth=1.8, marker="o", markersize=3.5)
        a.plot(df["sprint_id"], df["SPI"], color=INK, linewidth=1.8, marker="o", markersize=3.5)
        a.axhline(1.0, color=SOFT, linestyle="--", linewidth=1)
        a.axvline(split + 0.5, color=SOFT, linestyle=":", linewidth=1)
        a.set_xlim(0.4, last_sprint + 2.0)
        style_ax(a, "CPI and SPI (1.0 = on track)")
        end_labels(
            a,
            [(df["CPI"].iloc[-1], "CPI", ACCENT), (df["SPI"].iloc[-1], "SPI", INK)],
            x=last_sprint + 0.4,
        )
        finish(fig)
    with pr:
        fig, a = fig_one(6.2, 3.7)
        x = df["sprint_id"].to_numpy()
        a.bar(x - 0.2, df["CV"], width=0.4, color=ACCENT, label="CV")
        a.bar(x + 0.2, df["SV"], width=0.4, color=SOFT, label="SV")
        a.axhline(0, color=INK, linewidth=0.8)
        a.set_xlim(0.4, last_sprint + 0.6)
        a.set_xticks(range(1, last_sprint + 1, 2 if last_sprint > 12 else 1))
        style_ax(a, "Variances (cumulative $m)")
        a.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="best")
        finish(fig)

# ── Forecast ─────────────────────────────────────────────────────────────────
with forecast_box:
    section(
        f"Forecast for sprints {split + 1}–{last_sprint}",
        f"The Random Forest is trained on sprints 1–{split} using sprint number, cumulative PV, EV and AC, "
        "CV, SV and last sprint's CPI and SPI. The baseline holds CPI and SPI flat at their checkpoint values. "
        "Shaded bands show each forecast's gap from the actual value.",
    )

    wins = []
    if m["MAE_CPI_rf"] < m["MAE_CPI_trad"]:
        wins.append("CPI")
    if m["MAE_SPI_rf"] < m["MAE_SPI_trad"]:
        wins.append("SPI")
    if len(wins) == 2:
        verdict = "<b>Random Forest</b> has the lower error on both CPI and SPI."
    elif wins:
        verdict = f"<b>Random Forest</b> has the lower error on {wins[0]} only."
    else:
        verdict = "The <b>traditional baseline</b> has the lower error on both CPI and SPI."
    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)

    fl, fr = st.columns(2, gap="large")
    with fl:
        forecast_chart(df, test, split, last_sprint, "CPI", "CPI_pred", "CPI_trad", "Cost Performance Index (CPI)")
    with fr:
        forecast_chart(df, test, split, last_sprint, "SPI", "SPI_pred", "SPI_trad", "Schedule Performance Index (SPI)")

    # Error summary — full width so nothing is cut off
    st.markdown('<div class="sub-t">Forecast error (lower is better)</div>', unsafe_allow_html=True)
    err_rows = []
    for k in ("CPI", "SPI"):
        mae_t, mae_r = m[f"MAE_{k}_trad"], m[f"MAE_{k}_rf"]
        rmse_t, rmse_r = m[f"RMSE_{k}_trad"], m[f"RMSE_{k}_rf"]
        rf_wins = mae_r < mae_t
        err_rows.append([
            (k, ""),
            (f"{mae_t:.4f}", "" if rf_wins else "best"),
            (f"{mae_r:.4f}", "best" if rf_wins else ""),
            (f"{rmse_t:.4f}", "" if rmse_r < rmse_t else "best"),
            (f"{rmse_r:.4f}", "best" if rmse_r < rmse_t else ""),
            ("Random Forest" if rf_wins else "Traditional", "best" if rf_wins else ""),
        ])
    html_table(
        ["", "MAE traditional", "MAE RF", "RMSE traditional", "RMSE RF", "Lower MAE"],
        err_rows,
        seps={1, 3, 5},
    )

    # Per-sprint predictions with grouped headers
    st.markdown('<div class="sub-t">Predictions by sprint</div>', unsafe_allow_html=True)
    pred_rows = []
    for _, r in test.iterrows():
        row = [(str(int(r["sprint_id"])), "")]
        for k in ("CPI", "SPI"):
            actual, t, p = r[k], r[f"{k}_trad"], r[f"{k}_pred"]
            rf_closer = abs(p - actual) < abs(t - actual)
            row += [
                (f"{actual:.4f}", ""),
                (f"{t:.4f}", "" if rf_closer else "best"),
                (f"{p:.4f}", "best" if rf_closer else ""),
            ]
        pred_rows.append(row)
    html_table(
        ["Sprint", "Actual", "Traditional", "RF", "Actual", "Traditional", "RF"],
        pred_rows,
        group=[("", 1), ("CPI", 3), ("SPI", 3)],
        seps={1, 4},
    )

# ── Data outputs ─────────────────────────────────────────────────────────────
with data_box:
    with st.expander("Full EVM table"):
        df_table(df.round(4))
    st.download_button(
        "Download EVM table (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="evm_table.csv",
        mime="text/csv",
    )