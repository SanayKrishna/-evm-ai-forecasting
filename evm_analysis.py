import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 9
SPLIT = 8  # train on sprints 1-8, predict 9-15

# ---------------------------------------------------------------- Part 0: data
def build_dataset() -> pd.DataFrame:
    # Per-sprint incremental values in $m. Narrative:
    #  1-4 on track, 5-8 troubled (scope creep / overruns),
    #  9-12 recovery, 13-15 plateau with slight slip.
    pv = [0.090, 0.095, 0.100, 0.100, 0.105, 0.105, 0.110, 0.100,
          0.100, 0.105, 0.100, 0.095, 0.095, 0.100, 0.105]
    ev = [0.088, 0.097, 0.102, 0.098, 0.090, 0.088, 0.095, 0.092,
          0.102, 0.108, 0.104, 0.098, 0.092, 0.096, 0.100]
    ac = [0.090, 0.094, 0.100, 0.102, 0.108, 0.110, 0.112, 0.105,
          0.100, 0.102, 0.098, 0.096, 0.098, 0.102, 0.106]
    df = pd.DataFrame({"sprint_id": range(1, 16), "PV": pv, "EV": ev, "AC": ac})
    for c in ["PV", "EV", "AC"]:
        df[f"cum_{c}"] = df[c].cumsum()
    return df


# ------------------------------------------------- Parts 1 & 2: EVM metrics
def add_evm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CV"] = df["cum_EV"] - df["cum_AC"]   # cost variance
    df["SV"] = df["cum_EV"] - df["cum_PV"]   # schedule variance
    df["CPI"] = df["cum_EV"] / df["cum_AC"]  # cost performance index
    df["SPI"] = df["cum_EV"] / df["cum_PV"]  # schedule performance index
    return df


def status_row(r) -> str:
    b = "over budget" if r["CV"] < 0 else ("on budget" if r["CV"] == 0 else "under budget")
    s = "behind schedule" if r["SV"] < 0 else ("on schedule" if r["SV"] == 0 else "ahead of schedule")
    return f"{b}, {s}"


# ------------------------------------------------------- Part 3: EAC / ETC
def evm_forecasts(df: pd.DataFrame, at_sprint: int = 8):
    BAC = df["cum_PV"].iloc[-1]  # budget at completion = final planned total
    row = df.loc[df["sprint_id"] == at_sprint].iloc[0]
    AC, EV, CPI, SPI = row["cum_AC"], row["cum_EV"], row["CPI"], row["SPI"]
    EAC_cpi = BAC / CPI                    # assumes current cost efficiency continues
    EAC_ac = AC + (BAC - EV)               # assumes future work at planned efficiency
    EAC_composite = AC + (BAC - EV) / (CPI * SPI)  # bonus: cost+schedule influence
    out = {
        "BAC": BAC, "at_sprint": at_sprint, "AC": AC, "EV": EV,
        "CPI": CPI, "SPI": SPI,
        "EAC_cpi": EAC_cpi, "ETC_cpi": EAC_cpi - AC, "VAC_cpi": BAC - EAC_cpi,
        "EAC_ac": EAC_ac, "ETC_ac": EAC_ac - AC, "VAC_ac": BAC - EAC_ac,
        "EAC_composite": EAC_composite, "ETC_composite": EAC_composite - AC,
    }
    return out


# ------------------------------------------------- Part 4: AI forecasting
FEATURES = ["sprint_id", "cum_PV", "cum_EV", "cum_AC", "CV", "SV", "CPI_lag1", "SPI_lag1"]

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CPI_lag1"] = df["CPI"].shift(1).bfill()
    df["SPI_lag1"] = df["SPI"].shift(1).bfill()
    return df


def train_and_predict(df: pd.DataFrame, split: int = SPLIT):
    train = df[df["sprint_id"] <= split].copy()
    test = df[df["sprint_id"] > split].copy()
    model = RandomForestRegressor(
        n_estimators=300, max_depth=4, random_state=RANDOM_STATE)
    model.fit(train[FEATURES], train[["CPI", "SPI"]])
    pred = model.predict(test[FEATURES])
    test = test.copy()
    test["CPI_pred"] = pred[:, 0]
    test["SPI_pred"] = pred[:, 1]
    # Traditional EVM forecast: flat-line CPI/SPI from checkpoint sprint
    CPI8 = train.iloc[-1]["CPI"]
    SPI8 = train.iloc[-1]["SPI"]
    test["CPI_trad"] = CPI8
    test["SPI_trad"] = SPI8
    metrics = {
        "CPI8": CPI8, "SPI8": SPI8,
        "MAE_CPI_trad": mean_absolute_error(test["CPI"], test["CPI_trad"]),
        "MAE_SPI_trad": mean_absolute_error(test["SPI"], test["SPI_trad"]),
        "MAE_CPI_rf": mean_absolute_error(test["CPI"], test["CPI_pred"]),
        "MAE_SPI_rf": mean_absolute_error(test["SPI"], test["SPI_pred"]),
        "RMSE_CPI_trad": float(np.sqrt(mean_squared_error(test["CPI"], test["CPI_trad"]))),
        "RMSE_SPI_trad": float(np.sqrt(mean_squared_error(test["SPI"], test["SPI_trad"]))),
        "RMSE_CPI_rf": float(np.sqrt(mean_squared_error(test["CPI"], test["CPI_pred"]))),
        "RMSE_SPI_rf": float(np.sqrt(mean_squared_error(test["SPI"], test["SPI_pred"]))),
    }
    return train, test, model, metrics


# ------------------------------------------------------------------ Charts
def plot_dashboard(df: pd.DataFrame, path: str = "evm_dashboard.png"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("EVM Performance Dashboard (cumulative, $m)", fontsize=13, fontweight="bold")
    # S-curve
    a = ax[0, 0]
    a.plot(df["sprint_id"], df["cum_PV"], marker="o", label="PV (planned)")
    a.plot(df["sprint_id"], df["cum_EV"], marker="s", label="EV (earned)")
    a.plot(df["sprint_id"], df["cum_AC"], marker="^", label="AC (actual)")
    a.set_title("S-Curve: PV / EV / AC"); a.set_xlabel("Sprint"); a.set_ylabel("$m")
    a.legend(); a.grid(alpha=0.3)
    # CPI / SPI
    b = ax[0, 1]
    b.plot(df["sprint_id"], df["CPI"], marker="o", label="CPI (EV/AC)")
    b.plot(df["sprint_id"], df["SPI"], marker="s", label="SPI (EV/PV)")
    b.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    b.set_title("Performance Indices (1.0 = on track)"); b.set_xlabel("Sprint")
    b.legend(); b.grid(alpha=0.3); b.set_ylim(0.85, 1.05)
    # Variances
    c = ax[1, 0]
    x = df["sprint_id"].to_numpy()
    c.bar(x - 0.2, df["CV"], width=0.4, label="CV (EV-AC)")
    c.bar(x + 0.2, df["SV"], width=0.4, label="SV (EV-PV)")
    c.axhline(0, color="black", linewidth=0.8)
    c.set_title("Variances per Sprint (cumulative)"); c.set_xlabel("Sprint"); c.set_ylabel("$m")
    c.legend(); c.grid(alpha=0.3, axis="y")
    # Final snapshot text
    d = ax[1, 1]
    d.axis("off")
    last = df.iloc[-1]
    txt = (
        f"Final sprint (15) snapshot\n"
        f"-------------------------\n"
        f"BAC (total PV): ${df['cum_PV'].iloc[-1]:.3f}m\n"
        f"EV total: ${last['cum_EV']:.3f}m\n"
        f"AC total: ${last['cum_AC']:.3f}m\n"
        f"CV = ${last['CV']:.3f}m  ({'over budget' if last['CV'] < 0 else 'under budget'})\n"
        f"SV = ${last['SV']:.3f}m  ({'behind schedule' if last['SV'] < 0 else 'ahead'})\n"
        f"CPI = {last['CPI']:.3f}   SPI = {last['SPI']:.3f}\n\n"
        f"Trough at sprint 8:\n"
        f"CPI = {df.loc[7, 'CPI']:.3f}, SPI = {df.loc[7, 'SPI']:.3f}"
    )
    d.text(0.05, 0.95, txt, va="top", ha="left", fontsize=10, family="monospace",
           bbox=dict(boxstyle="round", facecolor="whitesmoke"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_comparison(df, test, path="evm_forecast_comparison.png"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    fig.suptitle("AI (Random Forest) vs Traditional EVM Forecast — CPI / SPI (sprints 9-15)",
                 fontsize=12, fontweight="bold")
    for i, (col, pred, trad, title) in enumerate([
            ("CPI", "CPI_pred", "CPI_trad", "Cost Performance Index (CPI)"),
            ("SPI", "SPI_pred", "SPI_trad", "Schedule Performance Index (SPI)")]):
        a = ax[i]
        a.plot(df["sprint_id"], df[col], "k-o", label="Actual", linewidth=2)
        a.plot(test["sprint_id"], test[trad], "r--s", label="Traditional EVM (flat CPI/SPI @ sprint 8)")
        a.plot(test["sprint_id"], test[pred], color="b", marker="^",
               linestyle=(0, (3, 1)), label="AI predicted (Random Forest)")
        a.axvline(SPLIT + 0.5, color="gray", linestyle=":", label=f"Train/test split (after sprint {SPLIT})")
        a.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        a.set_title(title); a.set_xlabel("Sprint"); a.set_xticks(range(1, 16))
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Report
def build_report(df, fc, test, metrics) -> str:
    last = df.iloc[-1]
    r8 = df.loc[df["sprint_id"] == 8].iloc[0]
    lines = [
        "# EVM Analysis with AI-enhanced Forecasting — Report",
        "",
        f"BAC (budget at completion, final cumulative PV): **${fc['BAC']:.3f}m**. "
        f"All values in $m. Checkpoint for forecasting: sprint {fc['at_sprint']}.",
        "",
        "## Part 1 — Variance Calculation (CV = EV − AC, SV = EV − PV)",
        "",
        "| sprint | cum_PV | cum_EV | cum_AC | CV | SV | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(f"| {r.sprint_id} | {r.cum_PV:.3f} | {r.cum_EV:.3f} | {r.cum_AC:.3f} "
                     f"| {r.CV:+.3f} | {r.SV:+.3f} | {status_row(r._asdict())} |")
    lines += [
        "",
        f"Finding: sprints 1–4 hover around zero (CV/SV ≈ 0). Sprints 5–8 deteriorate to "
        f"CV=${r8['CV']:.3f}m / SV=${r8['SV']:.3f}m — the project is over budget and behind schedule. "
        f"Sprints 9–12 partially recover (CV improves from ${df.loc[8, 'CV']:.3f}m to "
        f"${df.loc[11, 'CV']:.3f}m) before plateauing; final sprint 15 ends at "
        f"CV=${last['CV']:.3f}m, SV=${last['SV']:.3f}m — still over budget / behind, but stable.",
        "",
        "## Part 2 — Performance Index Analysis (CPI = EV/AC, SPI = EV/PV)",
        "",
        f"CPI trough at sprint 8: **{r8['CPI']:.3f}**; SPI trough: **{r8['SPI']:.3f}** "
        f"(both < 1.0 ⇒ inefficient / behind). Recovery to sprint 12: "
        f"CPI {df.loc[11, 'CPI']:.3f}, SPI {df.loc[11, 'SPI']:.3f}; final sprint 15: "
        f"CPI {last['CPI']:.3f}, SPI {last['SPI']:.3f}. "
        "See `evm_dashboard.png` (S-curve, CPI/SPI trends, CV/SV bars).",
        "",
        "## Part 3 — EVM Forecasting (EAC / ETC at sprint 8)",
        "",
        f"- Formula A (typical, efficiency continues): EAC = BAC / CPI = "
        f"${fc['EAC_cpi']:.3f}m; ETC = EAC − AC = ${fc['ETC_cpi']:.3f}m; "
        f"VAC = BAC − EAC = ${fc['VAC_cpi']:.3f}m (overrun).",
        f"- Formula B (atypical, future at planned rate): EAC = AC + (BAC − EV) = "
        f"${fc['EAC_ac']:.3f}m; ETC = ${fc['ETC_ac']:.3f}m; VAC = ${fc['VAC_ac']:.3f}m.",
        f"- (Bonus composite EAC = AC + (BAC−EV)/(CPI·SPI) = ${fc['EAC_composite']:.3f}m.)",
        f"Actual final cost was ${last['cum_AC']:.3f}m: Formula B (${fc['EAC_ac']:.3f}m) was closer "
        f"here because the team partially recovered after sprint 8, while Formula A "
        f"extrapolated the trough efficiency forward and over-predicted the overrun.",
        "",
        "## Part 4 — AI-enhanced Forecasting (Random Forest, train 1–8 → predict 9–15)",
        "",
        f"Features: {', '.join(FEATURES)}. Model: RandomForestRegressor "
        f"(n_estimators=300, max_depth=4, random_state={RANDOM_STATE}), one multi-output model for CPI/SPI.",
        "",
        "| sprint | CPI actual | CPI trad (flat) | CPI RF | SPI actual | SPI trad | SPI RF |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in test.itertuples():
        lines.append(f"| {r.sprint_id} | {r.CPI:.3f} | {r.CPI_trad:.3f} | {r.CPI_pred:.3f} "
                     f"| {r.SPI:.3f} | {r.SPI_trad:.3f} | {r.SPI_pred:.3f} |")
    lines += [
        "",
        f"Error on sprints 9–15 — CPI: RF MAE {metrics['MAE_CPI_rf']:.4f} vs traditional "
        f"{metrics['MAE_CPI_trad']:.4f} (RMSE {metrics['RMSE_CPI_rf']:.4f} vs {metrics['RMSE_CPI_trad']:.4f}); "
        f"SPI: RF MAE {metrics['MAE_SPI_rf']:.4f} vs traditional {metrics['MAE_SPI_trad']:.4f} "
        f"(RMSE {metrics['RMSE_SPI_rf']:.4f} vs {metrics['RMSE_SPI_trad']:.4f}). "
        "Chart: `evm_forecast_comparison.png` (actual vs flat EVM vs RF trajectory).",
        "",
        "### Where AI diverged from traditional EVM, and why it matters",
        "",
        "The flat EVM forecast froze CPI/SPI at their sprint-8 trough (~0.914/0.932) and therefore "
        "under-predicted every recovery sprint, while the Random Forest predicted a partial rebound "
        "(CPI ≈0.918–0.924, SPI ≈0.934–0.939) — right direction, but still below the actual recovery "
        "to CPI ≈0.955 / SPI ≈0.964, because its training window (sprints 1–8, mostly deteriorating) "
        "contained no recovery example to learn from. This matters for dynamic software projects: "
        "constant-efficiency EVM extrapolations mislead at regime shifts (e.g. after scope-creep is fixed), "
        "and small-data ML improves on them by learning the trend shape yet still underestimates rebounds "
        "it has never seen — so forecasts should be retrained as new sprints arrive rather than trusted blindly.",
        "",
    ]
    return "\n".join(lines)


def main():
    df = add_evm_metrics(build_dataset())
    df = add_lag_features(df)
    fc = evm_forecasts(df, at_sprint=SPLIT)
    train, test, model, metrics = train_and_predict(df, split=SPLIT)

    print("=== EVM table (cumulative, $m) ===")
    print(df[["sprint_id", "PV", "EV", "AC", "cum_PV", "cum_EV", "cum_AC",
              "CV", "SV", "CPI", "SPI"]].round(4).to_string(index=False))
    print(f"\nBAC = ${fc['BAC']:.4f}m")
    print(f"EAC (BAC/CPI) = ${fc['EAC_cpi']:.4f}m, ETC = ${fc['ETC_cpi']:.4f}m")
    print(f"EAC (AC+BAC-EV) = ${fc['EAC_ac']:.4f}m, ETC = ${fc['ETC_ac']:.4f}m")
    print(f"\nRF vs traditional — MAE CPI: {metrics['MAE_CPI_rf']:.4f} vs {metrics['MAE_CPI_trad']:.4f}; "
          f"MAE SPI: {metrics['MAE_SPI_rf']:.4f} vs {metrics['MAE_SPI_trad']:.4f}")

    df.to_csv("evm_table.csv", index=False)
    plot_dashboard(df)
    plot_forecast_comparison(df, test)
    report = build_report(df, fc, test, metrics)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nSaved: evm_table.csv, evm_dashboard.png, evm_forecast_comparison.png, report.md")


if __name__ == "__main__":
    main()
