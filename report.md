# EVM Analysis with AI-enhanced Forecasting — Report

BAC (budget at completion, final cumulative PV): **$1.505m**. All values in $m. Checkpoint for forecasting: sprint 8.

## Part 1 — Variance Calculation (CV = EV − AC, SV = EV − PV)

| sprint | cum_PV | cum_EV | cum_AC | CV | SV | status |
|---|---|---|---|---|---|---|
| 1 | 0.090 | 0.088 | 0.090 | -0.002 | -0.002 | over budget, behind schedule |
| 2 | 0.185 | 0.185 | 0.184 | +0.001 | +0.000 | under budget, on schedule |
| 3 | 0.285 | 0.287 | 0.284 | +0.003 | +0.002 | under budget, ahead of schedule |
| 4 | 0.385 | 0.385 | 0.386 | -0.001 | +0.000 | over budget, on schedule |
| 5 | 0.490 | 0.475 | 0.494 | -0.019 | -0.015 | over budget, behind schedule |
| 6 | 0.595 | 0.563 | 0.604 | -0.041 | -0.032 | over budget, behind schedule |
| 7 | 0.705 | 0.658 | 0.716 | -0.058 | -0.047 | over budget, behind schedule |
| 8 | 0.805 | 0.750 | 0.821 | -0.071 | -0.055 | over budget, behind schedule |
| 9 | 0.905 | 0.852 | 0.921 | -0.069 | -0.053 | over budget, behind schedule |
| 10 | 1.010 | 0.960 | 1.023 | -0.063 | -0.050 | over budget, behind schedule |
| 11 | 1.110 | 1.064 | 1.121 | -0.057 | -0.046 | over budget, behind schedule |
| 12 | 1.205 | 1.162 | 1.217 | -0.055 | -0.043 | over budget, behind schedule |
| 13 | 1.300 | 1.254 | 1.315 | -0.061 | -0.046 | over budget, behind schedule |
| 14 | 1.400 | 1.350 | 1.417 | -0.067 | -0.050 | over budget, behind schedule |
| 15 | 1.505 | 1.450 | 1.523 | -0.073 | -0.055 | over budget, behind schedule |

Finding: sprints 1–4 hover around zero (CV/SV ≈ 0). Sprints 5–8 deteriorate to CV=$-0.071m / SV=$-0.055m — the project is over budget and behind schedule. Sprints 9–12 partially recover (CV improves from $-0.069m to $-0.055m) before plateauing; final sprint 15 ends at CV=$-0.073m, SV=$-0.055m — still over budget / behind, but stable.

## Part 2 — Performance Index Analysis (CPI = EV/AC, SPI = EV/PV)

CPI trough at sprint 8: **0.914**; SPI trough: **0.932** (both < 1.0 ⇒ inefficient / behind). Recovery to sprint 12: CPI 0.955, SPI 0.964; final sprint 15: CPI 0.952, SPI 0.963. See `evm_dashboard.png` (S-curve, CPI/SPI trends, CV/SV bars).

## Part 3 — EVM Forecasting (EAC / ETC at sprint 8)

- Formula A (typical, efficiency continues): EAC = BAC / CPI = $1.647m; ETC = EAC − AC = $0.826m; VAC = BAC − EAC = $-0.142m (overrun).
- Formula B (atypical, future at planned rate): EAC = AC + (BAC − EV) = $1.576m; ETC = $0.755m; VAC = $-0.071m.
- (Bonus composite EAC = AC + (BAC−EV)/(CPI·SPI) = $1.708m.)
Actual final cost was $1.523m: Formula B ($1.576m) was closer here because the team partially recovered after sprint 8, while Formula A extrapolated the trough efficiency forward and over-predicted the overrun.

## Part 4 — AI-enhanced Forecasting (Random Forest, train 1–8 → predict 9–15)

Features: sprint_id, cum_PV, cum_EV, cum_AC, CV, SV, CPI_lag1, SPI_lag1. Model: RandomForestRegressor (n_estimators=300, max_depth=4, random_state=9), one multi-output model for CPI/SPI.

| sprint | CPI actual | CPI trad (flat) | CPI RF | SPI actual | SPI trad | SPI RF |
|---|---|---|---|---|---|---|
| 9 | 0.925 | 0.914 | 0.917 | 0.941 | 0.932 | 0.934 |
| 10 | 0.938 | 0.914 | 0.918 | 0.950 | 0.932 | 0.934 |
| 11 | 0.949 | 0.914 | 0.918 | 0.959 | 0.932 | 0.934 |
| 12 | 0.955 | 0.914 | 0.923 | 0.964 | 0.932 | 0.938 |
| 13 | 0.954 | 0.914 | 0.924 | 0.965 | 0.932 | 0.939 |
| 14 | 0.953 | 0.914 | 0.924 | 0.964 | 0.932 | 0.939 |
| 15 | 0.952 | 0.914 | 0.923 | 0.963 | 0.932 | 0.939 |

Error on sprints 9–15 — CPI: RF MAE 0.0255 vs traditional 0.0330 (RMSE 0.0267 vs 0.0346); SPI: RF MAE 0.0215 vs traditional 0.0265 (RMSE 0.0225 vs 0.0278). Chart: `evm_forecast_comparison.png` (actual vs flat EVM vs RF trajectory).

### Where AI diverged from traditional EVM, and why it matters

The flat EVM forecast froze CPI/SPI at their sprint-8 trough (~0.914/0.932) and therefore under-predicted every recovery sprint, while the Random Forest predicted a partial rebound (CPI ≈0.918–0.924, SPI ≈0.934–0.939) — right direction, but still below the actual recovery to CPI ≈0.955 / SPI ≈0.964, because its training window (sprints 1–8, mostly deteriorating) contained no recovery example to learn from. This matters for dynamic software projects: constant-efficiency EVM extrapolations mislead at regime shifts (e.g. after scope-creep is fixed), and small-data ML improves on them by learning the trend shape yet still underestimates rebounds it has never seen — so forecasts should be retrained as new sprints arrive rather than trusted blindly.
