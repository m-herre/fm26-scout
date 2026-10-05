# Hidden-attribute benchmark

data: sample/players.parquet · train rows per target: 10,000 · test players: 10,041

Mean absolute error on held-out players (lower is better; market value on log scale).
Skill = share of the median-guess error removed by the best model.
TabPFN vs best other: MAE of TabPFN minus MAE of the best of ridge, random forest and
CatBoost, with a 95% bootstrap interval (1,000 paired resamples of the test players,
seed 0). Better: the whole interval is below 0; tied: it contains 0; worse: above 0.
⚠ marks targets where TabPFN is not better than the median guess by the same test.

| target | median | ridge | random_forest | catboost | tabpfn | best | skill | R² (best) | TabPFN vs best other (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| potential_ability | 28.771 | 11.734 | 9.959 | 9.564 | **8.477** | tabpfn | 71% | 0.890 | -1.087 [-1.182, -0.988] vs catboost | better |
| current_ability | 26.182 | 4.085 | 4.960 | 3.012 | **2.815** | tabpfn | 89% | 0.987 | -0.198 [-0.241, -0.158] vs catboost | better |
| consistency | 3.413 | 1.926 | 1.827 | 1.818 | **1.799** | tabpfn | 47% | 0.647 | -0.019 [-0.027, -0.011] vs catboost | better |
| important_matches | 2.645 | 2.521 | 2.473 | 2.476 | **2.437** | tabpfn | 8% | 0.161 | -0.036 [-0.046, -0.027] vs random_forest | better |
| injury_proneness | 2.943 | 2.815 | 2.799 | 2.806 | **2.779** | tabpfn | 6% | 0.097 | -0.020 [-0.028, -0.013] vs random_forest | better |
| dirtiness | 2.529 | 2.470 | 2.458 | 2.455 | **2.426** | tabpfn | 4% | 0.074 | -0.029 [-0.040, -0.019] vs catboost | better |
| versatility | 3.096 | 1.617 | 1.501 | 1.498 | **1.483** | tabpfn | 52% | 0.710 | -0.015 [-0.022, -0.008] vs catboost | better |
| professionalism | 1.757 | 1.745 | 1.714 | 1.601 | **1.551** | tabpfn | 12% | 0.291 | -0.051 [-0.061, -0.039] vs catboost | better |
| ambition | 3.818 | 3.115 | 2.962 | 2.948 | **2.888** | tabpfn | 24% | 0.343 | -0.060 [-0.075, -0.047] vs catboost | better |
| adaptability | 2.905 | 2.935 | 2.930 | 2.957 | **2.923** | tabpfn | -1% | -0.001 | -0.007 [-0.013, +0.000] vs random_forest | tied ⚠ vs median: worse +0.018 [+0.012, +0.023] |
| loyalty | 2.707 | 2.726 | 2.657 | 2.652 | **2.613** | tabpfn | 4% | 0.088 | -0.040 [-0.050, -0.028] vs catboost | better |
| pressure | 2.918 | 2.678 | 2.609 | 2.623 | **2.561** | tabpfn | 12% | 0.213 | -0.049 [-0.061, -0.037] vs random_forest | better |
| temperament | 3.078 | 3.070 | 3.031 | 3.034 | **2.981** | tabpfn | 3% | 0.061 | -0.050 [-0.060, -0.039] vs random_forest | better |
| sportsmanship | 2.741 | 2.715 | 2.694 | 2.692 | **2.639** | tabpfn | 4% | 0.088 | -0.053 [-0.065, -0.042] vs catboost | better |
| controversy | 3.045 | 3.061 | 2.981 | 3.007 | **2.969** | tabpfn | 3% | 0.077 | -0.012 [-0.021, -0.004] vs random_forest | better |
| market_value | 2.000 | 0.712 | 0.671 | 0.566 | **0.478** | tabpfn | 76% | 0.927 | -0.089 [-0.098, -0.078] vs catboost | better |

Against the best other model TabPFN is better on 15, tied on 1 of 16 targets.

## Baselines with 4× the rows

rows asked: 40,000 · models: catboost, random_forest

The rivals refitted on more training rows (the whole train split when it is
smaller); TabPFN keeps its 10,000-row fit. Same test players, same paired CI.

| target | model | train rows | MAE (more rows) | MAE (10,000) | TabPFN | TabPFN vs it (95% CI) | verdict |
|---|---|---|---|---|---|---|---|
| potential_ability | catboost | 40,000 | 8.910 | 9.564 | 8.477 | -0.434 [-0.520, -0.339] | better |
| potential_ability | random_forest | 40,000 | 9.380 | 9.959 | 8.477 | -0.904 [-1.005, -0.796] | better |
| current_ability | catboost | 40,000 | 2.558 | 3.012 | 2.815 | +0.257 [+0.215, +0.294] | worse |
| current_ability | random_forest | 40,000 | 4.584 | 4.960 | 2.815 | -1.770 [-1.837, -1.704] | better |
| market_value | catboost | 21,004 | 0.535 | 0.566 | 0.478 | -0.057 [-0.067, -0.048] | better |
| market_value | random_forest | 21,004 | 0.630 | 0.671 | 0.478 | -0.152 [-0.165, -0.140] | better |

## TabPFN error by slice (MAE)

| target | <=18 | 19-21 | 22-25 | 26-29 | 30+ | GK | DEF | MID | ATT | Obscure | Local | Regional | National | Continental | Worldwide | known | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| potential_ability | 9.17 | 7.66 | 8.19 | 8.27 | 9.02 | 9.55 | 8.64 | 8.29 | 8.14 | 7.13 | 9.45 | 8.53 | 7.99 | 7.02 | 4.06 | 7.59 | 9.46 |
| current_ability | 3.04 | 2.72 | 2.70 | 2.66 | 2.95 | 2.95 | 2.59 | 2.95 | 2.90 | 2.79 | 2.81 | 2.84 | 2.81 | 2.67 | 3.47 | 2.77 | 2.87 |
| consistency | 2.40 | 2.27 | 2.01 | 1.44 | 0.94 | 1.64 | 1.81 | 1.78 | 1.84 | 1.88 | 1.93 | 1.83 | 1.69 | 1.39 | 1.21 | 1.92 | 1.67 |
| important_matches | 2.65 | 2.49 | 2.38 | 2.36 | 2.33 | 2.80 | 2.53 | 2.36 | 2.30 | 2.49 | 2.54 | 2.45 | 2.37 | 2.15 | 1.56 | 2.43 | 2.45 |
| injury_proneness | 2.71 | 2.71 | 2.82 | 2.82 | 2.83 | 2.79 | 2.77 | 2.73 | 2.82 | 2.97 | 2.82 | 2.80 | 2.74 | 2.45 | 2.19 | 2.69 | 2.88 |
| dirtiness | 2.46 | 2.48 | 2.39 | 2.40 | 2.41 | 2.49 | 2.41 | 2.45 | 2.40 | 2.74 | 2.49 | 2.42 | 2.30 | 2.36 | 2.07 | 2.37 | 2.48 |
| versatility | 1.88 | 1.83 | 1.68 | 1.23 | 0.85 | 1.15 | 1.61 | 1.58 | 1.38 | 1.45 | 1.50 | 1.53 | 1.46 | 1.29 | 0.96 | 1.59 | 1.37 |
| professionalism | 1.89 | 1.62 | 1.45 | 1.41 | 1.42 | 1.47 | 1.57 | 1.56 | 1.55 | 1.59 | 1.43 | 1.69 | 1.55 | 1.41 | 1.16 | 1.65 | 1.45 |
| ambition | 2.78 | 2.83 | 2.89 | 2.95 | 2.97 | 2.96 | 2.89 | 2.80 | 2.93 | 2.65 | 2.73 | 3.05 | 2.97 | 2.70 | 2.25 | 2.92 | 2.85 |
| adaptability | 2.97 | 2.89 | 2.96 | 2.94 | 2.86 | 2.86 | 2.93 | 2.94 | 2.93 | 2.96 | 2.90 | 2.95 | 2.89 | 2.97 | 2.58 | 2.90 | 2.95 |
| loyalty | 2.57 | 2.58 | 2.60 | 2.59 | 2.72 | 2.62 | 2.58 | 2.59 | 2.67 | 2.20 | 2.38 | 2.67 | 2.85 | 2.89 | 2.65 | 2.73 | 2.49 |
| pressure | 2.82 | 2.64 | 2.51 | 2.42 | 2.44 | 2.94 | 2.80 | 2.39 | 2.34 | 2.64 | 2.61 | 2.59 | 2.47 | 2.49 | 2.08 | 2.58 | 2.54 |
| temperament | 3.03 | 2.99 | 3.02 | 2.94 | 2.93 | 3.05 | 2.98 | 2.94 | 2.99 | 2.77 | 2.92 | 3.02 | 3.05 | 2.99 | 2.84 | 3.00 | 2.96 |
| sportsmanship | 2.64 | 2.60 | 2.65 | 2.70 | 2.60 | 2.77 | 2.60 | 2.58 | 2.68 | 2.37 | 2.54 | 2.70 | 2.75 | 2.65 | 2.80 | 2.69 | 2.59 |
| controversy | 2.97 | 2.88 | 3.01 | 2.96 | 3.01 | 3.01 | 2.89 | 2.98 | 3.03 | 2.84 | 2.82 | 3.02 | 3.11 | 2.97 | 2.95 | 3.01 | 2.92 |
| market_value | 0.50 | 0.48 | 0.51 | 0.39 | 0.49 | 0.54 | 0.48 | 0.45 | 0.48 | 0.26 | 0.35 | 0.49 | 0.52 | 0.45 | 0.38 | 0.48 | nan |
