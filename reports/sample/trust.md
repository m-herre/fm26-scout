# Trust report

Out-of-fold TabPFN predictions for every player. Coverage: share of true values inside
the 80% range (should be ~0.80). Narrow/wide: error for the third of players with the
narrowest / widest ranges. Market value in log units.

| target | verdict | skill | coverage | MAE | narrow | wide | range shown |
|---|---|---|---|---|---|---|---|
| potential_ability | reliable | 71% | 0.83 | 8.28 | 5.62 | 11.40 | yes |
| current_ability | reliable | 89% | 0.86 | 2.82 | 2.27 | 3.46 | yes |
| consistency | reliable | 50% | 0.80 | 1.72 | 1.07 | 2.20 | yes |
| important_matches | not predictable | 9% | 0.82 | 2.43 | 1.85 | 2.93 | yes |
| injury_proneness | not predictable | 6% | 0.79 | 2.76 | 2.45 | 2.94 | yes |
| dirtiness | not predictable | 4% | 0.80 | 2.41 | 2.17 | 2.59 | yes |
| versatility | reliable | 53% | 0.83 | 1.46 | 0.94 | 1.98 | yes |
| professionalism | rough | 21% | 0.88 | 1.39 | 0.50 | 2.24 | yes |
| ambition | rough | 25% | 0.80 | 2.84 | 2.20 | 3.50 | yes |
| adaptability | not predictable | -0% | 0.78 | 2.88 | 2.89 | 2.89 | yes |
| loyalty | not predictable | 4% | 0.80 | 2.59 | 1.72 | 3.35 | yes |
| pressure | not predictable | 13% | 0.81 | 2.58 | 1.91 | 3.11 | yes |
| temperament | not predictable | 4% | 0.80 | 2.97 | 2.44 | 3.36 | yes |
| sportsmanship | not predictable | 5% | 0.80 | 2.61 | 2.08 | 3.21 | yes |
| controversy | not predictable | 4% | 0.81 | 2.94 | 2.41 | 3.22 | yes |
| market_value | reliable | 76% | 0.85 | 0.48 | 0.27 | 0.71 | yes |

## Chance of PA 150+: quoted vs real

From the full distribution: Brier 0.0137, AUC 0.983

| players | quoted | real |
|---|---|---|
| 45,740 | 0% | 0% |
| 2,427 | 10% | 11% |
| 784 | 29% | 31% |
| 410 | 49% | 57% |
| 349 | 70% | 79% |
| 492 | 93% | 95% |

A rare event: the base rate is the share of players who really reach it, PR-AUC
(average precision) the ranking quality against that, top 50/100 how many of the
players with the highest chance really reach it.

| players | n | base rate | AUC | PR-AUC | top 50 | top 100 |
|---|---|---|---|---|---|---|
| all players | 50,202 | 3.2% | 0.983 | 0.773 | 50 of 50 | 100 of 100 |
| age 21 or under | 18,624 | 2.1% | 0.987 | 0.672 | 43 of 50 | 85 of 100 |

## Chance of PA 160+: quoted vs real

From the full distribution: Brier 0.0064, AUC 0.983

| players | quoted | real |
|---|---|---|
| 48,171 | 0% | 0% |
| 1,128 | 10% | 9% |
| 435 | 29% | 27% |
| 212 | 49% | 45% |
| 109 | 69% | 67% |
| 147 | 93% | 96% |

A rare event: the base rate is the share of players who really reach it, PR-AUC
(average precision) the ranking quality against that, top 50/100 how many of the
players with the highest chance really reach it.

| players | n | base rate | AUC | PR-AUC | top 50 | top 100 |
|---|---|---|---|---|---|---|
| all players | 50,202 | 1.2% | 0.983 | 0.658 | 50 of 50 | 97 of 100 |
| age 21 or under | 18,624 | 0.8% | 0.987 | 0.570 | 39 of 50 | 64 of 100 |
