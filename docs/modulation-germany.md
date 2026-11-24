# Output modulated during negative prices, Germany, 2019-2025

Axis 4 of the [evaluation plan](evaluation-plan.md) for Germany: how much solar and onshore wind
output was held back in the hours of negative day-ahead prices, estimated from a weather-based
potential, and set beside the market-based curtailment that Hirth (2026) estimates. The plan
reports the ratio to his figures without a verdict.

```
eu-solar-wind-audit estimate-modulation DE solar data/processed/modulation-de
eu-solar-wind-audit estimate-modulation DE wind_onshore data/processed/modulation-de
```

The estimate uses the fit frozen in `fits/DE` (tag `calibration-de-v1`), the SMARD feed-in of
Germany, 999 bootstrap replicates (seed 2026) and the volumes of `references/hirth`.

## Estimate

M̂ is the sum, over the N hours, of the calibrated potential p* minus the published feed-in. N
hours have a negative mean day-ahead price in DE-LU and meet every other eligibility rule; they
never enter the fit. The share is M̂ over the potential at the N hours.

| Year | Solar M̂, GWh (90 % interval) | Share | Onshore wind M̂, GWh (90 % interval) | Share |
|---|---|---|---|---|
| 2019 | 46.2 (8.7 to 84.2) | 3.4 % | 409.5 (271.1 to 548.3) | 6.7 % |
| 2020 | 82.7 (−8.7 to 159.0) | 2.7 % | 774.4 (502.2 to 1,019.8) | 9.1 % |
| 2021 | 143.6 (70.3 to 216.0) | 6.9 % | 288.7 (185.1 to 391.5) | 8.0 % |
| 2022 | 80.8 (45.3 to 114.3) | 8.4 % | 559.3 (453.9 to 671.2) | 25.0 % |
| 2023 | 97.9 (−13.8 to 217.3) | 2.1 % | 1,618.6 (1,349.9 to 1,886.7) | 19.2 % |
| 2024 | 530.5 (243.5 to 852.5) | 4.4 % | 1,267.1 (1,061.9 to 1,480.0) | 15.2 % |
| 2025 | 1,432.9 (814.0 to 2,093.5) | 7.4 % | 1,755.0 (1,479.5 to 2,026.0) | 17.7 % |

N hours: solar 94, 185, 105, 43, 209, 391 and 545 from 2019 to 2025, with the sun up; onshore wind
211, 298, 139, 69, 301, 457 and 576.

## Against Hirth (2026)

Hirth estimates market-based curtailment as the potential minus actual generation minus
grid-related curtailment, with a potential fitted on ERA5 as well
([arXiv 2609.10053v2](https://arxiv.org/abs/2609.10053), section 5). The two estimates differ in
perimeter:

| | This estimate | Hirth (2026) |
|---|---|---|
| Wind | Onshore | Onshore and offshore |
| Grid-related curtailment | Kept in M̂ | Netted out, hour by hour |
| Published series | SMARD feed-in | TSO generation plus grid-related curtailment |
| Fit hours | Day-ahead price of 5 €/MWh or more, away from negative and missing hours | Day-ahead price of 5 €/MWh or more and imbalance price of −50 €/MWh or more |
| Negative-price hours of 2025 | 576 | 573 |

| Year | Onshore wind M̂, GWh | Hirth wind, GWh | Ratio | Share held back | Hirth curtailment rate |
|---|---|---|---|---|---|
| 2022 | 559.3 | | | 25.0 % | 22 % |
| 2023 | 1,618.6 | 1,800 | 0.90 | 19.2 % | 18 % |
| 2024 | 1,267.1 | 1,600 | 0.79 | 15.2 % | 15 % |
| 2025 | 1,755.0 | 2,100 | 0.84 | 17.7 % | 17 % |

| Year | Solar M̂, GWh | Hirth solar, GWh | Ratio | Share held back | Hirth curtailment rate |
|---|---|---|---|---|---|
| 2025 | 1,432.9 | 600 | 2.39 | 7.4 % | 3 % |

- Onshore wind: M̂ is 79 to 90 % of his volume, which includes offshore wind, and the share held
  back lies within 3 points of his curtailment rate in each year from 2022 (his Table 5). His
  offshore wind adds to his volume and the grid-related curtailment kept here adds to M̂, so the
  two differences of perimeter push in opposite directions.
- Solar: M̂ is 2.4 times his volume in 2025, and its interval leaves his figure out. Two
  differences can explain it. M̂ keeps grid-related curtailment, which his Figure 8 puts at about
  2.7 TWh of solar in 2025. And the feed-in falls at sunny midday hours when home batteries and
  self-consumption take more of the output, with no curtailment at all. The placebo measures part
  of this: in 2024, published solar at hours priced 5 to 15 €/MWh lies 4.4 % below p*, the share
  the N hours hold back that year, so the 2024 estimate cannot be told apart from that bias; in
  2025 the placebo gap is 0.9 % and the N hours hold back 7.4 %.

## Detectability and placebo

| | Solar | Onshore wind |
|---|---|---|
| Minimal detectable effect, 2023, 2024, 2025 | 115, 206, 274 GWh | 1,159, 816, 1,107 GWh |
| Half-width of the interval, 2024 and 2025 | 57 %, 45 % | 17 %, 16 % |
| Daily error of the potential, standard deviation | 0.053 over 124 days | 0.254 over 107 days |

Every estimate of 2023 to 2025 exceeds its minimal detectable effect except solar 2023. The
minimal detectable effect of onshore wind is 63 to 72 % of its estimate, against 16 to 20 % in
France in 2024 and 2025: the daily error of the German wind potential is larger.

Placebo hours are P hours priced 5 to 15 €/MWh, matched to the N hours of the same year on month,
Paris hour and day type, then on the capacity factor of p* and the load of the public grid. They
leave the fit and should show no gap.

| Year | Solar: matched hours | Placebo gap, % of potential (90 % interval) | Wind: matched hours | Placebo gap, % of potential (90 % interval) |
|---|---|---|---|---|
| 2019 | 13 | −2.8 (−6.6 to 0.9) | 48 | 0.4 (−3.8 to 4.6) |
| 2020 | 94 | −0.7 (−5.4 to 3.8) | 155 | 0.6 (−3.9 to 4.6) |
| 2021 | 11 | 2.1 (−4.2 to 9.0) | 14 | −6.4 (−11.6 to −1.6) |
| 2022 | 7 | −2.1 (−6.3 to 2.0) | 16 | 10.9 (6.2 to 15.5) |
| 2023 | 41 | 1.1 (−1.6 to 3.7) | 74 | 0.5 (−2.7 to 3.8) |
| 2024 | 73 | 4.4 (2.1 to 6.7) | 88 | 3.8 (1.1 to 6.1) |
| 2025 | 76 | 0.9 (−1.5 to 3.6) | 80 | 1.5 (−1.6 to 4.8) |

- As in France, no interval lies within ±2 % of the potential, as the plan asks.
- 315 of the 1,572 solar N hours and 475 of the 2,051 wind N hours find a placebo hour.
- A modulation of 30 % injected at the placebo hours is recovered in 6 of 7 years for solar and 4
  of 7 for wind. These hours never enter the fit, so the injection repeats the placebo: the years
  it misses are those whose placebo interval leaves zero out.

## Limits

- Offshore wind has no model, so the comparison of wind mixes perimeters.
- Grid-related curtailment is not netted out: it has no hourly series by technology here.
- Solar is estimated on the feed-in, which leaves out self-consumption and home storage.
- 34 to 63 % of the onshore wind estimate comes from strata with fewer than 50 training hours,
  where the stratum ratio is shrunk towards the pooled ratio (63 % in 2024, 48 % in 2025).
- Hirth's yearly volumes are read from his text and Figure 12; his hourly series are available on
  request only.
