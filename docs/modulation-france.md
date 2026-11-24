# Output modulated during negative prices, mainland France, 2019-2025

Axis 4 of the [evaluation plan](evaluation-plan.md): how much solar and onshore wind output was
held back in the hours of negative day-ahead prices, estimated from a weather-based potential and
compared with the monthly volumes that RTE publishes.

```
eu-solar-wind-audit estimate-modulation FR solar data/processed/modulation-fr
eu-solar-wind-audit estimate-modulation FR wind_onshore data/processed/modulation-fr
```

The estimate uses the fit frozen in `fits/FR` (tag `calibration-fr-v1`), 999 bootstrap replicates
(seed 2026) and the RTE files of `references/rte`.

## Estimate

M̂ is the sum, over the N hours, of the calibrated potential p* minus the published output. N hours
have a negative mean day-ahead price and meet every other eligibility rule; they never enter the
fit. p* is the frozen potential times the ratio of its stratum and the annual level of its year.

| Year | Solar M̂, GWh (90 % interval) | RTE | Onshore wind M̂, GWh (90 % interval) | RTE |
|---|---|---|---|---|
| 2020 | 24.4 (12.7 to 35.1) | 25.9 | 71.3 (50.9 to 89.4) | 79.0 |
| 2021 | 34.8 (26.4 to 44.0) | 22.0 | 81.5 (63.6 to 99.9) | 57.1 |
| 2023 | 172.5 (142.4 to 200.7) | 167.4 | no onshore series | 376.8 |
| 2024 | 722.5 (634.2 to 817.3) | 639.2 | 914.5 (833.3 to 998.9) | 894.9 |
| 2025 | 1,581.4 (1,320.1 to 1,842.0) | 1,461.2 | 1,247.5 (1,146.8 to 1,341.1) | 1,197.1 |

N hours: solar 70, 44, 97, 321 and 480 from 2020 to 2025 (none in 2022); onshore wind 102, 64,
352 and 513. Published wind holds offshore output from June 2022 to 2023, so onshore wind has no
estimate for these months.

The interval comes from a stationary bootstrap of the days of each month, which refits the
calibration, the strata and the annual levels, and draws the DC/AC ratio and a capacity factor per
month. It covers the errors of these estimates, not the error of the model at the N hours
themselves: the placebo below measures that one.

## Verdict

| Test | Solar | Onshore wind |
|---|---|---|
| Annual ratio M̂ / RTE in [0.8, 1.25], 2024 and 2025 | passes: 1.13, 1.08 | passes: 1.02, 1.04 |
| Monthly log error below the prorata baseline | fails: 0.111 against 0.095 | passes: 0.109 against 0.257 |
| Spearman correlation of monthly rates ≥ 0.6 | fails: −0.08 | fails: −0.59 |
| Half-width of the interval, 2024 and 2025 | 13 %, 17 % | 9 %, 8 % |
| Minimal detectable effect, 2024 and 2025 | 66, 114 GWh | 182, 200 GWh |
| Verdict | concordant in volume only | concordant in volume only |

The minimal detectable effect of onshore wind in 2024 exceeds 20 % of the RTE volume (182 against
179 GWh), so the plan announces that year as likely inconclusive; the estimate is still well above
it, and its interval narrow.

The monthly tests fail for a reason that the data show: over the tested months, the RTE volumes
are an almost constant share of the potential at N hours, 0.19 to 0.22 for solar and 0.37 to 0.47
for wind. A constant rate is the prorata baseline itself, and it leaves the rank test little to
rank. RTE's method is not published, so this is a finding about the two series, not a judgement on
either.

## Placebo and injection

Placebo hours are P hours priced 5 to 15 €/MWh, matched to the N hours of the same year on month,
Paris hour and day type, then on the capacity factor of p* and the national consumption. They
leave the fit and should show no gap.

| Year | Solar: matched hours | Placebo gap, % of potential (90 % interval) | Wind: matched hours | Placebo gap, % of potential (90 % interval) |
|---|---|---|---|---|
| 2019 | 13 | −3.3 (−8.1 to 2.4) | 17 | 2.4 (−4.2 to 8.9) |
| 2020 | 53 | −4.6 (−8.9 to −0.5) | 79 | −0.1 (−2.9 to 2.6) |
| 2021 | 10 | 2.5 (−2.2 to 7.2) | 13 | 0.9 (−3.4 to 5.1) |
| 2023 | 16 | −2.1 (−5.6 to 1.6) | | |
| 2024 | 105 | −1.4 (−4.6 to 1.8) | 128 | 2.2 (−1.4 to 5.7) |
| 2025 | 138 | −2.8 (−5.9 to 0.2) | 158 | 2.7 (−0.7 to 6.0) |

- The plan asks the interval of each year to lie within ±2 % of the potential. It does not, in
  any year: the half-widths run from 3 to 7 % of the potential, and solar shows a gap of −1.4 to
  −4.6 % in five years of six. At hours of low positive prices, published solar exceeds p*, so at N hours
  M̂ may be too low by a few percent of their potential.
- About a third of the N hours find a placebo hour in their cell (335 of 1,032 for solar, 395 of
  1,058 for wind): low positive prices rarely fall in the same month, hour and day type.
- A modulation of 30 % injected at the placebo hours is recovered by the interval in 10 of the 11
  years of both technologies, against a target coverage of 85 to 95 %. Since these hours never
  enter the fit, the estimate moves by exactly the injected volume, so the injection repeats the
  placebo.

## Limits

- The check of the ENTSO-E day-ahead forecast, and its use as a covariate, wait for access to the
  ENTSO-E Transparency Platform. The placebo matches on the modelled capacity factor instead.
- On wind, 61 % of M̂ in 2024 and 47 % in 2025 comes from strata with fewer than 50 training hours,
  where the stratum ratio is shrunk towards the pooled ratio.
- The RTE wind series covers onshore farms, Corsica included; the éCO2mix series leave Corsica out.
