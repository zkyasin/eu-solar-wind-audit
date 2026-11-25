# Output modulated during negative prices, Netherlands, 2019-2025

Axis 4 of the [evaluation plan](evaluation-plan.md) for Dutch onshore wind: how much output of the
farms behind the series of the NL zone on the ENTSO-E Transparency Platform was held back in the
hours of negative day-ahead prices, estimated from a weather-based potential. No published volume
of Dutch modulation is known, so the estimates carry no reference and, as the plan says, count
only where they reach their minimal detectable effect; below it they are inconclusive.

```
ENTSOE_API_TOKEN=... eu-solar-wind-audit estimate-modulation NL wind_onshore data/processed/modulation-nl
```

The estimate uses the fit frozen in `fits/NL` (tag `calibration-nl-v1`) and the onshore wind of
the NL zone on the Transparency Platform, which holds the farms of the transmission grid. Its
potential runs on 0.486 times the capacity of CBS, the share of the CBS output that the series
holds over 2019-2022, and its annual level follows the change of that share (plan, Deviations).
999 bootstrap replicates (seed 2026).

## Estimate

M̂ is the sum, over the N hours, of the calibrated potential p* minus the published output. N
hours have a negative mean day-ahead price in the NL zone and meet every other eligibility rule;
they never enter the fit. The share is M̂ over the potential at the N hours. Estimates below the
minimal detectable effect (MDE) are marked inconclusive.

| Year | Onshore wind M̂, GWh (90 % interval) | Share | MDE, GWh | N hours |
|---|---|---|---|---|
| 2019 | −0.1 (−0.3 to 0.0), inconclusive | −7.0 % | 7.0 | 3 |
| 2020 | −1.5 (−5.7 to 2.7), inconclusive | −1.4 % | 166.9 | 97 |
| 2021 | 6.7 (4.2 to 9.5), inconclusive | 8.4 % | 120.8 | 70 |
| 2022 | 18.1 (14.3 to 22.2), inconclusive | 16.5 % | 158.0 | 85 |
| 2023 | 86.5 (72.1 to 101.0), inconclusive | 16.2 % | 488.5 | 316 |
| 2024 | 95.9 (79.2 to 113.8), inconclusive | 17.4 % | 411.1 | 458 |
| 2025 | 135.1 (116.9 to 154.3), inconclusive | 22.7 % | 494.7 | 584 |

- No year reaches its MDE: under the plan, every Dutch estimate is inconclusive.
- The MDE rests on the daily error of the potential over the eligible hours of clear days with the
  sun at 30° or more: its standard deviation is 2.08 over 123 days of 2023-2025, against 0.32 and
  0.44 for the onshore wind of DK1 and DK2. Half of these days have an error within 19 % of their
  potential, but on calm days the potential of the selected hours is 0.2 to 2 % of the capacity,
  and their errors reach 2 to 22 times it. Without the 22 days whose potential is below 5 % of the
  capacity, the standard deviation would be 0.21 and the MDE about 50, 42 and 50 GWh in 2023,
  2024 and 2025, below the estimates. The plan fixed this rule for every country before any
  result; a rule better suited to wind would enter the plan, for all countries, before it is
  applied.
- The estimate grows from 86 GWh in 2023 to 135 GWh in 2025, 16 to 23 % of the potential at the
  N hours, as these hours grow from 316 to 584.

## Detectability and placebo

| | Onshore wind |
|---|---|
| Minimal detectable effect, 2023, 2024, 2025 | 488.5, 411.1, 494.7 GWh |
| Half-width of the interval, 2023, 2024, 2025 | 17, 18, 14 % |
| Daily error of the potential, standard deviation | 2.078 over 123 days |

Placebo hours are P hours priced 5 to 15 €/MWh, matched to the N hours of the same year on month,
Paris hour and day type, then on the capacity factor of p* and the actual total load of the NL
zone on the Transparency Platform. They leave the fit and should show no gap. Gap as a share of
the potential of the matched hours, with its 90 % interval and the matched hours:

| Year | Onshore wind |
|---|---|
| 2019 | −48.1 (−56.6 to −39.0), 1 hour |
| 2020 | −7.2 (−12.0 to −2.4), 52 |
| 2021 | −4.9 (−9.5 to 0.1), 9 |
| 2022 | 6.8 (0.2 to 14.3), 9 |
| 2023 | 1.6 (−1.5 to 4.6), 53 |
| 2024 | 4.1 (1.0 to 7.0), 78 |
| 2025 | −1.0 (−4.9 to 2.9), 77 |

- As in France, Germany and Denmark, no interval lies within ±2 % of the potential, as the plan
  asks.
- In 2023-2025 the placebo gaps, −1.0 to 4.1 %, are a fraction of the shares held back at the N
  hours, 16 to 23 %.
- A modulation of 30 % injected at the placebo hours is recovered in 3 of 7 years. These hours
  never enter the fit, so the injection repeats the placebo: the years it misses are those whose
  placebo interval leaves zero out.

## Limits

- No published volume of Dutch modulation exists to compare with: these estimates are not
  validated against any reference.
- The series holds the farms of the transmission grid only. The share of the fleet behind it is
  fixed on 2019-2022 and the annual level follows its change from year to year, but a farm that
  joins the grid behind the series within a year moves the level, as the two level breaks of
  [anomalies-2019-2025.md](anomalies-2019-2025.md) show.
- Grid-related curtailment is not netted out: it has no hourly series here.
- 27 to 51 % of the estimate of 2023-2025 comes from strata with fewer than 50 training hours,
  where the stratum ratio is shrunk towards the pooled ratio.
- Solar and offshore wind have no model.
