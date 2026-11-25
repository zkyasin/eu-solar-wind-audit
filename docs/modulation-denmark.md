# Output modulated during negative prices, Denmark, 2019-2025

Axis 4 of the [evaluation plan](evaluation-plan.md) for the two Danish price areas: how much solar
and onshore wind output was held back in the hours of negative day-ahead prices, estimated from a
weather-based potential. No published volume of Danish modulation is known, so the estimates carry
no reference and, as the plan says, are published only where they reach their minimal detectable
effect; below it they are inconclusive.

```
eu-solar-wind-audit estimate-modulation DK1 solar data/processed/modulation-dk1
eu-solar-wind-audit estimate-modulation DK1 wind_onshore data/processed/modulation-dk1
eu-solar-wind-audit estimate-modulation DK2 solar data/processed/modulation-dk2
eu-solar-wind-audit estimate-modulation DK2 wind_onshore data/processed/modulation-dk2
```

The estimates use the fits frozen in `fits/DK1` and `fits/DK2` (tag `calibration-dk-v1`), the
hourly settlement data of each price area on Energi Data Service, solar with the self-consumed
output that Energinet estimates, and 999 bootstrap replicates (seed 2026).

## Estimate

M̂ is the sum, over the N hours, of the calibrated potential p* minus the published output. N
hours have a negative mean day-ahead price in the price area and meet every other eligibility
rule; they never enter the fit. The share is M̂ over the potential at the N hours. Estimates below
the minimal detectable effect (MDE) are marked inconclusive.

DK1:

| Year | Solar M̂, GWh (90 % interval) | Share | MDE, GWh | Onshore wind M̂, GWh (90 % interval) | Share | MDE, GWh |
|---|---|---|---|---|---|---|
| 2019 | 0.6 (0.2 to 1.0), inconclusive | 5.1 % | 0.8 | 23.2 (12.1 to 33.5), inconclusive | 7.2 % | 71.7 |
| 2020 | 1.6 (0.7 to 2.6), inconclusive | 5.8 % | 1.7 | 14.3 (2.0 to 26.0), inconclusive | 3.6 % | 75.5 |
| 2021 | 2.5 (1.6 to 3.2) | 9.5 % | 1.7 | 1.7 (−4.0 to 7.4), inconclusive | 1.1 % | 43.0 |
| 2022 | −0.2 (−1.2 to 0.9), inconclusive | −1.0 % | 2.2 | 1.0 (−2.9 to 5.0), inconclusive | 1.2 % | 24.7 |
| 2023 | 44.4 (38.8 to 49.5) | 21.7 % | 8.9 | 137.5 (115.4 to 158.4) | 22.1 % | 106.4 |
| 2024 | 120.0 (106.0 to 132.6) | 27.6 % | 14.1 | 143.3 (123.4 to 162.5) | 23.5 % | 78.9 |
| 2025 | 298.8 (272.6 to 323.8) | 41.7 % | 19.5 | 164.3 (141.0 to 186.5) | 26.1 % | 77.0 |

DK2:

| Year | Solar M̂, GWh (90 % interval) | Share | MDE, GWh | Onshore wind M̂, GWh (90 % interval) | Share | MDE, GWh |
|---|---|---|---|---|---|---|
| 2019 | 0.0 (−0.1 to 0.1), inconclusive | −1.1 % | 0.3 | 0.3 (−1.9 to 2.5), inconclusive | 0.6 % | 17.9 |
| 2020 | 0.3 (−0.1 to 0.6), inconclusive | 3.2 % | 0.8 | 1.2 (−0.2 to 2.5), inconclusive | 3.0 % | 13.5 |
| 2021 | 0.3 (0.2 to 0.4), inconclusive | 11.8 % | 0.4 | 0.9 (0.4 to 1.5), inconclusive | 8.1 % | 6.3 |
| 2022 | 0.1 (0.0 to 0.2), inconclusive | 8.3 % | 0.2 | 0.2 (0.1 to 0.3), inconclusive | 9.6 % | 1.5 |
| 2023 | −0.1 (−1.6 to 1.4), inconclusive | −0.2 % | 3.0 | 20.8 (16.4 to 25.4), inconclusive | 20.9 % | 29.4 |
| 2024 | 2.7 (−1.2 to 6.4), inconclusive | 2.1 % | 5.4 | 19.7 (16.6 to 22.7) | 24.2 % | 16.9 |
| 2025 | 10.5 (5.0 to 16.0) | 7.2 % | 5.4 | 9.6 (7.4 to 12.0), inconclusive | 15.9 % | 12.0 |

N hours with the sun up, solar: 53, 110, 67, 28, 181, 329 and 417 in DK1 from 2019 to 2025, and
25, 51, 12, 2, 128, 246 and 251 in DK2. Onshore wind: 132, 192, 87, 38, 281, 375 and 441 in DK1,
and 95, 89, 24, 5, 231, 275 and 265 in DK2.

- DK1 solar holds back a growing share of its potential at negative prices: 22 % in 2023, 28 % in
  2024 and 42 % in 2025, about 0.3 TWh that year. DK1 holds 3.7 of the 5.1 GW of Danish solar
  capacity at the end of 2025, and IEA PVPS (country page of Denmark) puts the growth of the market
  in ground-mounted plants of 50 MW to more than 200 MW. A share this large fits plants that stop
  when the price turns negative, but the estimate cannot tell that cause from grid curtailment.
- DK1 onshore wind holds back 22 to 26 % of its potential at the N hours of 2023-2025, 137 to
  164 GWh a year, against 15 to 19 % for German onshore wind over the same years.
- DK2, with less capacity and fewer N hours, reaches its MDE in two cases only: solar in 2025
  (10.5 GWh, 7 %) and onshore wind in 2024 (19.7 GWh, 24 %).

## Detectability and placebo

| | DK1 solar | DK1 onshore wind | DK2 solar | DK2 onshore wind |
|---|---|---|---|---|
| Minimal detectable effect, 2023, 2024, 2025 | 8.9, 14.1, 19.5 GWh | 106.4, 78.9, 77.0 GWh | 3.0, 5.4, 5.4 GWh | 29.4, 16.9, 12.0 GWh |
| Half-width of the interval, 2023, 2024, 2025 | 12, 11, 9 % | 16, 14, 14 % | 1,704, 142, 52 % | 21, 15, 24 % |
| Daily error of the potential, standard deviation | 0.089 over 97 days | 0.319 over 93 days | 0.103 over 153 days | 0.441 over 171 days |

Placebo hours are P hours priced 5 to 15 €/MWh, matched to the N hours of the same year on month,
Paris hour and day type, then on the capacity factor of p* and the gross consumption of the price
area. They leave the fit and should show no gap. Gap as a share of the potential of the matched
hours, with its 90 % interval:

| Year | DK1 solar | DK1 onshore wind | DK2 solar | DK2 onshore wind |
|---|---|---|---|---|
| 2019 | −2.7 (−9.9 to 5.9), 6 hours | −0.9 (−5.2 to 3.0), 26 | −10.5 (−19.2 to −1.1), 3 | −1.9 (−6.2 to 2.3), 23 |
| 2020 | 2.3 (−1.4 to 6.5), 80 | 0.9 (−2.6 to 4.0), 142 | −11.3 (−16.7 to −6.4), 46 | −2.4 (−6.2 to 1.0), 84 |
| 2021 | 13.0 (5.7 to 20.0), 11 | −17.0 (−22.1 to −12.1), 13 | 39.3 (29.5 to 48.3), 1 | −5.2 (−12.9 to 1.7), 6 |
| 2022 | −6.9 (−10.3 to −2.3), 6 | −12.9 (−18.1 to −6.9), 12 | no match | no match |
| 2023 | −1.6 (−4.8 to 1.7), 51 | 0.1 (−3.8 to 4.1), 80 | −0.2 (−4.9 to 4.5), 41 | 4.8 (−0.5 to 10.4), 80 |
| 2024 | −6.0 (−8.8 to −3.3), 69 | 0.2 (−2.7 to 2.9), 80 | 0.8 (−1.7 to 3.7), 55 | −0.2 (−3.5 to 3.3), 63 |
| 2025 | 2.1 (−0.8 to 4.7), 83 | −2.7 (−6.3 to 0.7), 89 | −3.7 (−7.0 to −0.6), 80 | −7.4 (−11.1 to −3.5), 82 |

- As in France and Germany, no interval lies within ±2 % of the potential, as the plan asks.
- In 2023-2025, the placebo gaps of DK1 solar (−6.0 to 2.1 %) and onshore wind (−2.7 to 0.2 %)
  are a fraction of the shares held back at the N hours (22 to 42 % and 22 to 26 %).
- A modulation of 30 % injected at the placebo hours is recovered in 4 of 7 years for DK1 solar,
  5 of 7 for DK1 wind, 2 of 6 for DK2 solar and 5 of 6 for DK2 wind. These hours never enter the
  fit, so the injection repeats the placebo: the years it misses are those whose placebo interval
  leaves zero out.

## Limits

- No published volume of Danish modulation exists to compare with: these estimates are not
  validated against any reference.
- Grid-related curtailment is not netted out: it has no hourly series by technology here.
- Solar includes the self-consumed output that Energinet estimates, whose timing is wrong from 17
  June to 14 August 2020 (DK-2 in [anomalies-2019-2025.md](anomalies-2019-2025.md)).
- 29 to 64 % of the DK1 onshore wind estimate of 2023-2025 comes from strata with fewer than 50
  training hours, where the stratum ratio is shrunk towards the pooled ratio; 1 to 8 % for DK1
  solar.
- Offshore wind, published apart, has no model.
