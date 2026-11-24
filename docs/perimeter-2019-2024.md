# Self-consumption in the published solar series, 2019-2024

Axis 3 of the [evaluation plan](evaluation-plan.md), level evidence: does a published solar series
include the output that producers consume on site? Its yearly level against Eurostat gross
production answers when the self-consumed share is large enough.

```
eu-solar-wind-audit audit-perimeter data/processed/perimeter
```

## Method

Eurostat gross production (dataset `nrg_bal_peh`, updated on 2 June 2026) counts all the output,
self-consumption included. Each published series is set against it, solar against wind:

- the solar ratio is the published output over the gross production inside the perimeter of the
  series, and the wind ratio likewise;
- the gap is the log of the solar ratio over the wind ratio. Wind has no self-consumption, so the
  gap removes what the two ratios share;
- a series that leaves out a self-consumed share s of solar output has a gap of log(1 − s); one
  that includes it has a gap of zero.

The national éCO2mix series cover mainland France without Corsica, while Eurostat's France
includes Corsica and the five overseas departments. Gross production is cut to the share of the
national capacity inside the perimeter (SDES quarterly capacity, mean of the year's quarter ends):
94.1 % of solar capacity in 2019 and 96.4 % in 2024, 99.5 % of wind capacity.

The self-consumed share s comes from estimates that the producers of statistics publish:

- France: the photovoltaic self-consumption of the SDES
  ([L'autoconsommation d'électricité en France](https://www.statistiques.developpement-durable.gouv.fr/lautoconsommation-delectricite-en-france),
  2 October 2026) over Eurostat gross production, 1.0 % in 2019 and 7.1 % in 2024;
- Germany: the estimate of Fraunhofer ISE that Energy-Charts publishes, total minus public solar
  power over total, 6.0 % in 2019 and 19.7 % in 2024.

The error of the gap is measured on the validation series whose solar content is known: éCO2mix,
which includes self-consumption, and SMARD, which leaves it out. Each year gives the gap minus the
gap of the known content. Their root mean square over 2019-2024 is 0.039 (0.050 for éCO2mix,
0.021 for SMARD). With that error, a one-sided test at 5 % detects a self-consumed share with
80 % power from 9.1 %: the detection threshold. A year below it is inconclusive; a year at or above
it goes to the content whose gap is closer to the observed one.

## Results

| Year | éCO2mix gap | s, France | SMARD gap | Energy-Charts total gap | s, Germany | log(1 − s), Germany |
|---|---|---|---|---|---|---|
| 2019 | +0.111 | 1.0 % | −0.048 | +0.018 | 6.0 % | −0.062 |
| 2020 | +0.031 | 1.7 % | −0.062 | −0.001 | 6.3 % | −0.066 |
| 2021 | −0.019 | 2.4 % | −0.083 | −0.021 | 8.4 % | −0.088 |
| 2022 | −0.008 | 3.5 % | −0.098 | +0.004 | 9.2 % | −0.097 |
| 2023 | +0.032 | 4.8 % | −0.152 | −0.018 | 13.2 % | −0.141 |
| 2024 | +0.023 | 7.1 % | −0.172 | +0.006 | 19.7 % | −0.220 |

| Series | Expected content (plan) | Decided years | Content from the level |
|---|---|---|---|
| éCO2mix, national solar | Included | None | Inconclusive |
| SMARD, solar feed-in | Excluded | 2022-2024 | Excluded |
| Energy-Charts, total solar power | Included | 2022-2024 | Included |

- SMARD falls short of Eurostat by the self-consumed share: its gap stays within 0.015 of
  log(1 − s) up to 2023. In 2024 it is −0.172 against −0.220: Energy-Charts puts self-consumption
  at 19.7 % of German solar output, while the SMARD feed-in is 16.2 % below Eurostat's gross
  production.
- Energy-Charts total power, the feed-in plus the ISE estimate, has a gap within 0.021 of zero in
  every year: a German series with the same fleet and weather as SMARD, classified the other way.
- RTE counts self-consumption in its production figures, but the French self-consumed share
  stays below the threshold, so the level cannot confirm it. Without its 2019 error, the largest
  of the twelve, the threshold would fall to 5.5 % and 2024 would count éCO2mix as including
  self-consumption: the French verdict rests on one year either way.
- In 2019, published French solar is 7 % above Eurostat gross production inside its perimeter
  (12.10 TWh against 11.31 TWh), while wind is 4 % below. Eurostat gives 12.03 TWh for the whole
  country, Corsica and the overseas departments included. The gap between the two official
  sources exceeds the French self-consumed share of any year.

## Limits

- Eurostat has not published 2025 yet, and the SDES series ends in 2024.
- The capacity share stands for the production share of Corsica and the overseas departments,
  whose yields differ from those of the mainland.
- The self-consumed shares are estimates of the SDES and of Fraunhofer ISE, not measurements.
- Twelve validation years give an imprecise error, and so an imprecise threshold.
- The level is one type of evidence; the plan decides a cell only when two types agree.
