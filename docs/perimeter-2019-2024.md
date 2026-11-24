# What the published series contain, 2019-2024

Axis 3 of the [evaluation plan](evaluation-plan.md): does a published solar and wind series
include the output that producers consume on site, and does it cover the plants connected to the
distribution grids? Each question is answered from two types of evidence, the yearly level of the
series against Eurostat gross production and what its producer states, and decided only when they
agree.

```
eu-solar-wind-audit audit-perimeter data/processed/perimeter
```

## Matrix

| Series | Self-consumption: level | Producer | Decided | Distribution grids: level | Producer | Decided | Cell | Expected cell | Validation |
|---|---|---|---|---|---|---|---|---|---|
| SMARD, Germany | Excluded | Excluded | Excluded | Covered | Covered | Covered | Feed-in only | Feed-in only | Pass |
| Energy-Charts total power, Germany | Included | Included | Included | Covered | Covered | Covered | Complete | Complete | Pass |
| éCO2mix, mainland France | Inconclusive | Included | Inconclusive | Covered | Covered | Covered | Inconclusive | Complete | Inconclusive |

- The method sets two German series with the same fleet and weather in different cells: the SMARD
  feed-in leaves self-consumption out, Energy-Charts total power adds it back.
- France covers its distribution grids, but its self-consumed share stays below the detection
  threshold in every year: the level cannot confirm what RTE states, so the cell stays
  inconclusive rather than complete.
- No validation series is placed in a wrong cell. The Netherlands, the validation case of a series
  without its distribution grids, waits for access to the ENTSO-E Transparency Platform.
- The feed-in signature, a third type of evidence, fails its positive control and is not used
  ([Feed-in signature](#feed-in-signature)).

## What the producers state

Each statement is quoted from the document listed, with its SHA-256, in
`references/documentation/manifest.json`.

| Series | Dimension | Statement |
|---|---|---|
| SMARD | Self-consumption: excluded | « Ebenfalls umfassen die Daten nur die Einspeisungen in das Netz der allgemeinen Versorgung. » (SMARD handbook, November 2026, section 1.1) |
| SMARD | Distribution: covered | « Die Realisierte Erzeugung in Relation zur installierten Leistung […] wird jedoch vollständig über alle Regelzonen hinweg abgedeckt (Abdeckungsgrad 100%). » (same section) |
| Energy-Charts total power | Self-consumption: included | « Die gesamte Nettostromerzeugung beinhaltet neben der öffentlichen Nettostromerzeugung auch den solaren Selbstverbrauch […] » (Fraunhofer ISE, Stromerzeugung in Deutschland im Jahr 2025, page 4) |
| Energy-Charts total power | Distribution: covered | « Total net electricity production (including industrial self supply) […] » (Energy-Charts API description, `/v2/total_power`) |
| éCO2mix | Self-consumption: included | « la production nette nationale (toute filière confondue, nette de la consommation des auxiliaires et avec prise en compte de l’autoproduction/autoconsommation) » (RTE, Bilan électrique 2025, page 12) |
| éCO2mix | Distribution: covered | « la visibilité de RTE sur les injections de productions à partir de comptages RPT et RPD permet de constituer de manière plus précise les rubriques » (éCO2mix file specification, June 2025, section 1.2.2) |

RTE's statement covers national net production in its annual review; the review gives 32.9 TWh of
solar output in 2025, and the éCO2mix national series 32.5 TWh without Corsica.

## Level against Eurostat

Eurostat gross production (dataset `nrg_bal_peh`, updated on 2 June 2026) counts all the output,
self-consumption included. Each published series is set against it, solar against wind: the solar
ratio is the published output over the gross production inside the perimeter of the series, and
the wind ratio likewise.

The national éCO2mix series cover mainland France without Corsica, while Eurostat's France
includes Corsica and the five overseas departments. Gross production is cut to the share of the
national capacity inside the perimeter (SDES quarterly capacity, mean of the year's quarter ends):
94.1 % of solar capacity in 2019 and 96.4 % in 2024, 99.5 % of wind capacity.

Each dimension has its own statistic, error and detection threshold. The error is the root mean
square, over 2019-2024, of the statistic of the series whose content is known, éCO2mix and SMARD,
minus its value for that content. A one-sided test at 5 % detects a share left out of a series
with 80 % power from the detection threshold. A year below it is inconclusive; a year at or above
it goes to the content whose value is closer to the observed one, and a series takes the content
that its decided years agree on.

### Self-consumption

Wind has no self-consumption, so the gap, the log of the solar ratio over the wind ratio, removes
what the two ratios share. A series that leaves out a self-consumed share s of solar output has a
gap of log(1 − s); one that includes it has a gap of zero. The self-consumed share s comes from
estimates that the producers of statistics publish:

- France: the photovoltaic self-consumption of the SDES
  ([L'autoconsommation d'électricité en France](https://www.statistiques.developpement-durable.gouv.fr/lautoconsommation-delectricite-en-france),
  2 October 2026) over Eurostat gross production, 1.0 % in 2019 and 7.1 % in 2024;
- Germany: the estimate of Fraunhofer ISE that Energy-Charts publishes, total minus public solar
  power over total, 6.0 % in 2019 and 19.7 % in 2024.

The error of the gap is 0.039 (0.050 for éCO2mix, 0.021 for SMARD), and the detection threshold
9.1 %.

| Year | éCO2mix gap | s, France | SMARD gap | Energy-Charts total gap | s, Germany | log(1 − s), Germany |
|---|---|---|---|---|---|---|
| 2019 | +0.111 | 1.0 % | −0.048 | +0.018 | 6.0 % | −0.062 |
| 2020 | +0.031 | 1.7 % | −0.062 | −0.001 | 6.3 % | −0.066 |
| 2021 | −0.019 | 2.4 % | −0.083 | −0.021 | 8.4 % | −0.088 |
| 2022 | −0.008 | 3.5 % | −0.098 | +0.004 | 9.2 % | −0.097 |
| 2023 | +0.032 | 4.8 % | −0.152 | −0.018 | 13.2 % | −0.141 |
| 2024 | +0.023 | 7.1 % | −0.172 | +0.006 | 19.7 % | −0.220 |

- SMARD falls short of Eurostat by the self-consumed share: its gap stays within 0.015 of
  log(1 − s) up to 2023. In 2024 it is −0.172 against −0.220: Energy-Charts puts self-consumption
  at 19.7 % of German solar output, while the SMARD feed-in is 16.2 % below Eurostat's gross
  production. Its years are decided from 2022.
- Energy-Charts total power, the feed-in plus the ISE estimate, has a gap within 0.021 of zero in
  every year, decided from 2022.
- The French self-consumed share stays below the threshold in every year. Without its 2019 error,
  the largest of the twelve, the threshold would fall to 5.5 % and 2024 would count éCO2mix as
  including self-consumption: the French verdict rests on one year either way.
- In 2019, published French solar is 7 % above Eurostat gross production inside its perimeter
  (12.10 TWh against 11.31 TWh), while wind is 4 % below. Eurostat gives 12.03 TWh for the whole
  country, Corsica and the overseas departments included. The gap between the two official
  sources exceeds the French self-consumed share of any year.

### Distribution grids

A series that leaves out the plants connected to the distribution grids, where a share d of wind
capacity is connected, has a wind ratio of about 1 − d; one that covers them has a wind ratio of
one. The share d at the end of 2024 is 86.2 % in mainland France without Corsica (national
register of production installations on ODRÉ: wind capacity outside the grid of RTE) and 84.3 % in
Germany (Marktstammdatenregister: onshore wind below extra-high voltage; offshore wind counts as
transmission, since the transmission system operators connect it). The error of the log wind ratio
is 0.017 (0.021 for éCO2mix, 0.010 for SMARD), and the detection threshold 4.1 %.

| Year | éCO2mix wind ratio | SMARD wind ratio | Energy-Charts total wind ratio |
|---|---|---|---|
| 2019 | 0.958 | 0.986 | 0.981 |
| 2020 | 0.979 | 0.984 | 0.981 |
| 2021 | 0.983 | 0.995 | 1.001 |
| 2022 | 0.996 | 1.007 | 0.977 |
| 2023 | 0.993 | 1.005 | 0.984 |
| 2024 | 0.990 | 0.995 | 0.981 |

Every year of every series is decided as covering the distribution grids: a series without them
would give a wind ratio near 0.15.

## Feed-in signature

```
eu-solar-wind-audit feed-in-signature data/processed/signature
```

A producer that consumes part of its solar output on site feeds in what its consumption leaves
over. At the same hour and date its consumption barely depends on the sky, so it feeds in a larger
share of its output on a clear day than on an overcast one. A feed-in series should therefore rise
more than the solar potential from overcast to clear days, and a series of the whole output should
not (evidence 4 of the plan).

The contrast of a series is the clear minus overcast difference of log(published / potential)
within cells of the same local hour, month, year and day type, over the eligible hours of the
2019-2025 calibration, all priced 5 €/MWh or more. The potential is the physics model with nominal
parameters, never fitted to a series, and the sky class of a day comes from the fleet clear-sky
index (clear from 0.9, overcast below 0.6). The fit has one effect per cell and standard errors
clustered by day. A series shows the signature when the 90 % interval of its contrast lies above
zero; the evidence counts only if éCO2mix, the negative control, does not show it and SMARD, the
positive control, does.

| Hours | Season | éCO2mix | SMARD |
|---|---|---|---|
| All eligible hours | Year | +0.006 (−0.008 to +0.021) | −0.035 (−0.054 to −0.016) |
| | Winter | −0.062 (−0.095 to −0.029) | −0.071 (−0.135 to −0.007) |
| | Spring | +0.060 (+0.033 to +0.088) | +0.039 (+0.010 to +0.068) |
| | Summer | +0.061 (+0.034 to +0.089) | −0.009 (−0.026 to +0.007) |
| | Autumn | −0.007 (−0.033 to +0.019) | −0.076 (−0.111 to −0.041) |
| Modelled capacity factor from 0.05 | Year | +0.011 (−0.004 to +0.025) | +0.029 (+0.011 to +0.047) |

- The positive control fails. Over all eligible hours, SMARD is lower against the model on clear
  days than on overcast ones (−0.035), the opposite of the signature, so the evidence is not used.
- The negative control holds over the year only. éCO2mix includes self-consumption, yet its
  contrast runs from −0.062 in winter to +0.061 in summer: errors of the physics model that depend
  on the sky and the season. On this model, a contrast of about 0.06 cannot be told from
  self-consumption.
- Without the hours whose modelled capacity factor is below 0.05, mostly the first and last hours
  of daylight, SMARD turns positive (+0.029) and éCO2mix stays near zero over the year, but the
  French seasons still run from −0.055 to +0.064. The verdict of the positive control turns on
  hours that the plan does not single out.
- The SMARD handbook lists the solar values of Amprion, TenneT and TransnetBW as extrapolations
  (Hochrechnung), and those of 50Hertz as measurements and extrapolations (annex, table 4, coverage
  of January 2026). Its hourly shape comes from an extrapolation and need not follow the feed-in of
  each plant.
- Energy-Charts total power has the contrast of SMARD to the third decimal. Within each month it is
  the SMARD feed-in times one factor (from 1.04 to 1.75 over 2019-2025), so it holds
  self-consumption in its monthly and yearly sums, not hour by hour.

## Limits

- Eurostat has not published 2025 yet, and the SDES series ends in 2024.
- The capacity share stands for the production share of Corsica and the overseas departments,
  whose yields differ from those of the mainland, and the capacity share on the distribution
  grids for their share of wind output.
- The self-consumed shares are estimates of the SDES and of Fraunhofer ISE, not measurements.
- Twelve validation years give imprecise errors, and so imprecise thresholds.
- The consumption signature, the last type of evidence of the plan, is not used yet.
