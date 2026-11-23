# Anomalies in published solar and wind series, 2019-2025

The detectors of axis 2 of the [evaluation plan](evaluation-plan.md) were run on the hourly solar
and wind series of mainland France and Germany, over 1 January 2019 to 31 December 2025:

```
eu-solar-wind-audit detect-anomalies FR 2019-01-01 2026-01-01 data/processed/anomalies-fr
eu-solar-wind-audit detect-anomalies DE 2019-01-01 2026-01-01 data/processed/anomalies-de
```

France uses the thresholds of `config/anomaly-thresholds/FR.json`, fixed on 2019-2022 (tag
`anomaly-thresholds-fr-v1`). Germany has no physics model yet, so it gets the rule-based
detectors only. Each run writes its findings, the level breaks left unattributed, and a JSON file
with the SHA-256 of every input series.

A finding is not a defect. The cases below were checked against the raw files of the publisher
and are written as the tickets that would be sent to it. The other findings follow, with the
reason they are not attributed to the published data.

## Findings by kind

| Kind | France (éCO2mix) | Germany (SMARD, Energy-Charts) |
|---|---|---|
| gap | 26 | 7 |
| negative | 30 | 0 |
| above_capacity | 71 | 0 |
| frozen | 31 | 3 |
| parts_gap | 19 | 51 |
| cross_source_gap | 0 | 22 |
| timestamp_shift | 5 | - |
| clock_change_jump | 1 | - |
| residual_outlier | 5 | - |
| level_break | 4 | - |
| clock_change_convention | 2 | 0 |
| Level breaks left unattributed | 10 | - |

Times are in UTC unless a time zone is given.

## Checked cases

### FR-1. The half-hour ending at midnight on 31 December 2025 is empty

- Datasets: ODRÉ `eco2mix-national-cons-def` and `eco2mix-regional-cons-def`.
- The row stamped 2026-01-01 00:00 (Europe/Paris), which closes the half-hour from 23:30 to 24:00
  on 31 December 2025, has no value for consumption, solar or wind, nationally and in each of
  the 12 regions. It is the first row with the status "consolidated" after the definitive rows
  of 2025. The rows that close the previous years, from 2019 to 2024, all have values.
- Effect: the hour from 23:00 to 24:00 (Paris) on 31 December 2025 is incomplete in the 26 solar
  and wind series (26 `gap` findings).

### FR-2. Grand Est solar, 2019: about 1.7 times its later level

- Series: Grand Est solar, `eco2mix-regional-cons-def`.
- Published over modelled output: 1.32 in 2019, 0.76 in 2020, 0.75 in 2021. Published energy:
  998 GWh in 2019, 656 GWh in 2020, although the SDES capacity of the region grew from 516 MW to
  668 MW over these two years.
- In 174 hours of 2019 the published output exceeds 1.05 times the SDES capacity, up to 1.62
  times (913.5 MWh in an hour for 564 MW).
- The level-break search finds a step of −44 % on 1 January 2020 in this series only.

### FR-3. Grand Est solar, 30 June 2025: 2.4 times the modelled output

- Series: Grand Est solar, `eco2mix-regional-cons-def`.
- On 30 June 2025 the published output is 33.6 GWh, 2.40 times the modelled output, against 0.98
  on the days from 22 June to 8 July: about 20 GWh more than the level of the surrounding days.
  The hour from 13:00 to 14:00 (Paris) holds 3,816.5 MWh for an SDES capacity of 2,278 MW.
- National solar, which equals the sum of the regions, carries the excess: 1.06 times the
  modelled output that day, against 0.87 around it.

### FR-4. Nouvelle-Aquitaine wind, January 2023: about half its level

- Series: Nouvelle-Aquitaine wind, `eco2mix-regional-cons-def`.
- Published over modelled output by month: 0.78 in November 2022, 0.83 in December, 0.47 in
  January 2023, 0.96 in February, 0.84 in March. In January 2023 the neighbouring regions do not
  move: Occitanie 0.90, Pays de la Loire 1.21.
- Published in January 2023: 231 GWh; at the median ratio of the surrounding months, about
  414 GWh.
- The joint search of the region's solar and wind finds a drop on 2 January 2023 (wind −48 %,
  solar −17 %) and its reversal on 31 January.
- National wind equals the sum of the regions over this month (no `parts_gap`), so it carries the
  deficit.

### FR-5. Île-de-France wind above the SDES capacity, 2022-2025

- Series: Île-de-France wind (and onshore wind from 2024), `eco2mix-regional-cons-def`, against
  the SDES quarterly capacity of the region.
- The SDES capacity stays at 127.3 MW from the end of 2021 to early 2025, while the published
  output reaches 144.5 to 145 MWh in an hour. Hours above 1.05 times the capacity: 205 in 2022,
  211 in 2023, 164 in 2024, 56 in 2025; at most 1.14 times.
- Either a farm is missing from the SDES capacity of the region, or éCO2mix counts in
  Île-de-France output that SDES places in another region.

### FR-6. Hauts-de-France solar, 3 January 2025: one hour four times its neighbours

- Series: Hauts-de-France solar, `eco2mix-regional-cons-def`.
- The hour from 12:00 to 13:00 (Paris) holds 1,125.5 MWh, against 254.5 and 383.5 MWh in the
  hours around it, 451 MWh modelled and 781 MW of SDES capacity (1.44 times).

### FR-7. National solar below zero at night until 2020

- Series: national solar, `eco2mix-national-cons-def`.
- 2,368 hours of 2019 and 2,227 hours of 2020 are below zero, down to −12 MWh, nearly all at
  night; none from 2021 on. Nouvelle-Aquitaine solar has 2,898 such hours in 2019.
- The series seem to have counted the night consumption of installations until 2020. The
  change is not stated in the dataset description.

### DE-1. TransnetBW solar and onshore wind published as zero, December 2025

- Series: SMARD feed-in, filters 4068 (solar) and 4067 (onshore wind), region TransnetBW,
  quarter-hours.
- Solar is 0.0 from 2 December 16:00 to 11 December 07:00 (207 hours), onshore wind from
  2 December 23:00 to 10 December 23:00 (192 hours), and onshore wind on 18-19 November from 23:00
  to 23:00 (24 hours). The values are present and equal to zero.
- The series of Germany keeps the output of the area: on the Berlin days from 3 to 10 December,
  Germany exceeds the sum of its four control areas by 4.3 to 15.5 GWh a day for solar, and by
  14.7 GWh on 19 November for onshore wind. On the other days the two agree.

### DE-2. Germany against the sum of its control areas, May 2022 and August 2025

- Series: SMARD feed-in of Germany and of 50Hertz, Amprion, TenneT and TransnetBW.
- Solar, August 2025: on most days Germany is below the sum of its areas, by up to 5.9 GWh a
  day; July and September agree.
- Onshore wind, 10 to 28 May 2022: on most days Germany is above the sum, by up to 4.9 GWh a
  day.

### DE-3. Amprion solar, 12 June to 14 July 2025: about 50 MWh an hour at night

- Series: SMARD feed-in, filter 4068 (solar), region Amprion.
- At night the series holds 2 to 5 MWh an hour before 12 June and after 14 July 2025, and about
  50 MWh an hour in between (Berlin days). Germany carries it: 55 to 59 MWh an hour at night,
  against 5 to 14 MWh outside these dates.
- Energy-Charts public solar does not carry it, which is how the run found it: 232 hours of
  departure from SMARD between 11 June and 14 July, at the first hours of daylight.

## Findings not attributed to the published data

- Zero runs of offshore wind (31 `frozen`): 24 to 96 hours at exactly zero in Normandie (7),
  Pays de la Loire (3), Bretagne (5) and Provence-Alpes-Côte d'Azur (16), from 2024. Without an
  offshore model they cannot be told apart from outages of the farms.
- Offshore output above the SDES capacity at commissioning (`above_capacity`): the floating farm
  of Provence-Alpes-Côte d'Azur injects from 19 November 2024 while SDES counts it from the first
  quarter of 2025; the second farm of Pays de la Loire injects from July 2025 (up to 736.5 MWh in
  September for 490 MW) while SDES adds it from the fourth quarter. These findings show when
  SDES counts a farm, not an error in the output.
- Level breaks left unattributed (10): steps of a single technology, or near the first days of
  2019, such as Hauts-de-France solar +27 % on 27 September 2021 and Auvergne-Rhône-Alpes wind on
  15 January 2019. A step of one technology can come from the published series or from the
  model inputs, capacity or weather, so it is not claimed.
- Timestamp shifts (5), a clock-change jump (1) and residual outliers (5): three shifts and the
  jump are in Île-de-France solar, a small series (106 MW in 2019, 529 MW at the end of 2025)
  whose daily timing is noisy. No shift lasts more than four valid days.
- National wind against the sum of its regions (19 `parts_gap`): gaps of 10 to 24 MWh at hours of
  low wind, just above the rounding of 13 published values (6.5 MWh).
- Energy-Charts against SMARD (22 `cross_source_gap`): besides DE-3, offshore wind in 2019 and
  2022, and four solar hours at night in February and March 2025. Energy-Charts public power
  follows SMARD up to a monthly factor, so these departures are not independent evidence.
- SMARD gaps (7): empty quarter-hours of TransnetBW and 50Hertz in July and December 2022.
- éCO2mix clock changes (2 `clock_change_convention`): the datasets publish the same rows for
  every local day, as documented, which repeats the hour after each spring change and leaves out
  the hour before each autumn change.

## Limits

- Statistical detectors run on France only, and their thresholds rest on 2019-2022.
- A finding against the SDES capacity depends on its quarterly figures, which this project
  interpolates linearly between quarter ends.
- No case has been reported to its publisher yet.
