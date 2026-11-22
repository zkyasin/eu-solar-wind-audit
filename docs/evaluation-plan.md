# Evaluation plan

This plan fixes the validation rules before any result is computed. The git tag `eval-plan-v1`
marks the version written before the first computation. Any later change is recorded in
[Deviations](#deviations) with its date and reason, and the original rule stays visible.

## What is already known

Some reference figures were read before this plan was written:

- the monthly negative-price modulation volumes published by RTE for France (January 2020 to
  October 2026);
- the 2025 market curtailment estimates for Germany in Hirth (2026, arXiv 2609.10053);
- the renewable redispatch volumes published by the Bundesnetzagentur;
- the ratio between Dutch solar generation on the ENTSO-E Transparency Platform and the CBS
  statistics;
- the published self-consumption totals for France (RTE) and Germany (Fraunhofer ISE).

The plan therefore protects analysis choices, not ignorance of these figures. It matters most for
the countries without a reference: Belgium, Spain, Portugal, Italy, Austria and Poland.

## Scope

| Item | Value |
|---|---|
| Countries | FR, DE, BE, NL, DK (DK1 and DK2), ES, PT, IT (bidding zones aggregated), AT, PL |
| Technologies | Solar PV, onshore wind, offshore wind where it exists |
| Period | 2019-01-01 to six days before the run (ERA5T latency) |
| Validation data | Consolidated or definitive series only; provisional months are flagged and excluded |
| Release order | FR, DE, NL and DK complete first; the other six with their evidence level |

## Conventions

- All series are converted to hourly energy in UTC, labelled at the start of the hour.
- éCO2mix consolidated and definitive values are 30-minute averages labelled at the end of the
  period: they are shifted by −30 minutes. Real-time values are 15-minute instantaneous values: an
  hour is the trapezoidal mean of its five points. An hour is kept only if it is complete.
- Daylight saving time: duplicated UTC stamps in spring are dropped, the missing autumn hour stays
  missing and is never interpolated.
- ERA5 radiation is accumulated over the hour ending at the timestamp: it is attributed to
  [t − 1 h, t), and the solar position is computed at the middle of that interval.
- Capacities are expressed in AC, with the source unit and a confidence level recorded.

## Price classes

| Class | Definition | Use |
|---|---|---|
| Negative (N) | Mean of the quarter-hour day-ahead prices < 0 €/MWh | Treated hours; total comparable with RTE |
| Near zero | Mean in [−0.01, 0) €/MWh (counted in N) or in [0, 5) €/MWh (not counted in N) | Reported separately |
| Reference (P) | Mean ≥ 5 €/MWh, no negative quarter-hour, more than one hour from an N hour or a missing hour | Calibration |
| Missing | No price | Excluded |

## Two model objects

The model is never calibrated on the series it is used to judge.

1. **Physics model** (perimeter tests and the high-wind diagnostic). Its few parameters are
   learned on series whose perimeter is known: France on its total series (itself an RTE
   estimate) and Germany on SMARD feed-in plus the published self-consumption (with its
   uncertainty). Denmark is added once the content of its published columns is confirmed. Parameters are transferred to the other countries with
   shrinkage towards the learned values. The DC/AC ratio is estimated per country on clear-sky days
   with positive prices.
2. **Calibrated potential** (modulation and anomaly detection). Two to three parameters per country
   and technology, Huber loss, leave-one-year-out, fitted on P hours only. The published/potential
   ratio is then estimated by stratum (solar elevation × decile of the *modelled* capacity factor ×
   day type), shrunk towards a common mean, plus an annual level. The observed capacity factor is
   never used to define strata, because it depends on modulation.

A test checks that the calibration code never reads the reference series (RTE, BNetzA, Hirth).

## Axis 1: model quality

- Leave-one-year-out per country and technology on 2019-2025; 2026 is used only as a prospective
  test.
- Transfer error of the physics model: leave-one-zone-out over the éCO2mix regions, the four German
  control areas and DK1/DK2.
- Metrics on P hours: hourly correlation, RMSE in capacity factor (AC basis), annual bias, and bias
  by season and sky class.
- Baselines: raw physics with nominal parameters, monthly-hourly climatology, Renewables.ninja and
  PECD 4.2. The last two are calibrated on national production, so they are reported as reference
  points, not as independent truth.
- Reference levels from the literature: hourly wind correlation of about 0.98 with ERA5 in France
  and Germany; solar RMSE of about 0.07 in capacity factor, obtained with SARAH irradiance (ERA5 is
  expected to do worse). These levels are reported, not used to tune the model.

## Axis 2: anomalies in published series

Thresholds are fixed on 2019-2022 before any detector runs on later years.

| Detector | Rule |
|---|---|
| Completeness | Expected grid, duplicates, gaps, resolution changes dated from the source documentation |
| Timestamp | Daily centroid of the published solar curve against the modelled centroid, excluding hours with price ≤ 0; jumps tested at daylight saving dates. Only shifts of one hour or more are claimed |
| Outliers | Robust residual score above the fixed threshold; values above 1.05 × AC capacity; negative values; values frozen for 24 hours or more |
| Level breaks | Change-point detection (PELT) on the log ratio published/physics, penalty set for at most one false alarm per technology and year on 2019-2022. A break is attributed to the data only if it appears in solar and wind at once or matches a gap between sources. ±7 days around the ERA5T/ERA5 junction are excluded |
| Cross-source gaps | ENTSO-E against national sources where both exist |

Evaluation on synthetic defects injected into clean weeks of 2019-2022 (shifts of ±1 h, daylight
saving duplicates, gaps of 1 to 48 hours, zero runs, level steps of −5 % and −10 %):

- recall ≥ 0.9 for shifts of one hour or more, within two valid days;
- level steps of −10 % dated within ±3 days;
- at most one false alarm per technology and year.

Real cases are documented as tickets with the evidence that would be sent to the publisher.

## Axis 3: what each published series contains

Each country and series (production, consumption) gets one cell of this matrix:

| | Distribution grids covered | Not covered |
|---|---|---|
| Self-consumption included | Complete | Partial |
| Self-consumption excluded | Feed-in only | Partial feed-in |

A cell can also be *ambiguous* or *inconclusive*.

Evidence:

1. documentation (ENTSO-E definitions 16.1 and 6.1.a, national documentation);
2. published level against the physics model and against Eurostat gross production, with wind as a
   control (it has no self-consumption);
3. capacity by size or voltage class, which gives the expected ratio under each hypothesis;
4. feed-in signature: in a feed-in-only series, the injected share rises on sunnier days at fixed
   hour and calendar. It is identified from day-to-day sunshine shocks, excluding hours with
   price ≤ 0. Negative control: éCO2mix (self-consumption included) must not show it. Positive
   control: SMARD (feed-in only) must show it;
5. consumption signature: a load series net of self-consumption shows a midday dip on sunny days,
   explained by modelled rooftop PV. Lighting, which also falls on sunny days, is a declared
   confounder.

Decision rules:

- A cell is decided only when at least two independent types of evidence agree and the level gap
  exceeds the detection threshold.
- The detection threshold comes from a power simulation on Germany, whose self-consumption is
  known, given the measured transfer error. Below it, the cell is *inconclusive*.
- The method must classify the validation countries as expected before any other country is
  published:

| Country | Series | Expected cell | Basis |
|---|---|---|---|
| France | Production (éCO2mix solar) | Complete | RTE states that its solar values include self-consumption |
| France | Consumption | Complete | éCO2mix consumption is computed from the balance of production, exchanges and pumping |
| Germany | Production (SMARD) | Feed-in only | SMARD publishes feed-in to the public grid |
| Netherlands | Production (ENTSO-E solar) | Distribution not covered (either row) | About 2 % of the CBS solar total |

Denmark joins the validation set once its expected cell is confirmed from Energinet's
documentation; the confirmation is logged in [Deviations](#deviations).

If a validation country fails, the method is revised, the change is logged, and no other country is
published until the validation passes.

Harmonized shares: monthly shares of solar and wind in consumption use the published load, corrected
to a common perimeter, as denominator. The annual sums of harmonized generation must match Eurostat
gross generation within ±5 % over 2019-2023; otherwise the country is flagged.

## Axis 4: modulation during negative prices

Estimand: M = Σ over N hours of (Y(0) − Y), the output that would have been published without the
price incentive, minus the published output.

Estimator: M̂ = Σ over N hours of (p* − y), with p* the calibrated potential. Sums are signed, never
truncated at zero hour by hour. Results are monthly and annual, on consolidated or definitive data.

Selection on price: day-ahead prices are set from forecasts, so at equal modelled potential, N hours
tend to be hours where the model underestimates output.

- The ENTSO-E day-ahead wind and solar forecast enters the calibration as a covariate, only if it
  passes this check: its ratio to the modelled potential is not lower on N hours than on matched P
  hours. Otherwise it is not used, and the deviation is logged. The forecast is published after the
  day-ahead auction, so it only approximates the information available to the market.
- On low-modulation years, the remaining gap must no longer correlate with the price.
- Contemporaneous placebo: hours of the same year priced 5 to 15 €/MWh, matched on forecast
  capacity factor, residual load, hour, month and day type, are treated as if they were negative.
  The 90 % interval of the placebo gap must lie within ±2 % of their potential.

Inference: stationary bootstrap over days with a mean block length of one week, 999 replicates. Each
replicate redoes the calibration and draws capacity (±10 %) and the DC/AC ratio (1.0 to 1.3), so the
published 90 % interval includes these structural errors. The share of M̂ coming from strata with
fewer than 50 P hours is reported.

Accuracy: injection-recovery. A known modulation is added to published series at the hours that
prices would select, and the full chain is run. Bias and interval coverage are reported; the target
coverage is 85 % to 95 %. This is the only test of accuracy.

Minimal detectable effect: computed per country and technology from the leave-one-year-out error on
clear-sky P hours at high solar elevation. An estimate below it is published as *inconclusive*. If
the minimal detectable effect exceeds 20 % of the expected volume, the annual verdict is announced
as likely inconclusive before computing it.

### France against RTE

The RTE series is itself an estimate with an unpublished method, so agreement is a replication, not
a proof of accuracy. Before the comparison, the perimeter of the RTE wind file is checked: if it
includes offshore wind and offshore output cannot be modelled on the same basis, no wind verdict is
issued.

| Test | Rule |
|---|---|
| Annual ratio | M̂ / RTE within [0.8, 1.25] for 2024 and 2025, per technology |
| Monthly detail | Monthly log error of M̂ lower than that of the prorata baseline (constant rate × potential of N hours, rate estimated leave-one-year-out), on months with more than 20 GWh at RTE and at least 30 N hours |
| Monthly rates | Spearman correlation between estimated and RTE monthly rates ≥ 0.6 on the same months |

Verdict, checked in this order:

1. *inconclusive*: the half-width of the 90 % interval exceeds 25 % of the estimate, or the
   estimate is below the minimal detectable effect;
2. *discordant*: an annual ratio fails;
3. *concordant in volume only*: annual ratios pass, a monthly test fails;
4. *concordant*: all three tests pass.

Germany: the ratio to Hirth's 2025 figures is reported, without a verdict. Other countries: estimates
are published only above the minimal detectable effect, labelled *no reference*.

## Grid curtailment

- Published figures are ordered volumes as reported: Bundesnetzagentur renewable redispatch by
  technology and quarter; Elia and REE volumes once access and licence are confirmed. RTE downward
  adjustments are cited, not labelled grid curtailment, because they mix balancing and congestion.
- Diagnostic: the output deficit at prices ≥ 5 €/MWh during strong local wind, computed with the
  physics model, compared between congested zones and control zones at the same hours. It is
  published only if both checks pass: no deficit beyond ±5 % of the potential on strong-wind
  episodes without redispatch, and recovery of a simulated curtailment by injection-recovery.
  Otherwise it is published as *not identified*.

## Publication rules

- A country's figures appear only after its validation passes; otherwise the page shows its
  evidence level (*no reference*, *inconclusive*, *not published*).
- Hourly curves come only from sources whose licence allows it (ODRÉ, SMARD, and others once
  confirmed). Values derived from ENTSO-E data are published only as monthly aggregates, flags or
  classes, and are not relicensed under CC BY without written confirmation.
- Every published figure carries the model version, weather source, data vintage and git commit.
- The R recomputation of the key statistics must match Python: relative difference ≤ 1e-6 for sums,
  ≤ 1e-3 for regression coefficients and standard errors.

## Deviations

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-11-22 | What is already known | Added: before this entry, the uncalibrated physics model was compared with published French solar and onshore wind for 2024 (annual and monthly ratios), and six days of 2025 appear in a network test. | Disclosure: 2024 and 2025 are not untouched for the choices below. |
| 2026-11-22 | Two model objects | France's DC/AC ratio is reported as the range its clear-sky profile allows (profile cost, minimum across subsets, bootstrap over days), stated as r(1 − losses) and as r at nominal losses. The calibrated potential uses the documented ratio of 1.2 (IEA PVPS, France 2025, table 3). | On clear-sky P hours of 2019-2022 the profile cost changes by less than 3 % between 1.0 and 1.4, and its minimum moves between 1.0 and 1.5 across subsets: the ratio is not identified. |
| 2026-11-22 | Two model objects | Solar calibrated potential: an AC level, the share of the fleet modelled as horizontal modules, and a seasonal term (a sine with its phase at the summer solstice), fitted on P hours with the capacity-weighted sun above the horizon. These forms, the parameter grids and the sky thresholds were chosen on 2019-2022 only. | The plan fixes the number of parameters, not their form. On 2019-2022 the all-south fleet of the physics model misses summer morning and evening output and overstates winter (published/modelled below 5° of sun elevation: 0.31 in winter, 1.31 in summer). |
| 2026-11-22 | Two model objects | Onshore wind calibrated potential: one parameter, the wind speed scale, plus the annual level. | On 2019-2022 a speed offset is not identified next to the scale: the two estimates are correlated at −0.98, and the leave-one-year-out RMSE is 0.0305 with the scale alone against 0.0306 with both. |
| 2026-11-22 | Two model objects | The annual level is the ratio of published output to potential × stratum ratio over the year's eligible P hours. Stratum ratios are shrunk towards the pooled ratio with weight n / (n + 7), n counting distinct days. | The plan names the annual level without defining it. Hours of one day are not independent, and the plan's 50 P hours is a reporting threshold, not a shrinkage constant. |
| 2026-11-22 | Axis 1: model quality | Onshore wind folds are 2019, 2020, 2021, January to May 2022, 2024 and 2025. The reference is national wind before 2022-06-01 (Europe/Paris) and the sum of regional onshore wind from 2024-01-01. | The first French offshore farm injected power in early June 2022 (EDF; AFP, 2022-06-10), and onshore and offshore wind are published separately only from 2024. |
| 2026-11-22 | Axis 1: model quality | Metrics are reported separately for the 2019-2022 folds, whose years chose the model forms, and the 2023-2025 folds; the minimal detectable effect uses the latter. Axis 1 also reports the potential with its annual level cross-fitted within the year: each held-out week takes the level of the year's other weeks. | The 2019-2022 folds are optimistic by construction, and axis 4 uses the potential with its annual level. |
| 2026-11-22 | Axis 1: model quality | Clear-sky days for the DC/AC ratio: a fleet clear-sky index of at least 0.9 (ERA5 irradiance against the Ineichen clear sky), and every hour within 3 h of solar noon in class P. | Price selection removes most sunny peak hours in 2024 and 2025 (share of P hours with the sun above 40°: 0.56 and 0.42); partial days would tilt the sample to morning and evening hours. |
| 2026-11-22 | Axis 1: model quality | The final fit is committed and tagged before 2026 is evaluated, and 2026 is evaluated only from that commit, month by month once SDES publishes final capacity. | 2026 consolidated data are already public; only a frozen fit keeps the test prospective. |
| 2026-11-22 | Axis 4: modulation during negative prices | The DC/AC draw of the bootstrap stays 1.0 to 1.3; a sensitivity with 1.0 to 1.5 is reported next to it. | The French clear-sky profile leans towards 1.35-1.40 without excluding lower values (see the DC/AC entry above). |
| 2026-11-22 | Axis 2: anomalies in published series | The false-alarm budget applies per published series, technology and year, all statistical detectors together; counts per country, technology and year are reported too. Rule-based findings (completeness, negative values, values above capacity, frozen values, totals against their parts) are facts and are not budgeted. | The plan does not say over which series or detectors the budget runs. |
| 2026-11-22 | Axis 2: anomalies in published series | A timestamp shift is claimed only when the two deciding days are off by 45 min or more on average; its size is the run's median offset rounded to whole hours, and smaller offsets stay unpublished diagnostics. The synthetic evaluation adds ±30 min shifts (±15 min for quarter-hour sources), which must give no whole-hour claim beyond the budget. A series is tested for shifts only if it meets the budget on 2019-2022. | With a 30 min rule, a ±30 min relabelling of French national solar over 2019-2022 gave 141 and 146 one-hour claims. Timestamp noise ranges from 4.6 min (national) to 11.5 min (Île-de-France). |
| 2026-11-22 | Axis 2: anomalies in published series | The −10 % step criterion keeps a recall target of 0.9 within ±3 days. If missed, it is reported as failed, with the smallest step that reaches the target. | On 2019-2022 the ERA5-based ratio dates −10 % steps within ±3 days with recall 0.14-0.20 for solar and 0.34-0.38 for wind; the target is not restated after this measurement. |
| 2026-11-22 | Axis 2: anomalies in published series | Detectors that compare published wind with the onshore model skip hours that contain offshore output (national and Pays de la Loire wind from 2022-06-01, other regions from their first offshore output) until an offshore model or a split exists. Level breaks are never searched across the switch from wind to onshore wind on 2024-01-01. | Published wind includes offshore farms from June 2022, which the onshore model does not cover. |
| 2026-11-22 | Axis 2: anomalies in published series | The centroid, the residual score and the level-break ratio use P hours only. Statistical detectors run on consolidated and definitive hours; real-time hours get the rule-based detectors only and are labelled provisional. | Near-zero and buffer hours can be modulated too, and the Scope excludes provisional data from validation. |
| 2026-11-22 | Axis 2: anomalies in published series | Thresholds are fixed and tagged per country before that country's first run on later years. The thresholds file records its reference model (commit, parameters, SDES vintage, ERA5 store); a new reference gets a new file, fixed on 2019-2022 only, and earlier results stay published. | Adding a country or replacing the reference must not move thresholds after results on later years are known. |
| 2026-11-22 | Axis 2: anomalies in published series | Until ENTSO-E access is granted, cross-source checks compare national sources only: éCO2mix national against the sum of its regions, and SMARD Germany against its control areas. Energy-Charts public power is reported against SMARD but not counted as independent evidence. | Over 2019-2022 Energy-Charts public power equals SMARD times a factor that is constant within each month for solar and within each year for wind. |
