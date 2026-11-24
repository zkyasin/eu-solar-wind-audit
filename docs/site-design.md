# Design of the web page

The page in `site/` reads like an instrument at night: a night blue paper, quiet axes, and a few
lights for the data. Cool blue is what grid operators publish, amber is what the weather model
says the fleet could produce. Everything the page shows comes from the documents that
`eu-solar-wind-audit export-site-data` writes in `site/data`; the text quotes the results
documented in `docs/`.

The page is static: HTML, one stylesheet and native ES modules in `site/js`, with no build step
and no third-party script. The charts are drawn in SVG by `site/js/plot.js` and the module of each
chapter. `tests/test_site.py` checks that every file the page asks for exists.

## Color

Tokens are custom properties at the top of `site/style.css`; the charts read them from there.

| Token           | Value     | Role                                                       |
| --------------- | --------- | ---------------------------------------------------------- |
| `--paper`       | `#0a1220` | Page background                                            |
| `--surface`     | `#0f1a2c` | Chart frames, controls, cards                              |
| `--raised`      | `#15233a` | Chapter link in view                                       |
| `--rule`        | `#1f2e47` | Gridlines, borders                                         |
| `--rule-strong` | `#2c3e5c` | Baselines, axis ticks                                      |
| `--ink`         | `#eaf0f7` | Text, éCO2mix in the content chart                         |
| `--ink-2`       | `#b0bdd0` | Secondary text                                             |
| `--muted`       | `#8292ab` | Axes, captions, labels                                     |
| `--dim`         | `#4e5f7c` | Earlier years of the ridgeline, baseline models            |
| `--published`   | `#7cb4ff` | Published output, feed-in series, missing or shifted hours |
| `--model`       | `#ffa04f` | Modelled potential, held-back output, focus ring           |
| `--third`       | `#3fd2ae` | Series with self-consumption, findings against other series|
| `--flag`        | `#ff7d8c` | Impossible values, tickets                                 |

The held-back output and the bootstrap intervals are hatched with fine amber lines over a 10 %
amber fill (`#hatch` at the top of the page): a transparent amber turns brown on the night blue,
thin lines stay bright.

## Type

- Display: Bricolage Grotesque, weights 400 to 600; the headline figure at 88 % width with tight
  tracking and tabular figures.
- Interface and text: Geist, 12 to 17 px, line height 1.5 to 1.7, tabular figures in charts and
  tables.
- Both faces are served from `site/fonts` (latin subset) under the SIL Open Font License.

## Layout

- Content width `82rem`, gutter `clamp(1.25rem, …, 3rem)`.
- Hero: kicker, figure, lead and two facts on the left (5 of 12 columns), the ridgeline on the
  right.
- Chapters, numbered 01 to 05 after the questions of the study: number, title and summary on the
  left (4 of 12), sticky on wide screens; readouts, controls and charts on the right (8 of 12).
  One column below `60rem`; below `34rem` the year choice spans the width and the content matrix
  turns into one card per series.

## Motion

- The hero: the figure counts up from a tenth below, the years of the ridgeline draw one after
  another (90 ms apart, 1.1 s each), then the held-back output fades in.
- Each chart wipes in from the left the first time it scrolls into view (1.1 s); the bars of the
  model error grow in, one step after another.
- Curves: `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)` for entries, `--ease-in-out:
  cubic-bezier(0.77, 0, 0.175, 1)` for the wipe. Hover and color changes take 160 ms. With
  `prefers-reduced-motion: reduce`, everything is drawn at once.
