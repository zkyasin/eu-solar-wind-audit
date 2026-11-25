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

Tokens are custom properties at the top of `site/style.css`; the charts take them as `var()`
references, so a change of theme recolors every mark without a redraw. The page opens dark; the
light theme keeps the same roles on a white paper, with hues darkened until lines and small
figures hold their contrast.

| Token           | Dark      | Light     | Role                                                        |
| --------------- | --------- | --------- | ----------------------------------------------------------- |
| `--paper`       | `#0a1220` | `#f6f7f9` | Page background                                             |
| `--surface`     | `#0f1a2c` | `#ffffff` | Chart frames, controls, cards                               |
| `--raised`      | `#15233a` | `#eaeef4` | Chapter link in view                                        |
| `--rule`        | `#1f2e47` | `#dfe4ec` | Gridlines, borders                                          |
| `--rule-strong` | `#2c3e5c` | `#c5cedb` | Baselines, axis ticks                                       |
| `--ink`         | `#eaf0f7` | `#0d1726` | Text, éCO2mix in the content chart                          |
| `--ink-2`       | `#b0bdd0` | `#364459` | Secondary text                                              |
| `--muted`       | `#8292ab` | `#5e6b80` | Axes, captions, labels                                      |
| `--dim`         | `#4e5f7c` | `#a3aebf` | Earlier years of the ridgeline, baseline models             |
| `--published`   | `#7cb4ff` | `#2160c4` | Published output, feed-in series, missing or shifted hours  |
| `--model`       | `#ffa04f` | `#c75a06` | Modelled potential, held-back output, focus ring            |
| `--third`       | `#3fd2ae` | `#0b8f71` | Series with self-consumption, findings against other series |
| `--flag`        | `#ff7d8c` | `#c9344d` | Impossible values, tickets                                  |

The held-back output and the bootstrap intervals are hatched with fine amber lines over a light
amber fill (`#hatch` at the top of the page, 10 % by night, 14 % by day): a transparent amber
turns brown on the night blue, thin lines stay bright.

## Theme

The sun or moon button in the masthead switches themes; the icon shows the theme a click brings
and its label says so ("Passer au thème clair", "Passer au thème sombre"). The choice is stored
under `theme` in `localStorage`. An inline script in `<head>` applies it before the stylesheet
paints, so a reader who chose the light theme never sees the dark one first; `meta theme-color`
follows.

## Language

The page is written in French (`lang="fr"`). Numbers go through `Intl.NumberFormat("fr-FR")`
in `site/js/format.js`: a decimal comma, a narrow no-break space between thousands and before a
unit (`1,58 TWh`, `95,5 %`), months written out (`11 juin 2025`). The Copernicus notice keeps
its prescribed English wording and is marked `lang="en"`.

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
