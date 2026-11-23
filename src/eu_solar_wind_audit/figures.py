"""The figures of the README, drawn with matplotlib in a light and a dark version.

Each `write_*` function draws one figure from the tables that the commands write and saves it as
`<name>-light.svg` and `<name>-dark.svg`, the same bytes when drawn again from the same inputs.
"""

import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import dates
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.typing import RcKeyType

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies import findings

NAMES = {"solar": "Solar", "wind_onshore": "Onshore wind"}
WIDTH_INCHES = 8.0
# Characters of a footer line at 7 points across WIDTH_INCHES.
FOOTER_CHARACTERS = 140
# Half the gap between the estimate and the RTE volume of a year, in years.
OFFSET = 0.12


@dataclass(frozen=True)
class Theme:
    """The colours of one version of the figures."""

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    baseline: str
    shade: str
    first: str
    second: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    shade="#e1e0d9",
    first="#2a78d6",
    second="#eb6834",
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    shade="#383835",
    first="#3987e5",
    second="#d95926",
)
THEMES = (LIGHT, DARK)


def write_week(
    directory: Path,
    hourly: Mapping[str, pd.DataFrame],
    negative: pd.Series,
    footer: str,
) -> list[Path]:
    """Draw `week`: for each technology of `hourly`, its frame of `published_mwh` and
    `potential_mwh` in GW, one panel each over the days of Europe/Paris, with the hours where
    `negative` is true shaded. The frames and `negative` share their UTC hours."""

    def draw(theme: Theme) -> Figure:
        figure = Figure(figsize=(WIDTH_INCHES, 5.4), layout="constrained")
        axes = figure.subplots(len(hourly), 1, sharex=True, squeeze=False)[:, 0]
        hours = pd.DatetimeIndex(negative.index)
        flagged = findings.runs(hours[negative.to_numpy()], merge_gap=pd.Timedelta(0))
        starts = _days(pd.DatetimeIndex(flagged["start"]))
        ends = _days(pd.DatetimeIndex(flagged["end"]))
        for ax, (technology, frame) in zip(axes, hourly.items(), strict=True):
            for start, end in zip(starts, ends, strict=True):
                ax.axvspan(start, end, color=theme.shade, linewidth=0)
            middle = _days(pd.DatetimeIndex(frame.index) + grid.HOUR / 2)
            ax.plot(middle, frame["published_mwh"] / 1000, color=theme.first)
            ax.plot(middle, frame["potential_mwh"] / 1000, color=theme.second)
            ax.set_title(NAMES[technology])
            ax.set_ylabel("GW")
            ax.set_ylim(bottom=0)
        span = _days(pd.DatetimeIndex([hours[0], hours[-1] + grid.HOUR]))
        axes[-1].set_xlim(span[0], span[1])
        axes[-1].xaxis.set_major_locator(dates.DayLocator(tz="Europe/Paris"))
        axes[-1].xaxis.set_major_formatter(dates.DateFormatter("%a %d %b", tz="Europe/Paris"))
        handles = [
            Line2D([], [], color=theme.first, label="Published"),
            Line2D([], [], color=theme.second, label="Calibrated potential"),
            Patch(color=theme.shade, label="Negative day-ahead price"),
        ]
        _frame(figure, footer, handles, theme)
        return figure

    return _write(directory, "week", draw)


def write_modulation(directory: Path, years: Mapping[str, pd.DataFrame], footer: str) -> list[Path]:
    """Draw `modulation`: for each technology of `years`, its frame indexed by year with
    `estimate_gwh`, `low_gwh`, `high_gwh` and `rte_gwh`, the estimate and its interval beside the
    RTE volume of each year, one panel each on a log scale that they share."""

    def draw(theme: Theme) -> Figure:
        figure = Figure(figsize=(WIDTH_INCHES, 3.6), layout="constrained")
        axes = figure.subplots(1, len(years), sharey=True, squeeze=False)[0]
        for ax, (technology, frame) in zip(axes, years.items(), strict=True):
            x = np.arange(len(frame))
            estimate = frame["estimate_gwh"].to_numpy()
            errors = [
                estimate - frame["low_gwh"].to_numpy(),
                frame["high_gwh"].to_numpy() - estimate,
            ]
            ax.errorbar(x - OFFSET, estimate, yerr=errors, fmt="o", color=theme.first, markersize=5)
            ax.plot(x + OFFSET, frame["rte_gwh"], "D", color=theme.second, markersize=5)
            ax.set_xticks(x, [str(year) for year in frame.index])
            ax.set_xlim(-0.5, len(frame) - 0.5)
            ax.set_title(NAMES[technology])
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        axes[0].set_ylabel("GWh")
        handles = [
            Line2D([], [], color=theme.first, marker="o", label="Estimate, 90 % interval"),
            Line2D([], [], color=theme.second, marker="D", linestyle="none", label="RTE"),
        ]
        _frame(figure, footer, handles, theme)
        return figure

    return _write(directory, "modulation", draw)


def write_sum_of_parts(
    directory: Path,
    gap_gwh: pd.Series,
    flagged: pd.DataFrame,
    time_zone: str,
    title: str,
    footer: str,
) -> list[Path]:
    """Draw `sum-of-parts` under `title`: `gap_gwh`, a daily gap indexed by the midnights of
    `time_zone`, with the runs of `flagged`, UTC `start` and exclusive `end`, shaded."""

    def draw(theme: Theme) -> Figure:
        figure = Figure(figsize=(WIDTH_INCHES, 3.4), layout="constrained")
        ax = figure.subplots()
        starts = _days(pd.DatetimeIndex(flagged["start"]), time_zone)
        ends = _days(pd.DatetimeIndex(flagged["end"]), time_zone)
        for start, end in zip(starts, ends, strict=True):
            ax.axvspan(start, end, color=theme.shade, linewidth=0)
        ax.axhline(0, color=theme.baseline, linewidth=0.8)
        ax.plot(_days(pd.DatetimeIndex(gap_gwh.index), time_zone), gap_gwh, color=theme.first)
        ax.xaxis.set_major_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b"))
        ax.set_title(title)
        ax.set_ylabel("GWh per day")
        handles = [
            Line2D([], [], color=theme.first, label="Total minus the sum of its parts"),
            Patch(color=theme.shade, label="Hours found by the detector"),
        ]
        _frame(figure, footer, handles, theme)
        return figure

    return _write(directory, "sum-of-parts", draw)


def daily_gap_gwh(total: pd.Series, parts: pd.DataFrame, time_zone: str) -> pd.Series:
    """Over the days of `time_zone`, indexed by their midnights, the sum of the hourly `total`
    minus the sum of the hourly `parts`, in GWh, from energies in MWh on the same UTC hours; NaN on
    a day with an hour missing from the total or from a part."""
    gap = total - parts.sum(axis="columns", min_count=parts.shape[1])
    days = pd.DatetimeIndex(gap.index).tz_convert(time_zone).normalize()
    daily: pd.Series = gap.groupby(days).sum() / 1000
    return daily.mask(gap.isna().groupby(days).any())


def _frame(figure: Figure, footer: str, handles: list[Any], theme: Theme) -> None:
    """The legend above the panels, and the footer below them."""
    figure.legend(handles=handles, loc="outside upper left", ncols=len(handles))
    lines = [textwrap.fill(line, FOOTER_CHARACTERS) for line in footer.splitlines()]
    figure.supxlabel("\n".join(lines), x=0.01, ha="left", fontsize=7, color=theme.muted)


def _write(directory: Path, name: str, draw: Callable[[Theme], Figure]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in THEMES:
        path = directory / f"{name}-{theme.name}.svg"
        with mpl.rc_context(_style(theme)):
            draw(theme).savefig(path, metadata={"Date": None})
        paths.append(path)
    return paths


def _style(theme: Theme) -> dict[RcKeyType, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "axes.edgecolor": theme.baseline,
        "axes.labelcolor": theme.secondary,
        "axes.titlecolor": theme.ink,
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.6,
        "xtick.color": theme.baseline,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "xtick.labelcolor": theme.secondary,
        "ytick.labelcolor": theme.secondary,
        "text.color": theme.ink,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "svg.fonttype": "path",
        # Fixed ids, so that drawing again gives the same file.
        "svg.hashsalt": "eu-solar-wind-audit",
    }


def _days(times: pd.DatetimeIndex, time_zone: str = "UTC") -> npt.NDArray[np.float64]:
    """Matplotlib day numbers of `times` read in `time_zone`. Day numbers have no time zone: the
    date locators and formatters given another one show them in it."""
    numbers: npt.NDArray[np.float64] = dates.date2num(times.tz_convert(time_zone).tz_localize(None))
    return numbers
