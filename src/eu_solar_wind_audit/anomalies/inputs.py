"""Inputs of the anomaly detectors (evaluation plan, axis 2) for a detection run of one country.

The physics model covers onshore wind only, so the detectors that compare published wind with it
skip the hours that contain offshore output (evaluation plan, Deviations). `offshore_starts` dates
these hours for each French area.
"""

import pandas as pd

from eu_solar_wind_audit.sources import odre, sdes

# Saint-Nazaire, the first French offshore farm (odre.FIRST_OFFSHORE_POWER), is in Pays de la Loire.
FIRST_FARM_REGION = "52"
# Elsewhere, offshore output counts from the first quarter in which a region has more than 10 MW
# offshore on the SDES panel, which leaves out the single 10 MW installation that the panel gives
# Pays de la Loire from the end of 2018 on.
OFFSHORE_FROM_MW = 10.0


def offshore_starts(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """First hour, in UTC, from which the published wind of each French area holds offshore
    output, for the areas that have some, keyed by area: "FR" for mainland France, "FR-<code>"
    for a region of odre.REGIONS.

    Mainland France and Pays de la Loire start at odre.FIRST_OFFSHORE_POWER. Another region starts
    on the first day, in Europe/Paris, of the first quarter whose offshore capacity exceeds
    OFFSHORE_FROM_MW on `panel`, a quarterly panel of sdes.read_capacity.
    """
    mainland = sdes.metropolitan_without_corsica(panel)
    offshore = mainland[mainland["technology"] == "wind_offshore"]
    by_quarter = offshore.groupby(["region", "quarter_end"])["capacity_mw"].sum().reset_index()
    above = by_quarter[by_quarter["capacity_mw"] > OFFSHORE_FROM_MW]
    first_quarter = above.groupby("region")["quarter_end"].min()
    starts = {"FR": odre.FIRST_OFFSHORE_POWER, f"FR-{FIRST_FARM_REGION}": odre.FIRST_OFFSHORE_POWER}
    for code, name in odre.REGIONS.items():
        if code != FIRST_FARM_REGION and name in first_quarter.index:
            first_day = pd.Timestamp(first_quarter[name]).to_period("Q").start_time
            starts[f"FR-{code}"] = first_day.tz_localize(odre.TIMEZONE).tz_convert("UTC")
    return starts
