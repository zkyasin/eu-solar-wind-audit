from pathlib import Path

import geopandas as gpd
import httpx
import pandas as pd
import pytest
from shapely.geometry import box

from eu_solar_wind_audit.sources import rivm

HEADER = "FID,id,geom,diam,ash,kw,x,y,land,naam,wt_type,gem_naam,prov_naam,ondergrond,datum"
# Positions on the grid of the map (RD New): 5.165 E and 52.667 N, then 1 km further east.
ROWS = [
    "f.7,7,POINT (140000 520000),40,70.4,500,140000,520000,Nederland,,,Urk,Flevoland,land,"
    "2026-01-12",
    "f.3,3,POINT (141000 520000),120,70.6,3000,141000,520000,Nederland,,,Urk,Flevoland,binnenwater,"
    "2026-01-12",
    "f.5,5,POINT (60000 520000),160,110,8000,60000,520000,Nederland,,,,,zee,2026-01-12",
    "f.1,1,POINT (170000 300000),80,105,3000,170000,300000,België,,,,,land,2026-01-12",
]


def map_file(tmp_path: Path, rows: list[str] = ROWS) -> Path:
    path = tmp_path / "map.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


def digest(path: Path) -> str:
    return rivm.rows_sha256(pd.read_csv(path, dtype=str, keep_default_na=False))


def test_the_map_is_downloaded_once(tmp_path: Path) -> None:
    queries: list[httpx.QueryParams] = []

    def answer(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params)
        return httpx.Response(200, content="\n".join([HEADER, *ROWS]).encode())

    client = httpx.Client(transport=httpx.MockTransport(answer))
    first = rivm.download(client, tmp_path)
    second = rivm.download(client, tmp_path)

    assert first == second
    assert len(queries) == 1
    assert queries[0]["typeNames"] == rivm.LAYER
    assert queries[0]["outputFormat"] == "csv"


def test_the_checksum_ignores_the_feature_names_and_the_order_of_rows(tmp_path: Path) -> None:
    renamed = [row.replace("f.", "g.", 1) for row in reversed(ROWS)]

    assert digest(map_file(tmp_path)) == digest(map_file(tmp_path, renamed))


def test_other_rows_raise(tmp_path: Path) -> None:
    path = map_file(tmp_path)
    expected = digest(path)
    path.write_text(path.read_text().replace(",500,", ",600,"))

    with pytest.raises(ValueError, match="rows with SHA-256"):
        rivm.read_turbines(path, expected)


def test_dutch_turbines_on_land_and_inland_water_are_onshore(tmp_path: Path) -> None:
    path = map_file(tmp_path)
    turbines = rivm.read_turbines(path, digest(path))

    dutch = rivm.onshore(turbines)

    assert dutch["turbine"].tolist() == [7, 3]
    assert dutch["longitude"].iloc[0] == pytest.approx(5.165, abs=0.001)
    assert dutch["latitude"].iloc[0] == pytest.approx(52.667, abs=0.001)


def test_hub_height_classes_round_to_the_metre(tmp_path: Path) -> None:
    path = map_file(tmp_path)
    dutch = rivm.onshore(rivm.read_turbines(path, digest(path)))

    classes = rivm.hub_height_classes(dutch[dutch["turbine"] == 7]).set_index("hub_height_class")

    assert classes.index.tolist() == ["up to 70 m"]
    assert classes.loc["up to 70 m", "capacity_mw"] == 0.5
    assert classes.loc["up to 70 m", "hub_height_m"] == pytest.approx(70.4)
    # 500 kW over the area swept by a rotor of 40 m.
    assert classes.loc["up to 70 m", "specific_power_w_m2"] == pytest.approx(397.9, abs=0.1)

    both = rivm.hub_height_classes(dutch).set_index("hub_height_class")
    assert both.loc["71 to 95 m", "capacity_mw"] == 3.0


def test_a_turbine_outside_every_polygon_goes_to_the_nearest(tmp_path: Path) -> None:
    path = map_file(tmp_path)
    dutch = rivm.onshore(rivm.read_turbines(path, digest(path)))
    west, east = dutch["longitude"].min(), dutch["longitude"].max()
    latitude = dutch["latitude"].iloc[0]
    # The first polygon holds the western turbine; the eastern one lies between the two.
    municipalities = gpd.GeoDataFrame(
        {"region": ["GM0184", "GM0171"]},
        geometry=[
            box(west - 0.001, latitude - 0.001, west + 0.001, latitude + 0.001),
            box(east + 0.02, latitude - 0.01, east + 0.03, latitude + 0.01),
        ],
        crs="EPSG:4326",
    )

    placed = rivm.municipality_capacity(dutch, municipalities)

    assert placed.to_dict("records") == [
        {"province": "Flevoland", "region": "GM0184", "capacity_mw": 3.5}
    ]
