import hashlib
import os
from pathlib import Path

import pytest

from eu_solar_wind_audit.sources import gisco

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def lau(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return gisco.download_lau(raw_dir=tmp_path_factory.mktemp("gisco"))


def test_download_matches_the_pinned_checksum(lau: Path) -> None:
    with lau.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == gisco.SHA256


def test_denmark_has_its_98_municipalities_and_christianso(lau: Path) -> None:
    denmark = gisco.load_municipalities("DK", lau)

    # Christiansø (411) is administered by the state, outside the 98 municipalities.
    assert len(denmark) == 99
    assert denmark["region"].str.fullmatch(r"\d{3}").all()
    assert {"101", "400", "411", "860"} <= set(denmark["region"])


def test_the_netherlands_has_its_342_municipalities_of_2024(lau: Path) -> None:
    netherlands = gisco.load_municipalities("NL", lau)

    assert len(netherlands) == 342
    assert netherlands["region"].str.fullmatch(r"GM\d{4}").all()
