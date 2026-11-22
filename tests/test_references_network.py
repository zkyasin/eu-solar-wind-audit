import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

RTE = Path(__file__).parents[1] / "references" / "rte"


@pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")
def test_published_rte_files_are_unchanged_since_the_manifest() -> None:
    manifest = json.loads((RTE / "manifest.json").read_text(encoding="utf-8"))
    changed = []
    with httpx.Client(timeout=30) as client:
        for name, entry in manifest["files"].items():
            response = client.get(entry["url"])
            response.raise_for_status()
            digest = hashlib.sha256(response.content).hexdigest()
            if digest != entry["sha256"]:
                changed.append(
                    f"{name}: now {len(response.content)} bytes with SHA-256 {digest}, "
                    f"last modified {response.headers.get('last-modified')}"
                )

    assert not changed, "RTE republished files since the manifest:\n" + "\n".join(changed)
