"""The web page in site/ is served as static files: every file it asks for must be there."""

import re
from pathlib import Path

import pytest

SITE = Path(__file__).parents[1] / "site"


def local_references(text: str, pattern: str) -> list[str]:
    return [ref for ref in re.findall(pattern, text) if not re.match(r"[a-z]+:|#|/", ref)]


def test_the_page_links_only_files_that_exist() -> None:
    page = (SITE / "index.html").read_text()
    references = local_references(page, r'(?:href|src)="([^"]+)"')

    assert references
    assert [ref for ref in references if not (SITE / ref).is_file()] == []


def test_the_stylesheet_loads_only_files_that_exist() -> None:
    stylesheet = (SITE / "style.css").read_text()
    references = local_references(stylesheet, r'url\("([^"]+)"\)')

    assert references
    assert [ref for ref in references if not (SITE / ref).is_file()] == []


@pytest.mark.parametrize("module", sorted((SITE / "js").glob("*.js")), ids=lambda path: path.name)
def test_each_module_imports_modules_that_exist(module: Path) -> None:
    imports = re.findall(r'from "(\./[^"]+)"', module.read_text())

    assert [name for name in imports if not (module.parent / name).is_file()] == []


def test_the_data_documents_read_by_the_page_are_exported() -> None:
    scripts = "".join(path.read_text() for path in (SITE / "js").glob("*.js"))
    documents = set(re.findall(r'load\("([^"]+)"\)', scripts))
    documents |= {
        f"daily-{country}-{technology}.json"
        for country in ("FR", "DE")
        for technology in ("solar", "wind_onshore")
    }

    assert [name for name in sorted(documents) if not (SITE / "data" / name).is_file()] == []
