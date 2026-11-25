import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from eu_solar_wind_audit.sources import pinned

URL = "https://example.org/data.zip"
BODY = b"pinned content"
SHA256 = hashlib.sha256(BODY).hexdigest()

Handler = Callable[[httpx.Request], httpx.Response]


def serve(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def respond(status: int, content: bytes, requests: list[str]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, content=content)

    return handler


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def test_download_is_checked_then_cached(tmp_path: Path) -> None:
    requests: list[str] = []
    dest = tmp_path / "cache" / "data.zip"

    with serve(respond(200, BODY, requests)) as client:
        first = pinned.download(URL, SHA256, dest, client)
        second = pinned.download(URL, SHA256, dest, client)

    assert requests == [URL]
    assert first == second == dest
    assert dest.read_bytes() == BODY
    assert list(dest.parent.iterdir()) == [dest]


def test_dropped_connection_is_retried(tmp_path: Path) -> None:
    attempts: list[str] = []

    def drop_first(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        if len(attempts) == 1:
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return httpx.Response(200, content=BODY)

    with serve(drop_first) as client:
        path = pinned.download(URL, SHA256, tmp_path / "data.zip", client)

    assert len(attempts) == 2
    assert path.read_bytes() == BODY


def test_wrong_checksum_raises_and_caches_nothing(tmp_path: Path) -> None:
    with (
        serve(respond(200, b"other content", [])) as client,
        pytest.raises(ValueError, match="SHA-256"),
    ):
        pinned.download(URL, SHA256, tmp_path / "data.zip", client)

    assert list(tmp_path.iterdir()) == []


def test_http_error_is_not_retried_and_caches_nothing(tmp_path: Path) -> None:
    requests: list[str] = []

    with (
        serve(respond(404, b"Not Found", requests)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        pinned.download(URL, SHA256, tmp_path / "data.zip", client)

    assert len(requests) == 1
    assert list(tmp_path.iterdir()) == []
