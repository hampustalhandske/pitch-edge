from __future__ import annotations

import pytest
import responses

from pitch_edge.data.http import CachedHttpClient, RobotsDisallowedError

pytestmark = pytest.mark.unit


@responses.activate
def test_cache_hit_avoids_second_request(tmp_path):
    responses.add(responses.GET, "https://example.test/a.csv", body=b"x,y\n1,2\n", status=200)
    c = CachedHttpClient(tmp_path, min_interval_s=0)
    assert c.get_text("https://example.test/a.csv") == "x,y\n1,2\n"
    assert c.get_text("https://example.test/a.csv") == "x,y\n1,2\n"
    assert len(responses.calls) == 1
    assert c.stats == {"hits": 1, "misses": 1, "errors": 0}


@responses.activate
def test_force_refresh_and_cache_false(tmp_path):
    responses.add(responses.GET, "https://example.test/a", body=b"1", status=200)
    responses.add(responses.GET, "https://example.test/a", body=b"2", status=200)
    c = CachedHttpClient(tmp_path, min_interval_s=0)
    assert c.get_bytes("https://example.test/a") == b"1"
    assert c.get_bytes("https://example.test/a", force=True) == b"2"
    assert len(responses.calls) == 2


@responses.activate
def test_retries_on_5xx_then_succeeds(tmp_path):
    responses.add(responses.GET, "https://example.test/flaky", status=503)
    responses.add(responses.GET, "https://example.test/flaky", body=b"ok", status=200)
    c = CachedHttpClient(tmp_path, min_interval_s=0, max_retries=3)
    assert c.get_bytes("https://example.test/flaky") == b"ok"


@responses.activate
def test_404_raises_immediately(tmp_path):
    responses.add(responses.GET, "https://example.test/missing", status=404)
    c = CachedHttpClient(tmp_path, min_interval_s=0)
    import requests

    with pytest.raises(requests.HTTPError):
        c.get_bytes("https://example.test/missing")
    assert len(responses.calls) == 1


@responses.activate
def test_robots_disallow_is_respected(tmp_path):
    responses.add(responses.GET, "https://blocked.test/robots.txt", body="User-agent: *\nDisallow: /\n", status=200)
    responses.add(responses.GET, "https://blocked.test/page", body=b"nope", status=200)
    c = CachedHttpClient(tmp_path, min_interval_s=0)
    with pytest.raises(RobotsDisallowedError):
        c.get_bytes("https://blocked.test/page", check_robots=True)


@responses.activate
def test_circuit_breaker_opens_after_repeated_host_failures(tmp_path):
    from pitch_edge.data.http import HostCircuitOpenError

    for i in range(3):
        responses.add(responses.GET, f"https://down.test/{i}", status=503)
    c = CachedHttpClient(tmp_path, min_interval_s=0, max_retries=1)
    c.max_consecutive_failures = 2
    for i in range(2):
        with pytest.raises(RuntimeError):
            c.get_bytes(f"https://down.test/{i}")
    with pytest.raises(HostCircuitOpenError):
        c.get_bytes("https://down.test/2")
    assert len(responses.calls) == 2  # third URL never requested


@responses.activate
def test_get_json_with_params(tmp_path):
    responses.add(responses.GET, "https://api.test/x?limit=2", json={"ok": True}, status=200, match_querystring=True)
    c = CachedHttpClient(tmp_path, min_interval_s=0)
    assert c.get_json("https://api.test/x", params={"limit": 2}) == {"ok": True}
