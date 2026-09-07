"""Polite, cached HTTP client shared by every connector.

Every external source goes through this so the ToS guardrails are enforced
in one place: honest User-Agent, per-host throttling, exponential-backoff
retries, on-disk caching (never re-download an immutable historical file),
and a robots.txt check for anything that is a web page rather than a
published API/dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from pitch_edge.config import USER_AGENT

logger = logging.getLogger(__name__)


class RobotsDisallowedError(RuntimeError):
    """Raised when a URL is disallowed by the host's robots.txt (we never bypass it)."""


class HostCircuitOpenError(RuntimeError):
    """Raised once a host has failed repeatedly in this session; we stop asking it."""


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: str | Path,
        min_interval_s: float = 1.0,
        timeout_s: int = 30,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_call: dict[str, float] = defaultdict(float)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._host_failures: dict[str, int] = defaultdict(int)
        self.max_consecutive_failures = 6  # per host; trips a circuit breaker so an outage isn't hammered
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

    # ------------------------------------------------------------------ cache
    def _cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}{suffix}"

    def _throttle(self, host: str) -> None:
        elapsed = time.monotonic() - self._last_call[host]
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call[host] = time.monotonic()

    # ----------------------------------------------------------------- robots
    def allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self._robots:
            rp = RobotFileParser()
            try:
                resp = self.session.get(f"{host}/robots.txt", timeout=self.timeout_s)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    self._robots[host] = rp
                else:
                    self._robots[host] = None
            except requests.RequestException:
                self._robots[host] = None
        rp_cached = self._robots[host]
        if rp_cached is None:
            return True
        return rp_cached.can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------------ fetch
    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        force: bool = False,
        check_robots: bool = False,
        cache: bool = True,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        full_url = requests.Request("GET", url, params=params).prepare().url or url
        path = self._cache_path(full_url, ".bin")
        if cache and path.exists() and not force:
            self.stats["hits"] += 1
            return path.read_bytes()

        if check_robots and not self.allowed_by_robots(full_url):
            raise RobotsDisallowedError(f"robots.txt disallows fetching {full_url}")

        host = urlparse(full_url).netloc
        if self._host_failures[host] >= self.max_consecutive_failures:
            self.stats["errors"] += 1
            raise HostCircuitOpenError(f"{host} failed {self._host_failures[host]} times in a row; circuit open")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle(host)
            try:
                resp = self.session.get(full_url, timeout=self.timeout_s, headers=headers)
                if resp.status_code == 404:
                    resp.raise_for_status()
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"{resp.status_code} from {host}", response=resp)
                resp.raise_for_status()
                self.stats["misses"] += 1
                self._host_failures[host] = 0
                if cache:
                    path.write_bytes(resp.content)
                return resp.content
            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code == 404:
                    self.stats["errors"] += 1
                    raise
                time.sleep(min(2**attempt, 8))
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 8))
        self.stats["errors"] += 1
        self._host_failures[host] += 1
        raise RuntimeError(f"Failed to fetch {full_url} after {self.max_retries} attempts") from last_exc

    def get_text(self, url: str, encoding: str = "utf-8", **kwargs: Any) -> str:
        return self.get_bytes(url, **kwargs).decode(encoding, errors="replace")

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return json.loads(self.get_bytes(url, **kwargs).decode("utf-8"))
