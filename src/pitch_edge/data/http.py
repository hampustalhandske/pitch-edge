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
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests

from pitch_edge.config import USER_AGENT

logger = logging.getLogger(__name__)


class RobotsDisallowedError(RuntimeError):
    """Raised when a URL is disallowed by the host's robots.txt (we never bypass it)."""


class HostCircuitOpenError(RuntimeError):
    """Raised once a host has failed repeatedly in this session; we stop asking it."""


class LiveHttpError(RuntimeError):
    """A live (uncached) fetch failed. The message names the URL with secret query values redacted;
    `status_code` is the HTTP status, or None for a transport failure."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def redact_url(url: str, names: tuple[str, ...] | list[str]) -> str:
    """`url` with the values of the query parameters in `names` replaced by `REDACTED`."""
    if not names:
        return url
    parts = urlparse(url)
    query = [(k, "REDACTED" if k in names else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunparse(parts._replace(query=urlencode(query)))


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

    def get_live(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        redact: tuple[str, ...] | list[str] = (),
    ) -> requests.Response:
        """Fetch a live endpoint (current prices, quota headers): never read from or written to the disk cache.

        Same honest User-Agent, per-host throttle and circuit breaker as `get_bytes`. robots.txt is not
        consulted: this is for published JSON APIs (the-odds-api, Polymarket Gamma/CLOB, Kalshi), which are
        documented machine endpoints rather than web pages, the same rule `get_bytes` applies by default. 5xx/429 and transport
        errors are retried with backoff; any other 4xx raises at once. Errors raise `LiveHttpError` whose
        message carries the URL with the query parameters named in `redact` (e.g. an API key) replaced,
        and which is raised `from None` so the underlying exception (whose text contains the raw URL)
        is not chained into tracebacks. Returns the `requests.Response` so callers can read headers.
        """
        full_url = requests.Request("GET", url, params=params).prepare().url or url
        safe_url = redact_url(full_url, redact)
        host = urlparse(full_url).netloc
        if self._host_failures[host] >= self.max_consecutive_failures:
            raise LiveHttpError(f"{host} failed {self._host_failures[host]} times in a row; circuit open")
        last_status: int | None = None
        last_kind = ""
        for attempt in range(self.max_retries):
            self._throttle(host)
            try:
                resp = self.session.get(full_url, timeout=self.timeout_s, headers=headers)
            except requests.RequestException as exc:
                last_status, last_kind = None, type(exc).__name__
                time.sleep(min(2**attempt, 8))
                continue
            if resp.status_code >= 500 or resp.status_code == 429:
                last_status, last_kind = resp.status_code, "retryable status"
                time.sleep(min(2**attempt, 8))
                continue
            if resp.status_code >= 400:
                self.stats["errors"] += 1
                raise LiveHttpError(f"HTTP {resp.status_code} from {safe_url}", resp.status_code) from None
            self.stats["misses"] += 1
            self._host_failures[host] = 0
            return resp
        self.stats["errors"] += 1
        self._host_failures[host] += 1
        raise LiveHttpError(
            f"Failed to fetch {safe_url} after {self.max_retries} attempts ({last_kind} {last_status or ''})".strip(),
            last_status,
        ) from None
