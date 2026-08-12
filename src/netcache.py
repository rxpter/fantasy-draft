"""Tiny stdlib HTTP client with a TTL disk cache.

No third-party deps on purpose -- this has to run on a bare Python install
minutes before a draft without a pip step.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

_UA = "fantasy-draft-assistant/1.0 (personal use)"


def _make_ssl_context() -> ssl.SSLContext:
    """Full cert verification, minus OpenSSL's strict RFC 5280 pedantry.

    Python 3.13+ turns on VERIFY_X509_STRICT, which rejects otherwise valid
    chains when a CA in the Windows trust store omits the `critical` marker on
    its Basic Constraints extension. Certificate validation and hostname
    checking both stay enabled -- only the strictness flag is cleared.
    """
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


_SSL_CONTEXT = _make_ssl_context()


class FetchError(RuntimeError):
    pass


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{key}.json"


def get_json(url: str, ttl_hours: float = 0.0, timeout: int = 90):
    """GET a JSON URL, optionally served from a TTL disk cache.

    ttl_hours=0 disables the cache (used for live draft picks).
    """
    path = _cache_path(url)

    if ttl_hours > 0 and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < ttl_hours * 3600:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass  # corrupt cache entry -- fall through and refetch

    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept-Encoding": "gzip", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Fall back to a stale cache entry rather than dying mid-draft.
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
        raise FetchError(f"network error for {url}: {exc}") from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise FetchError(f"bad JSON from {url}: {exc}") from exc

    if ttl_hours > 0:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)
            tmp.replace(path)
        except OSError:
            pass  # cache is an optimisation, never fatal

    return data
