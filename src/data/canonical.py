"""
URL Normalisation and Canonicalisation Protocol.
Deterministic URL normalization and SHA-256 content addressing.
"""

from __future__ import annotations

import unicodedata
from urllib.parse import urlparse, urlunparse
import hashlib


def normalise_nfc(text: str) -> str:
    """Apply Unicode NFC normalisation."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return unicodedata.normalize("NFC", text)


def canonicalise_url(url: str) -> str:
    """
    Canonicalise URL string:
    - Unicode NFC normalisation.
    - Lowercase scheme and netloc.
    - Strip standard 'www.' prefix.
    - Strip trailing slashes on empty paths.
    - Retain path, query, and fragment where present.
    """
    u = normalise_nfc(url).strip()
    try:
        p = urlparse(u)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        
        path = p.path.rstrip("/")
        if not path and not p.query and not p.fragment:
            path = ""
            
        canonical = urlunparse((
            scheme,
            netloc,
            path,
            p.params,
            p.query,
            p.fragment
        ))
        return canonical
    except Exception:
        return u.lower().strip()


def compute_url_hash(url_string: str) -> str:
    """Compute SHA-256 hash of the canonical URL string."""
    return hashlib.sha256(canonicalise_url(url_string).encode("utf-8")).hexdigest()


def is_https(url: str) -> bool:
    """True if scheme is HTTPS."""
    try:
        p = urlparse(url)
        return p.scheme.lower() == "https" or url.lower().startswith("https://")
    except Exception:
        return False


def has_empty_path(url: str) -> bool:
    """True if path is empty or single slash."""
    try:
        p = urlparse(url)
        return len(p.path.strip("/")) == 0
    except Exception:
        return False


def is_s_ctrl(url: str) -> bool:
    """S-CTRL Predicate: HTTPS AND empty path."""
    return is_https(url) and has_empty_path(url)


def is_s_strict(url: str) -> bool:
    """
    S-STRICT Predicate (D-15):
    - HTTPS scheme
    - Empty path
    - No query string
    - No fragment
    - No params
    - No username / password in netloc
    - No non-default port
    """
    try:
        p = urlparse(url)
        if p.scheme.lower() != "https":
            return False
        if len(p.path.strip("/")) > 0:
            return False
        if p.params or p.query or p.fragment:
            return False
        if p.username or p.password:
            return False
        if p.port is not None and p.port != 443:
            return False
        return True
    except Exception:
        return False
