"""Data processing, canonicalisation, and split management."""
from .canonical import (
    canonicalise_url,
    normalise_nfc,
    is_https,
    has_empty_path,
    is_s_strict,
    is_s_ctrl,
)
from .dataset import load_dataset, get_stratum
from .split import create_group_disjoint_split, materialize_split

__all__ = [
    "canonicalise_url",
    "normalise_nfc",
    "is_https",
    "has_empty_path",
    "is_s_strict",
    "is_s_ctrl",
    "load_dataset",
    "get_stratum",
    "create_group_disjoint_split",
    "materialize_split",
]
