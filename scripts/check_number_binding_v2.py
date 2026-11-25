#!/usr/bin/env python3
"""
Checks numeric literals inside paper/main.tex against committed v2 JSON artifacts.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import POLICIES_DIR, REPO_ROOT, RESULTS_V2_DIR, read_json

DEFAULT_TEX = (
    REPO_ROOT / "paper" / "main.tex"
    if (REPO_ROOT / "paper" / "main.tex").exists()
    else REPO_ROOT / "paper" / "main_final_audit.tex"
)

DERIVED_WHITELIST: dict[str, str] = {
    "74.70": "100-25.30",
    "87.55": "100-12.45",
    "20.8": "100-79.23",
    "479": "1439-960",
    "59.8": "280/468",
    "74.7": "1-.253",
    "87.6": "1-.1245",
    "4.83": "E9 pi",
    "6.58": "eps",
    "98": "1-20/971",
    "99": "1-10/1439",
    "75": "100-25",
    "2": "half",
    "10": "",
    "4": "",
    "5": "",
    "3": "",
    "1": "",
    "0": "",
}

# Explicit allowlist of non-metric or auxiliary-artifact tokens matched by the
# regex on the committed camera-ready paper/main.tex:
KNOWN_UNMATCHED: frozenset[str] = frozenset(
    {
        # Corpus names, collection years, citation keys, URLs, and prompt hash prefix/suffix
        "017",   # EBBU2017
        "017}",  # \textbf{EBBU2017}
        "021",   # PWD2021
        "021}",  # \textbf{PWD2021}
        "023}",  # \cite{jitkrittum2023}
        "026",   # rivf2026-llm-url-routing URL
        "2025",  # URL-Phish 2025
        "2026",  # August / November 2026 harvest dates
        "797",   # frozen prompt SHA-256 snippet e67d6cb3...8797d8d3
        # LaTeX brace-attached formatting/structural tokens (\textbf, \mathbf, \cmidrule, exponents)
        "0}",    # brace-closed 0
        "10}",   # \begin{thebibliography}{10}
        "1}",    # brace-closed 1
        "21}",   # 10^{-21}
        "2}{",   # \multicolumn{2}{c}
        "3}",    # \cmidrule(lr){2-3}
        "4}",    # 10^{-4}
        "188}",  # \mathbf{+188} (188 is bound in JSON)
        "235}",  # \textbf{235} (235 is bound in JSON)
        "280}",  # \mathbf{+280} (280 is bound in JSON)
        "414}",  # \textbf{414} (414 is bound in JSON)
        "485}",  # \mathbf{+485} (485 is bound in JSON)
        "759}",  # \textbf{759} (759 is bound in JSON)
        "899}",  # \textbf{899} (899 is bound in JSON)
        # Values from other frozen artifacts (policy_freeze_v2, manifests, lineage, specialist baselines)
        "23273",  # August E3 raw response rows before deduplication
        "3605",   # November E3 backfill raw response rows
        "24.94",  # DEV_POLICY realized escalation % (policy_freeze_v2.json)
        "2993",   # unique canonical URLs in 3,000-row E9 manifest
        "4835",   # PWD2021-EBBU2017 shared URLs (dataset_lineage_matrix.json)
        "8.7",    # min F1 point advantage of Char TF-IDF over lexical trees (Table III)
    }
)

JSON_SOURCES = (
    "v2_main_metrics.json",
    "oos_sample_size.json",
    "v2_frontier_and_error.json",
    "v2_paper_extras.json",
    "v2_policies.json",
)


def _collect_number_strings(data: Any, vals: set[str]) -> None:
    def _add(x: Any) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            vals.add(str(abs(x)))
            return
        if isinstance(x, float):
            for scaled in (x, x * 100.0, x * 1e6):
                av = abs(scaled)
                for nd in range(0, 5):
                    s = f"{av:.{nd}f}"
                    vals.add(s)
                    vals.add(s.lstrip("0") if av < 1 else s)
            vals.add(f"{abs(x):.2e}".split("e")[0])

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        else:
            _add(obj)

    _walk(data)


def check_manuscript(
    tex_path: Path = DEFAULT_TEX,
    *,
    v2_dir: Path | None = None,
    policies_dir: Path = POLICIES_DIR,
) -> dict[str, Any]:
    """Scans Abstract and Sections IV–Conclusion of tex_path against v2 JSON numbers."""
    v2_base = v2_dir or Path(os.environ.get("V2DATA", str(RESULTS_V2_DIR)))
    vals: set[str] = set()
    _collect_number_strings(read_json(policies_dir / "tau_BE_v2.json"), vals)
    for fname in JSON_SOURCES:
        _collect_number_strings(read_json(v2_base / fname), vals)

    tex = tex_path.read_text(encoding="utf-8")
    abs_start = tex.index(r"\begin{abstract}")
    abs_end = tex.index(r"\end{abstract}")
    body_start = tex.index(r"\subsection{Partitions and Leakage Control}")
    body_end = tex.index(r"\section*{Acknowledgment}")
    body = tex[abs_start:abs_end] + tex[body_start:body_end]
    body = re.sub(r"%.*", "", body)
    body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", body, flags=re.S)

    nums = re.findall(r"(?<![A-Za-z\\])[-+]?\.?\d[\d,{}]*(?:\.\d+)?", body)
    unmatched: dict[str, int] = {}
    for raw in nums:
        k = raw.replace("{,}", "").replace(",", "").lstrip("+-")
        k2 = "0" + k if k.startswith(".") else k
        if k in vals or k2 in vals or k in DERIVED_WHITELIST or k.rstrip(".") in vals:
            continue
        try:
            if float(k2) in (100.0,):
                continue
        except ValueError:
            pass
        unmatched[k] = unmatched.get(k, 0) + 1

    sorted_unmatched = dict(sorted(unmatched.items()))
    unexpected_unmatched = {
        k: v for k, v in sorted_unmatched.items() if k not in KNOWN_UNMATCHED
    }
    return {
        "total_numbers": len(nums),
        "unmatched_distinct": len(unmatched),
        "unmatched": sorted_unmatched,
        "unexpected_unmatched": unexpected_unmatched,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tex_path", nargs="?", type=Path, default=DEFAULT_TEX, help="Path to LaTeX manuscript")
    parser.add_argument("--v2-dir", type=Path, default=None, help="Directory holding v2 JSONs")
    args = parser.parse_args(argv)

    res = check_manuscript(args.tex_path, v2_dir=args.v2_dir)
    unexpected = res["unexpected_unmatched"]
    print(
        res["total_numbers"],
        "numbers;",
        res["unmatched_distinct"],
        f"unmatched ({len(unexpected)} unexpected):",
    )
    print(sorted(res["unmatched"].items()))
    if unexpected:
        print("unexpected_unmatched:", sorted(unexpected.items()))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
