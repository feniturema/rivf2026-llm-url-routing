#!/usr/bin/env python3
"""
Option C selections (amendment A-2). Deterministic; reads only August artifacts + E3 manifest.

  backfill  = E3 TEST URLs with no Gemini output in August (never requested, or only HTTP 429)
  fallback  = E3 TEST URLs where August Gemini returned an empty response (-> specialist, P-52)
  bridge    = 1,000 URLs drawn with random.Random(42).sample from sha-sorted August Gemini successes
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    REPO_ROOT,
    SELECTIONS_DIR,
    read_jsonl,
    read_manifest_csv,
    sha256_file,
)


def compute_selections(
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Deterministically derives Option C selection lists and summary metadata."""
    test_rows = read_manifest_csv(repo_root / "data" / "manifests" / "e3_test_all_26886.csv")
    all_shas = {r["canonical_url_sha256"] for r in test_rows}

    cache_models: dict[str, set[str]] = collections.defaultdict(set)
    for rec in read_jsonl(repo_root / "results" / "E3" / "e3_gemini_test_cache.jsonl"):
        cache_models[rec["canonical_url_sha256"]].add(rec["model"])

    gemini_errors: dict[str, list[str]] = collections.defaultdict(list)
    for entry in read_jsonl(repo_root / "results" / "E3" / "e3_gemini_test_ledger.jsonl"):
        if entry["model"] == "gemini-flash-lite-latest":
            gemini_errors[entry["canonical_url_sha256"]].append(entry["error_status"])

    gem_ok = sorted(
        s
        for s in all_shas
        if "gemini-flash-lite-latest" in cache_models.get(s, ())
        and "gpt-5.6-luna" not in cache_models.get(s, ())
    )
    gem_ok_set = set(gem_ok)
    not_gem = sorted(s for s in all_shas if s not in gem_ok_set)
    fallback = sorted(s for s in not_gem if "empty_response" in gemini_errors.get(s, []))
    fallback_set = set(fallback)
    backfill = sorted(s for s in not_gem if s not in fallback_set)
    bridge = sorted(random.Random(42).sample(gem_ok, 1000))

    if len(gem_ok) + len(backfill) + len(fallback) != len(all_shas):
        raise RuntimeError("Selection partition does not sum to unique E3 URL count")

    lists = {
        "e3_backfill": backfill,
        "e3_aug_gemini_empty_fallback": fallback,
        "bridge": bridge,
        "e3_aug_gemini_ok": gem_ok,
    }
    meta = {}
    for name, lst in lists.items():
        lst_set = set(lst)
        content = ("\n".join(lst) + "\n").encode("utf-8")
        meta[name] = {
            "unique_urls": len(lst),
            "rows": sum(1 for r in test_rows if r["canonical_url_sha256"] in lst_set),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return lists, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify existing selection files match deterministically")
    parser.add_argument("--output-dir", type=Path, default=SELECTIONS_DIR, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing files in --output-dir")
    args = parser.parse_args(argv)

    lists, meta = compute_selections()
    if args.verify:
        mismatches = []
        for name, info in meta.items():
            path = args.output_dir / f"{name}.txt"
            if not path.exists():
                mismatches.append(f"missing {path}")
                continue
            actual = sha256_file(path)
            if actual != info["sha256"]:
                mismatches.append(f"{name}.txt: {actual} != {info['sha256']}")
        if mismatches:
            print("FAIL make_option_c_selections verification:", mismatches, file=sys.stderr)
            return 1
        print(f"PASS make_option_c_selections matches {args.output_dir.relative_to(REPO_ROOT)}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in lists.items():
        out_path = args.output_dir / f"{name}.txt"
        if out_path.exists() and not args.force:
            print(f"refusing to overwrite {out_path} (pass --verify or --force)", file=sys.stderr)
            return 1
        out_path.write_text("\n".join(lst) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
