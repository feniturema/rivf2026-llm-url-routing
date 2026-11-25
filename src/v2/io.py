"""Shared file I/O, frozen SHA-256 registry, manifest loaders, and JSON verification for v2 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_V2_DIR = RESULTS_DIR / "v2_runs"
POLICIES_DIR = RESULTS_DIR / "policies"
SELECTIONS_DIR = RESULTS_V2_DIR / "selections"

LEGACY_TAU_04363 = 0.4362536023939326

FROZEN_ARTIFACTS: dict[str, str] = {
    "results/policies/policy_freeze_v2.json": "3f7d24fcccb99e008d4eeb421df431270f0a1978ccbb1e8b2c16a5dff9375ae5",
    "results/policies/tau_BE_v2.json": "4aec35bac0177a746bccea8c061f741edc0a37ccfa8b86f4c5e81c37725cce2d",
    "results/v2_runs/selections/e3_backfill.txt": "40bae39e121b83c43e3e4f015ca8a49bf49315c8eed3008da9b40ff2f2cda51b",
    "results/v2_runs/selections/e3_aug_gemini_ok.txt": "1eb3ef8677335b43b5c1f3c68c22dc2e52aaf9fdb46f222cbfdb4a456319610e",
    "results/v2_runs/selections/bridge.txt": "fe714f832a727855658b8a093327f2b4ecb4c0e9b0ea42cdccdeb64af778feae",
    "results/v2_runs/selections/e3_aug_gemini_empty_fallback.txt": "c48e7cc47090618f478dd0ea35badd4a409e917e1b04bb6f14f20a988b07c0d5",
    "data/manifests/e3_test_all_26886.csv": "3a34662b9d86abb8846a0cdbf9079bee7759b142df7a9f5285011a90685b7d7e",
    "data/manifests/e4_cal_all_13446.csv": "7ecb9929e2939acaea687f16f7e1251d7587125673343e152972b5d35c373195",
    "data/manifests/e9_shared_test_cohort_3000.csv": "f33b8a178e95d98f4b9071f531b327f51d106918d18e903ab8f07871b31810c4",
    "results/v2_runs/E3_BACKFILL/gemini-3.5-flash-lite/cache.jsonl": "6414f2364c3ed6940427869e3687cc414b139d92c3ff5b52f365f40f780bf357",
    "results/E3/e3_gemini_test_cache.jsonl": "6548515f751653e1ecec0ef3f638d865a0eee8d1139234cfb9ac4dfc150e19ff",
}


def sha256_file(path: Path) -> str:
    """Computes streaming SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Reads UTF-8 JSON file using an explicit context manager."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Writes JSON payload to UTF-8 file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads non-empty lines from a UTF-8 JSONL file; returns [] if absent."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    """Reads a CSV manifest into a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_selection(name_or_path: str | Path) -> set[str]:
    """Reads a whitespace-delimited SHA-256 selection list."""
    p = Path(name_or_path)
    if not p.is_absolute() and not p.exists():
        p = SELECTIONS_DIR / name_or_path
    return set(p.read_text(encoding="utf-8").split())


def load_llm_verdict_map(
    cache_path: Path,
    *,
    model_filter: str | None = None,
    keep_shas: set[str] | None = None,
    with_prob: bool = False,
) -> dict[str, Any]:
    """
    Loads successful LLM predictions from a cache JSONL keyed by canonical_url_sha256.

    When `with_prob=True`, maps hash -> (binary_verdict, parsed_probability).
    Otherwise maps hash -> binary_verdict (1=phishing, 0=legitimate).
    """
    result: dict[str, Any] = {}
    for rec in read_jsonl(cache_path):
        if rec.get("error_status") != "success":
            continue
        verdict = rec.get("parsed_verdict")
        if verdict not in ("phishing", "legitimate"):
            continue
        if model_filter is not None and rec.get("model") != model_filter:
            continue
        sha = rec["canonical_url_sha256"]
        if keep_shas is not None and sha not in keep_shas:
            continue
        label = 1 if verdict == "phishing" else 0
        result[sha] = (label, float(rec["parsed_probability"])) if with_prob else label
    return result


def load_composite_gemini_e3(
    *,
    repo_root: Path = REPO_ROOT,
    with_prob: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Loads the Option C (amendment A-2) composite Gemini E3 map without mixing or overlap:
      23,060 August Gemini unique URLs + 3,601 v2 backfill unique URLs + 8 specialist fallbacks.
    """
    v2_dir = repo_root / "results" / "v2_runs"
    sel_dir = v2_dir / "selections"

    aug = load_llm_verdict_map(
        repo_root / "results" / "E3" / "e3_gemini_test_cache.jsonl",
        model_filter="gemini-flash-lite-latest",
        with_prob=with_prob,
    )
    backfill = load_llm_verdict_map(
        v2_dir / "E3_BACKFILL" / "gemini-3.5-flash-lite" / "cache.jsonl",
        with_prob=with_prob,
    )

    ok_shas = read_selection(sel_dir / "e3_aug_gemini_ok.txt")
    backfill_shas = read_selection(sel_dir / "e3_backfill.txt")
    fallback_shas = read_selection(sel_dir / "e3_aug_gemini_empty_fallback.txt")

    if ok_shas & backfill_shas:
        raise ValueError("Selection overlap between e3_aug_gemini_ok and e3_backfill")
    if (ok_shas | backfill_shas) & fallback_shas:
        raise ValueError("Selection overlap with specialist fallback URLs")

    composite: dict[str, Any] = {}
    for sha in ok_shas:
        composite[sha] = aug[sha]
    for sha in backfill_shas:
        composite[sha] = backfill[sha]

    source_counts = {
        "aug_gemini": len(ok_shas),
        "v2_backfill": len(backfill_shas),
        "specialist_fallback": len(fallback_shas),
    }
    return composite, source_counts


def load_e4_gemini_cal(
    *,
    repo_root: Path = REPO_ROOT,
    with_prob: bool = False,
) -> dict[str, Any]:
    """Loads the E4 CAL_ROUTER Gemini cache (amendment A-2)."""
    return load_llm_verdict_map(
        repo_root / "results" / "E4" / "e4_gemini_cal_cache.jsonl",
        with_prob=with_prob,
    )


def load_e9_vendor_maps(
    gemini_e3_map: dict[str, int],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, int]]:
    """Loads verdict maps for all 5 models evaluated on the shared E9 3,000-row cohort."""
    e9_dir = repo_root / "results" / "v2_runs" / "E9"
    return {
        "gemini-3.5-flash-lite (composite E3)": gemini_e3_map,
        "gpt-5.6-luna": load_llm_verdict_map(e9_dir / "gpt-5.6-luna" / "cache.jsonl"),
        "gpt-6-luna": load_llm_verdict_map(e9_dir / "gpt-6-luna" / "cache.jsonl"),
        "deepseek-v4.1-flash": load_llm_verdict_map(e9_dir / "deepseek-flash" / "cache.jsonl"),
        "claude-haiku-5-5": load_llm_verdict_map(e9_dir / "claude-haiku-5-5" / "cache.jsonl"),
    }


def manifest_to_arrays(
    rows: Sequence[dict[str, str]],
    llm_map: dict[str, Any],
    *,
    with_prob: bool = False,
) -> dict[str, Any]:
    """
    Converts manifest row dicts + LLM lookup map into aligned numpy arrays.
    Missing or failed LLM lookups deterministically fall back to specialist predictions.
    """
    y = np.array([int(r["y"]) for r in rows], dtype=int)
    s = np.array([int(r["specialist_pred"]) for r in rows], dtype=int)
    pc = np.array([float(r["specialist_prob_cal"]) for r in rows], dtype=float)
    pu = np.array([float(r["specialist_prob_uncal"]) for r in rows], dtype=float)
    ds = np.array([r["dataset"] for r in rows])

    if with_prob:
        l_arr = np.array(
            [
                llm_map[r["canonical_url_sha256"]][0]
                if r["canonical_url_sha256"] in llm_map
                else int(r["specialist_pred"])
                for r in rows
            ],
            dtype=int,
        )
        pl_arr = np.array(
            [
                llm_map[r["canonical_url_sha256"]][1]
                if r["canonical_url_sha256"] in llm_map
                else float(r["specialist_prob_cal"])
                for r in rows
            ],
            dtype=float,
        )
    else:
        l_arr = np.array(
            [
                llm_map.get(r["canonical_url_sha256"], int(r["specialist_pred"]))
                for r in rows
            ],
            dtype=int,
        )
        pl_arr = None

    n_fallback = sum(1 for r in rows if r["canonical_url_sha256"] not in llm_map)
    return {
        "y": y,
        "s": s,
        "pc": pc,
        "pu": pu,
        "ds": ds,
        "l": l_arr,
        "pl": pl_arr,
        "n_fallback": n_fallback,
    }


@dataclass
class V2DataBundle:
    """In-memory cache of frozen manifests, policies, and LLM response maps."""

    repo_root: Path
    policy_freeze_v2: dict[str, Any]
    tau_be_v2: dict[str, Any]
    e3_rows: list[dict[str, str]]
    e4_rows_all: list[dict[str, str]]
    e4_rows_p49_excluded: list[dict[str, str]]
    e9_rows: list[dict[str, str]]
    p49_excluded_shas: set[str]
    gem_e3_verdicts: dict[str, int]
    gem_e3_verdict_probs: dict[str, tuple[int, float]]
    e3_source_counts: dict[str, int]
    gem_e4_verdicts: dict[str, int]
    gem_e4_verdict_probs: dict[str, tuple[int, float]]
    e9_vendor_maps: dict[str, dict[str, int]]

    @classmethod
    def load(cls, repo_root: Path = REPO_ROOT) -> V2DataBundle:
        man_dir = repo_root / "data" / "manifests"
        pol_dir = repo_root / "results" / "policies"

        policy_freeze_v2 = read_json(pol_dir / "policy_freeze_v2.json")
        tau_be_v2 = read_json(pol_dir / "tau_BE_v2.json")
        p49 = set(policy_freeze_v2["p49_cal_test_shared_urls"]["list"])

        e3_rows = read_manifest_csv(man_dir / "e3_test_all_26886.csv")
        e4_rows_all = read_manifest_csv(man_dir / "e4_cal_all_13446.csv")
        e4_rows_p49_excluded = [r for r in e4_rows_all if r["canonical_url_sha256"] not in p49]
        e9_rows = read_manifest_csv(man_dir / "e9_shared_test_cohort_3000.csv")

        gem_e3_vp, src_counts = load_composite_gemini_e3(repo_root=repo_root, with_prob=True)
        gem_e3_v = {k: v[0] for k, v in gem_e3_vp.items()}
        gem_e4_vp = load_e4_gemini_cal(repo_root=repo_root, with_prob=True)
        gem_e4_v = {k: v[0] for k, v in gem_e4_vp.items()}
        e9_maps = load_e9_vendor_maps(gem_e3_v, repo_root=repo_root)

        return cls(
            repo_root=repo_root,
            policy_freeze_v2=policy_freeze_v2,
            tau_be_v2=tau_be_v2,
            e3_rows=e3_rows,
            e4_rows_all=e4_rows_all,
            e4_rows_p49_excluded=e4_rows_p49_excluded,
            e9_rows=e9_rows,
            p49_excluded_shas=p49,
            gem_e3_verdicts=gem_e3_v,
            gem_e3_verdict_probs=gem_e3_vp,
            e3_source_counts=src_counts,
            gem_e4_verdicts=gem_e4_v,
            gem_e4_verdict_probs=gem_e4_vp,
            e9_vendor_maps=e9_maps,
        )


DEFAULT_IGNORE_TIMESTAMP_KEYS: frozenset[str] = frozenset(
    {"generated_utc", "generated_at_utc", "timestamp_utc", "computed_utc"}
)


def compare_json_values(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-12,
    ignore_keys: frozenset[str] = DEFAULT_IGNORE_TIMESTAMP_KEYS,
    path: str = "$",
) -> list[str]:
    """
    Recursively compares `actual` against `expected` within float tolerance `atol`.

    Ignores keys in `ignore_keys` (such as generation timestamps).
    Returns a list of human-readable mismatch strings (empty list means exact match).
    """
    diffs: list[str] = []

    if isinstance(actual, dict) and isinstance(expected, dict):
        a_keys = {k for k in actual if k not in ignore_keys}
        e_keys = {k for k in expected if k not in ignore_keys}
        missing = sorted(e_keys - a_keys)
        extra = sorted(a_keys - e_keys)
        if missing:
            diffs.append(f"{path}: missing keys {missing}")
        if extra:
            diffs.append(f"{path}: unexpected extra keys {extra}")
        for k in sorted(a_keys & e_keys):
            diffs.extend(
                compare_json_values(
                    actual[k],
                    expected[k],
                    atol=atol,
                    ignore_keys=ignore_keys,
                    path=f"{path}.{k}",
                )
            )
        return diffs

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            diffs.append(f"{path}: length {len(actual)} != {len(expected)}")
            return diffs
        for idx, (a_item, e_item) in enumerate(zip(actual, expected)):
            diffs.extend(
                compare_json_values(
                    a_item,
                    e_item,
                    atol=atol,
                    ignore_keys=ignore_keys,
                    path=f"{path}[{idx}]",
                )
            )
        return diffs

    if isinstance(actual, bool) or isinstance(expected, bool):
        if type(actual) is not type(expected) or actual != expected:
            diffs.append(f"{path}: {actual!r} != {expected!r}")
        return diffs

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        a_f, e_f = float(actual), float(expected)
        if math.isnan(a_f) and math.isnan(e_f):
            return diffs
        if math.isinf(a_f) and math.isinf(e_f) and a_f == e_f:
            return diffs
        if abs(a_f - e_f) > atol:
            diffs.append(f"{path}: {actual!r} != {expected!r} (diff={abs(a_f - e_f):.3e} > {atol:.3e})")
        return diffs

    if actual != expected:
        diffs.append(f"{path}: {actual!r} != {expected!r}")
    return diffs


def verify_json_file(
    computed: Any,
    frozen_path: Path,
    *,
    atol: float = 1e-12,
    ignore_keys: frozenset[str] = DEFAULT_IGNORE_TIMESTAMP_KEYS,
) -> list[str]:
    """Compares an in-memory computed dict/list against a frozen JSON file on disk."""
    expected = read_json(frozen_path)
    return compare_json_values(computed, expected, atol=atol, ignore_keys=ignore_keys)
