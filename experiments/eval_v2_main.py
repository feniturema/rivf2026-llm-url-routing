#!/usr/bin/env python3
"""
v2 main offline evaluation (camera-ready). Reads frozen artifacts only; zero API calls.

E3 = Option C composite (amendment A-2). Operating points come from policy_freeze_v2.json
and tau_BE_v2.json. LLM failure falls back to specialist prediction.
Paired bootstrap: 10,000 resamples (seed 42).
Default output: results/v2_runs/v2_main_metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    FROZEN_ARTIFACTS,
    LEGACY_TAU_04363,
    REPO_ROOT,
    RESULTS_V2_DIR,
    V2DataBundle,
    manifest_to_arrays,
    read_json,
    read_jsonl,
    sha256_file,
    verify_json_file,
    write_json,
)
from src.v2.metrics import (
    binary_entropy_bits,
    clopper_pearson_ci,
    evaluate_cascade_cohort,
)

DEFAULT_OUT = RESULTS_V2_DIR / "v2_main_metrics.json"
CORPORA = ("PWD2021", "EBBU2017", "URLPhish2025")
INPUT_HASH_PATHS = (
    "results/policies/policy_freeze_v2.json",
    "results/policies/tau_BE_v2.json",
    "results/v2_runs/E3_BACKFILL/gemini-3.5-flash-lite/cache.jsonl",
    "results/E3/e3_gemini_test_cache.jsonl",
)


def _analyse_rows(
    rows: list[dict[str, str]],
    verdict_map: dict[str, int],
    taus: dict[str, float],
    *,
    boot: int = 10_000,
) -> dict[str, Any]:
    arr = manifest_to_arrays(rows, verdict_map, with_prob=False)
    entropy_bits = binary_entropy_bits(arr["pc"])
    return evaluate_cascade_cohort(
        arr["y"],
        arr["s"],
        arr["l"],
        entropy_bits,
        taus,
        fallback_rows=arr["n_fallback"],
        boot=boot,
        boot_keys=("b0.25", "tauBE"),
        seed=42,
    )


def run(bundle: V2DataBundle | None = None, *, boot: int = 10_000) -> dict[str, Any]:
    """Computes full v2 main evaluation dictionary without writing to disk."""
    data = bundle if bundle is not None else V2DataBundle.load()
    fz = data.policy_freeze_v2
    tbe = data.tau_be_v2
    taus = {
        "b0.25": fz["metadata"]["primary_tau"],
        "tauBE": tbe["tau_star_BE"],
        "legacy0.4363": LEGACY_TAU_04363,
    }

    inputs_sha = {p: sha256_file(data.repo_root / p) for p in INPUT_HASH_PATHS}
    for p, digest in inputs_sha.items():
        if p in FROZEN_ARTIFACTS and digest != FROZEN_ARTIFACTS[p]:
            raise RuntimeError(f"Frozen artifact SHA-256 drift for {p}: {digest}")

    result: dict[str, Any] = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "taus": taus,
        "e3_sources_unique_urls": data.e3_source_counts,
        "inputs_sha256": inputs_sha,
    }

    # ---- E3 TEST composite Gemini evaluation ----
    e3_results: dict[str, Any] = {
        "pooled": _analyse_rows(data.e3_rows, data.gem_e3_verdicts, taus, boot=boot)
    }
    for corpus in CORPORA:
        sub_rows = [r for r in data.e3_rows if r["dataset"] == corpus]
        e3_results[corpus] = _analyse_rows(sub_rows, data.gem_e3_verdicts, taus, boot=boot)
    result["E3"] = e3_results

    # ---- Pre-registered decision branch rule (b=0.25) ----
    ci_lo = e3_results["pooled"]["b0.25"]["dF1_vs_spec"]["ci95"][0]
    all_pos = all(e3_results[d]["b0.25"]["top_net"] > 0 for d in CORPORA)
    result["branch"] = "B1" if (ci_lo > 0 and all_pos) else ("B2" if ci_lo > 0 else "B3")

    # ---- OOS prediction: E4 (P-49 excluded) predicts TEST full-escalation sign ----
    oos_results: dict[str, Any] = {}
    for group in ("pooled", *CORPORA):
        sub_rows = (
            data.e4_rows_p49_excluded
            if group == "pooled"
            else [r for r in data.e4_rows_p49_excluded if r["dataset"] == group]
        )
        e4_eval = _analyse_rows(sub_rows, data.gem_e4_verdicts, taus, boot=0)
        pred_pos = e4_eval["eps"] < e4_eval["eps_star"]
        obs_pos = e3_results[group]["full_net"] > 0
        oos_results[group] = {
            "E4_eps": e4_eval["eps"],
            "E4_eps_star": e4_eval["eps_star"],
            "E4_full_net": e4_eval["full_net"],
            "predicted_positive": bool(pred_pos),
            "TEST_positive": bool(obs_pos),
            "correct": bool(pred_pos == obs_pos),
        }
    result["OOS"] = oos_results

    # ---- E9 five-model table on shared 3,000-row TEST cohort ----
    e9_results: dict[str, Any] = {}
    for vendor_name, vmap in data.e9_vendor_maps.items():
        per_vendor: dict[str, Any] = {
            "pooled": _analyse_rows(data.e9_rows, vmap, taus, boot=boot)
        }
        for corpus in CORPORA:
            sub_rows = [r for r in data.e9_rows if r["dataset"] == corpus]
            per_vendor[corpus] = _analyse_rows(sub_rows, vmap, taus, boot=0)
        e9_results[vendor_name] = per_vendor
    result["E9"] = e9_results

    # ---- E8 stability (5 repeats) + Option C bridge agreement ----
    e8_records = read_jsonl(
        data.repo_root / "results" / "v2_runs" / "E8" / "gemini-3.5-flash-lite" / "cache.jsonl"
    )
    by_url: dict[str, dict[str, Any]] = {}
    for rec in e8_records:
        entry = by_url.setdefault(
            rec["canonical_url_sha256"],
            {"stratum": rec["e8_stratum"], "v": []},
        )
        entry["v"].append(rec["parsed_verdict"])

    stability: dict[str, Any] = {}
    for stratum in ("top_entropy", "random_rest", "all"):
        items = [x for x in by_url.values() if stratum == "all" or x["stratum"] == stratum]
        flips = sum(1 for x in items if len(set(x["v"])) > 1)
        stability[stratum] = {
            "urls": len(items),
            "repeats_each": 5,
            "urls_with_any_flip": flips,
            "flip_rate": flips / len(items),
            "flip_rate_cp95": clopper_pearson_ci(flips, len(items)),
        }
    bridge_summary = read_json(
        data.repo_root / "results" / "v2_runs" / "BRIDGE" / "gemini-3.5-flash-lite" / "summary.json"
    )["bridge"]
    result["E8"] = stability
    result["bridge_aug_vs_v2"] = bridge_summary
    return result


def _print_summary_row(label: str, cohort: dict[str, Any]) -> None:
    b25, tbe = cohort["b0.25"], cohort["tauBE"]
    ci25 = b25.get("dF1_vs_spec", {}).get("ci95")
    cibe = tbe.get("dF1_vs_spec", {}).get("ci95")
    s_ci25 = f" CI[{ci25[0]:+.4f},{ci25[1]:+.4f}]" if ci25 else ""
    s_cibe = f" CI[{cibe[0]:+.4f},{cibe[1]:+.4f}]" if cibe else ""
    print(
        f"{label:13s} N={cohort['N']:5d} acc={cohort['spec_acc']:.2%} rho={cohort['rho']:.2%} "
        f"eps*={cohort['eps_star']:.2%} eps={cohort['eps']:.2%} full={cohort['full_net']:+5d} "
        f"p={cohort['full_mcnemar_p']:.1e} ok={'Y' if cohort['sign_matches_breakeven'] else 'N'} "
        f"| b.25 esc={b25['esc_rate']:.2%} top={b25['top_net']:+4d} bot={b25['bottom_net']:+5d} "
        f"F1 {cohort['F1_spec']:.4f}->{b25['F1_selective']:.4f}{s_ci25} "
        f"| BE esc={tbe['esc_rate']:.2%} top={tbe['top_net']:+4d} "
        f"F1->{tbe['F1_selective']:.4f}{s_cibe} | full F1 {cohort['F1_full']:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify against committed frozen JSON without writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run()
    if args.verify:
        diffs = verify_json_file(result, DEFAULT_OUT)
        if diffs:
            print("FAIL eval_v2_main verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS eval_v2_main matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=2)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()[:16]
    print("wrote", args.output, digest)
    print("\n== E3 composite Gemini; sources", result["e3_sources_unique_urls"])
    for grp in ("pooled", *CORPORA):
        _print_summary_row(grp, result["E3"][grp])
    print("BRANCH:", result["branch"])
    print(
        "\n== OOS (E4 -> TEST sign):",
        {
            g: (
                round(v["E4_eps"], 4),
                round(v["E4_eps_star"], 4),
                v["predicted_positive"],
                v["TEST_positive"],
                v["correct"],
            )
            for g, v in result["OOS"].items()
        },
    )
    print("\n== E9 (3,000 rows)")
    for k, x in result["E9"].items():
        print(" ", k)
        _print_summary_row("  pooled", x["pooled"])
        for d in CORPORA:
            _print_summary_row("  " + d, x[d])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
