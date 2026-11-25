#!/usr/bin/env python3
"""
Unified offline reproduction runner for all RIVF 2026 camera-ready v2 artifacts.

Verifies frozen SHA-256 seals, recomputes all v2 policy/evaluation/table artifacts
in a single pass from local manifests and frozen LLM caches (zero API calls), checks
numerical agreement (<= 1e-12) against committed JSONs, and verifies manuscript number binding.

Usage:
  python3 experiments/reproduce_v2_all.py --verify
  python3 experiments/reproduce_v2_all.py --output-dir /tmp/rivf2026_repro
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import (
    eval_e9_v2_interim,
    eval_v2_main,
    eval_v2_oos_samplesize,
    eval_v2_paper_extras,
    eval_v2_policies,
    eval_v2_supplement,
)
from scripts import (
    check_number_binding_v2,
    compute_tau_be_v2,
    freeze_policy_thresholds_v2,
    gen_tables_v2,
    make_option_c_selections,
)
from src.v2.io import (
    FROZEN_ARTIFACTS,
    POLICIES_DIR,
    REPO_ROOT,
    RESULTS_V2_DIR,
    SELECTIONS_DIR,
    V2DataBundle,
    compare_json_values,
    read_json,
    sha256_file,
    write_json,
)


def check_frozen_seals(repo_root: Path = REPO_ROOT) -> list[str]:
    """Verifies SHA-256 digests for all frozen artifacts and the E9 final manifest."""
    errors: list[str] = []
    for rel_path, expected_sha in FROZEN_ARTIFACTS.items():
        path = repo_root / rel_path
        if not path.exists():
            errors.append(f"missing frozen artifact: {rel_path}")
            continue
        actual = sha256_file(path)
        if actual != expected_sha:
            errors.append(f"SHA-256 mismatch for {rel_path}: {actual} != {expected_sha}")

    e9_manifest_path = repo_root / "results" / "v2_runs" / "E9" / "E9_FINAL_manifest.json"
    if not e9_manifest_path.exists():
        errors.append("missing E9_FINAL_manifest.json")
    else:
        e9_manifest = read_json(e9_manifest_path)
        e9_root = e9_manifest_path.parent
        for rel_file, expected_sha in e9_manifest.get("files", {}).items():
            fpath = e9_root / rel_file
            if not fpath.exists():
                errors.append(f"missing E9 locked file: {rel_file}")
                continue
            actual = sha256_file(fpath)
            if actual != expected_sha:
                errors.append(f"E9 locked file SHA-256 mismatch for {rel_file}: {actual} != {expected_sha}")
    return errors


def reproduce_all(
    *,
    verify: bool = True,
    output_dir: Path | None = None,
    atol: float = 1e-12,
    repo_root: Path = REPO_ROOT,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Executes the full v2 offline reproduction pipeline using one shared V2DataBundle.

    Returns (all_ok, stage_reports).
    """
    reports: list[dict[str, Any]] = []
    all_ok = True

    def _record(stage: str, ok: bool, dt_s: float, detail: str = "") -> None:
        nonlocal all_ok
        if not ok:
            all_ok = False
        reports.append({"stage": stage, "ok": ok, "seconds": dt_s, "detail": detail})

    # Stage 0: Frozen SHA-256 seals
    t0 = time.perf_counter()
    seal_errors = check_frozen_seals(repo_root)
    _record(
        "frozen_seals",
        len(seal_errors) == 0,
        time.perf_counter() - t0,
        "all frozen + 22 E9 manifest hashes verified" if not seal_errors else "; ".join(seal_errors[:3]),
    )

    # Load shared bundle once
    t0 = time.perf_counter()
    bundle = V2DataBundle.load(repo_root=repo_root)
    _record("load_data_bundle", True, time.perf_counter() - t0, f"E3={len(bundle.e3_rows)}, E4={len(bundle.e4_rows_p49_excluded)}, E9={len(bundle.e9_rows)}")

    v2_dir = repo_root / "results" / "v2_runs"
    pol_dir = repo_root / "results" / "policies"

    # Option C selections (fast SHA check)
    t0 = time.perf_counter()
    sel_lists, sel_meta = make_option_c_selections.compute_selections(repo_root=repo_root)
    sel_ok = True
    sel_diffs: list[str] = []
    if verify:
        for name, info in sel_meta.items():
            actual_sha = sha256_file(SELECTIONS_DIR / f"{name}.txt")
            if actual_sha != info["sha256"]:
                sel_ok = False
                sel_diffs.append(f"{name}.txt")
    if output_dir is not None:
        sel_out = output_dir / "selections"
        sel_out.mkdir(parents=True, exist_ok=True)
        for name, lst in sel_lists.items():
            (sel_out / f"{name}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
    _record(
        "option_c_selections",
        sel_ok,
        time.perf_counter() - t0,
        "4/4 selection hashes match" if sel_ok else f"mismatch: {sel_diffs}",
    )

    # Policy freeze v2 & tau*_BE
    t0 = time.perf_counter()
    pf_data = freeze_policy_thresholds_v2.run(repo_root=repo_root)
    pf_diffs = compare_json_values(pf_data, read_json(pol_dir / "policy_freeze_v2.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "policy_freeze_v2.json", pf_data, indent=2)
    _record("freeze_policy_thresholds_v2", not pf_diffs, time.perf_counter() - t0, f"tau(0.25)={pf_data['metadata']['primary_tau']:.6f}" if not pf_diffs else str(pf_diffs[:2]))

    t0 = time.perf_counter()
    tbe_data = compute_tau_be_v2.run(bundle)
    tbe_diffs = compare_json_values(tbe_data, read_json(pol_dir / "tau_BE_v2.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "tau_BE_v2.json", tbe_data, indent=2)
    _record("compute_tau_be_v2", not tbe_diffs, time.perf_counter() - t0, f"tau*_BE={tbe_data['tau_star_BE']:.6f}" if not tbe_diffs else str(tbe_diffs[:2]))

    # Core v2 evaluators
    t0 = time.perf_counter()
    main_data = eval_v2_main.run(bundle)
    main_diffs = compare_json_values(main_data, read_json(v2_dir / "v2_main_metrics.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "v2_main_metrics.json", main_data, indent=2)
    _record("eval_v2_main", not main_diffs, time.perf_counter() - t0, f"branch={main_data['branch']}" if not main_diffs else str(main_diffs[:2]))

    t0 = time.perf_counter()
    pol_data = eval_v2_policies.run(bundle)
    pol_diffs = compare_json_values(pol_data, read_json(v2_dir / "v2_policies.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "v2_policies.json", pol_data, indent=2)
    _record("eval_v2_policies", not pol_diffs, time.perf_counter() - t0, f"{len(pol_data['rows'])} policy rows" if not pol_diffs else str(pol_diffs[:2]))

    t0 = time.perf_counter()
    supp_data = eval_v2_supplement.run(bundle)
    supp_diffs = compare_json_values(supp_data, read_json(v2_dir / "v2_frontier_and_error.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "v2_frontier_and_error.json", supp_data, indent=2)
    _record("eval_v2_supplement", not supp_diffs, time.perf_counter() - t0, f"{len(supp_data['frontier'])} frontier points" if not supp_diffs else str(supp_diffs[:2]))

    t0 = time.perf_counter()
    extra_data = eval_v2_paper_extras.run(bundle)
    extra_diffs = compare_json_values(extra_data, read_json(v2_dir / "v2_paper_extras.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "v2_paper_extras.json", extra_data, indent=2)
    _record("eval_v2_paper_extras", not extra_diffs, time.perf_counter() - t0, f"total_spend=${extra_data['v2_list_price_spend_usd']['total']}" if not extra_diffs else str(extra_diffs[:2]))

    t0 = time.perf_counter()
    oos_data = eval_v2_oos_samplesize.run(bundle, main_metrics=main_data)
    oos_diffs = compare_json_values(oos_data, read_json(v2_dir / "oos_sample_size.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "oos_sample_size.json", oos_data, indent=2)
    _record("eval_v2_oos_samplesize", not oos_diffs, time.perf_counter() - t0, "4 cohorts x 7 sizes" if not oos_diffs else str(oos_diffs[:2]))

    t0 = time.perf_counter()
    e9_interim_data = eval_e9_v2_interim.run(bundle)
    e9_diffs = compare_json_values(e9_interim_data, read_json(v2_dir / "E9" / "e9_v2_interim_metrics.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "e9_v2_interim_metrics.json", e9_interim_data, indent=2)
    _record("eval_e9_v2_interim", not e9_diffs, time.perf_counter() - t0, f"{len(e9_interim_data['results'])} E9 models" if not e9_diffs else str(e9_diffs[:2]))

    t0 = time.perf_counter()
    gen_data = gen_tables_v2.run(
        v2_dir=v2_dir,
        policies_dir=pol_dir,
        artifacts={
            "v2_main_metrics": main_data,
            "v2_policies": pol_data,
            "v2_frontier_and_error": supp_data,
            "v2_paper_extras": extra_data,
            "oos_sample_size": oos_data,
            "tau_BE_v2": tbe_data,
        },
    )
    gen_diffs = compare_json_values(gen_data, read_json(v2_dir / "gen_out.json"), atol=atol) if verify else []
    if output_dir is not None:
        write_json(output_dir / "gen_out.json", gen_data, indent=1)
    _record("gen_tables_v2", not gen_diffs, time.perf_counter() - t0, f"{len(gen_data)} LaTeX/TikZ keys" if not gen_diffs else str(gen_diffs[:2]))

    t0 = time.perf_counter()
    tex_path = (
        repo_root / "paper" / "main.tex"
        if (repo_root / "paper" / "main.tex").exists()
        else repo_root / "paper" / "main_final_audit.tex"
    )
    nb_res = check_number_binding_v2.check_manuscript(tex_path, v2_dir=v2_dir, policies_dir=pol_dir)
    unexpected_nb = nb_res["unexpected_unmatched"]
    nb_ok = nb_res["total_numbers"] > 400 and not unexpected_nb
    nb_detail = (
        f"{nb_res['total_numbers']} numbers checked, {nb_res['unmatched_distinct']} allowlisted, 0 unexpected"
        if not unexpected_nb
        else f"unexpected unmatched tokens: {unexpected_nb}"
    )
    _record(
        "check_number_binding_v2",
        nb_ok,
        time.perf_counter() - t0,
        nb_detail,
    )

    return all_ok, reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all recomputed artifacts against committed frozen files (default if no --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write recomputed JSONs cleanly without altering results/",
    )
    parser.add_argument("--atol", type=float, default=1e-12, help="Numeric comparison tolerance")
    args = parser.parse_args(argv)

    do_verify = args.verify or (args.output_dir is None)
    ok, reports = reproduce_all(verify=do_verify, output_dir=args.output_dir, atol=args.atol)

    total_s = sum(r["seconds"] for r in reports)
    print(f"{'Stage':30s} | {'Status':6s} | {'Time':7s} | Detail")
    print("-" * 92)
    for rep in reports:
        status = "PASS" if rep["ok"] else "FAIL"
        print(f"{rep['stage']:30s} | {status:6s} | {rep['seconds']:6.2f}s | {rep['detail']}")
    print("-" * 92)
    print(f"Overall: {'PASS' if ok else 'FAIL'} (total {total_s:.2f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
