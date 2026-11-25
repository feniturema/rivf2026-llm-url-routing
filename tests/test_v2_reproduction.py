"""Regression and zero-drift reproduction tests for the RIVF 2026 v2 evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import (
    eval_e9_v2_interim,
    eval_v2_main,
    eval_v2_oos_samplesize,
    eval_v2_paper_extras,
    eval_v2_policies,
    eval_v2_supplement,
    reproduce_v2_all,
)
from scripts import (
    check_number_binding_v2,
    compute_tau_be_v2,
    freeze_policy_thresholds_v2,
    gen_tables_v2,
    make_option_c_selections,
)
from src.replay.engine import replay_cascade
from src.routing.policies import compute_routing_scores, select_escalation_mask
from src.v2.io import (
    POLICIES_DIR,
    REPO_ROOT,
    RESULTS_V2_DIR,
    SELECTIONS_DIR,
    V2DataBundle,
    compare_json_values,
    read_json,
    sha256_file,
)
from src.v2.metrics import (
    binary_entropy_bits,
    clopper_pearson_ci,
    compute_f1,
    compute_mcc,
    confusion_counts,
    evaluate_cascade_cohort,
    mcnemar_exact_p,
)


@pytest.fixture(scope="module")
def v2_bundle() -> V2DataBundle:
    return V2DataBundle.load(REPO_ROOT)


def test_frozen_hashes_and_e9_final_manifest() -> None:
    errors = reproduce_v2_all.check_frozen_seals(REPO_ROOT)
    assert errors == [], f"Frozen seal failures: {errors}"


def test_v2_metrics_and_core_routing_fixes() -> None:
    # Entropy clipping & scalar/vector equivalence
    assert binary_entropy_bits(0.5) == pytest.approx(1.0)
    ent_vec = binary_entropy_bits(np.array([0.0, 0.5, 1.0]))
    assert ent_vec[1] == pytest.approx(1.0)
    assert ent_vec[0] == pytest.approx(ent_vec[2])
    assert ent_vec[0] > 0.0

    # Strict > escalation semantics in evaluate_cascade_cohort, select_escalation_mask, and replay_cascade
    y = np.array([1, 0, 1, 0])
    s = np.array([1, 0, 0, 1])
    l = np.array([1, 0, 1, 0])
    scores = np.array([0.2, 0.5, 0.8, 0.5])  # scores == 0.5 should NOT escalate under strict >

    cohort = evaluate_cascade_cohort(y, s, l, scores, {"t": 0.5}, boot=0)
    assert cohort["t"]["escalated"] == 1
    assert cohort["t"]["top_B"] == 1
    assert cohort["t"]["top_C"] == 0

    mask_strict = select_escalation_mask(scores, 0.5, strict=True)
    mask_nonstrict = select_escalation_mask(scores, 0.5, strict=False)
    assert mask_strict.tolist() == [False, False, True, False]
    assert mask_nonstrict.tolist() == [False, True, True, True]

    rep_strict = replay_cascade(y, s, l, scores, threshold=0.5, strict=True)
    rep_nonstrict = replay_cascade(y, s, l, scores, threshold=0.5, strict=False)
    assert rep_strict.n_escalated == 1
    assert rep_nonstrict.n_escalated == 3

    # R2_margin uses calibrated probabilities when supplied
    uncal = np.array([0.5, 0.9])
    cal = np.array([0.1, 0.5])
    r2_cal = compute_routing_scores("R2_margin", uncal_probs=uncal, cal_probs=cal)
    assert r2_cal[0] < r2_cal[1]  # cal=0.5 is most uncertain -> higher score

    # Confusion, F1, MCC, McNemar, Clopper-Pearson
    conf = confusion_counts(y, l)
    assert conf == {"TP": 2, "FP": 0, "FN": 0, "TN": 2}
    assert compute_f1(y, l) == pytest.approx(1.0)
    assert compute_mcc(conf) == pytest.approx(1.0)
    assert mcnemar_exact_p(0, 0) == pytest.approx(1.0)
    lo, hi = clopper_pearson_ci(7, 200)
    assert 0.01 < lo < 0.035 < hi < 0.08


def test_v2_evaluators_reproduce_frozen_artifacts(v2_bundle: V2DataBundle) -> None:
    sel_lists, sel_meta = make_option_c_selections.compute_selections(REPO_ROOT)
    for name, info in sel_meta.items():
        assert sha256_file(SELECTIONS_DIR / f"{name}.txt") == info["sha256"]
    assert sum(len(v) for k, v in sel_lists.items() if k != "bridge") == 26669

    pf = freeze_policy_thresholds_v2.run(REPO_ROOT)
    assert compare_json_values(pf, read_json(POLICIES_DIR / "policy_freeze_v2.json"), atol=1e-12) == []

    tbe = compute_tau_be_v2.run(v2_bundle)
    assert compare_json_values(tbe, read_json(POLICIES_DIR / "tau_BE_v2.json"), atol=1e-12) == []

    main_res = eval_v2_main.run(v2_bundle)
    assert compare_json_values(main_res, read_json(RESULTS_V2_DIR / "v2_main_metrics.json"), atol=1e-12) == []

    pol_res = eval_v2_policies.run(v2_bundle)
    assert compare_json_values(pol_res, read_json(RESULTS_V2_DIR / "v2_policies.json"), atol=1e-12) == []

    supp_res = eval_v2_supplement.run(v2_bundle)
    assert compare_json_values(supp_res, read_json(RESULTS_V2_DIR / "v2_frontier_and_error.json"), atol=1e-12) == []

    extra_res = eval_v2_paper_extras.run(v2_bundle)
    assert compare_json_values(extra_res, read_json(RESULTS_V2_DIR / "v2_paper_extras.json"), atol=1e-12) == []

    oos_res = eval_v2_oos_samplesize.run(v2_bundle, main_metrics=main_res)
    assert compare_json_values(oos_res, read_json(RESULTS_V2_DIR / "oos_sample_size.json"), atol=1e-12) == []

    e9_interim_res = eval_e9_v2_interim.run(v2_bundle)
    assert (
        compare_json_values(
            e9_interim_res,
            read_json(RESULTS_V2_DIR / "E9" / "e9_v2_interim_metrics.json"),
            atol=1e-12,
        )
        == []
    )

    gen_res = gen_tables_v2.run(
        v2_dir=RESULTS_V2_DIR,
        policies_dir=POLICIES_DIR,
        artifacts={
            "v2_main_metrics": main_res,
            "v2_policies": pol_res,
            "v2_frontier_and_error": supp_res,
            "v2_paper_extras": extra_res,
            "oos_sample_size": oos_res,
            "tau_BE_v2": tbe,
        },
    )
    assert compare_json_values(gen_res, read_json(RESULTS_V2_DIR / "gen_out.json"), atol=1e-12) == []


def _resolve_tex_path() -> Path:
    primary = REPO_ROOT / "paper" / "main.tex"
    return primary if primary.exists() else REPO_ROOT / "paper" / "main_final_audit.tex"


def test_number_binding_v2_runs_cleanly() -> None:
    tex_path = _resolve_tex_path()
    report = check_number_binding_v2.check_manuscript(tex_path)
    assert report["total_numbers"] >= 450
    assert report["unexpected_unmatched"] == {}
    assert check_number_binding_v2.main([str(tex_path)]) == 0
    # Verify known paper numbers from frozen A-2 / lineage tables are bounded, not broken
    assert "26886" not in report["unmatched"]
    assert "3346" not in report["unmatched"]


def test_number_binding_v2_flags_unexpected_number(tmp_path: Path) -> None:
    tex_text = _resolve_tex_path().read_text(encoding="utf-8")
    tampered = tex_text.replace(
        r"\end{abstract}",
        " Unbound synthetic check value 98765.4321.\n\\end{abstract}",
        1,
    )
    tmp_tex = tmp_path / "main_tampered.tex"
    tmp_tex.write_text(tampered, encoding="utf-8")

    report = check_number_binding_v2.check_manuscript(tmp_tex)
    assert "98765.4321" in report["unexpected_unmatched"]
    assert check_number_binding_v2.main([str(tmp_tex)]) == 1
