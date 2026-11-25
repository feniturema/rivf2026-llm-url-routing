"""Shared statistical, entropy, OOF calibration, and cascade evaluation metrics for v2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def binary_entropy_bits(p: np.ndarray | Sequence[float] | float, eps: float = 1e-12) -> Any:
    """
    Computes binary Shannon entropy H(p) in bits using probability clipping [eps, 1-eps].

    Matches policy_freeze_v2.json (`eps=1e-12`).
    """
    arr = np.asarray(p, dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    ent = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    return float(ent) if np.ndim(arr) == 0 else ent


def compute_f1(y: np.ndarray, y_pred: np.ndarray) -> Any:
    """
    Computes binary F1 along the last axis.
    Supports both 1D arrays `(N,)` -> float and 2D bootstrap arrays `(B, N)` -> 1D array `(B,)`.
    """
    tp = ((y == 1) & (y_pred == 1)).sum(axis=-1)
    fp = ((y == 0) & (y_pred == 1)).sum(axis=-1)
    fn = ((y == 1) & (y_pred == 0)).sum(axis=-1)
    res = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(res) if np.ndim(res) == 0 else res


def confusion_counts(y: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Returns TP, FP, FN, TN counts for binary predictions."""
    return {
        "TP": int(((y == 1) & (y_pred == 1)).sum()),
        "FP": int(((y == 0) & (y_pred == 1)).sum()),
        "FN": int(((y == 1) & (y_pred == 0)).sum()),
        "TN": int(((y == 0) & (y_pred == 0)).sum()),
    }


def compute_mcc(conf: Mapping[str, int]) -> float:
    """Computes Matthews correlation coefficient from TP/FP/FN/TN dict."""
    num = conf["TP"] * conf["TN"] - conf["FP"] * conf["FN"]
    den_sq = (
        (conf["TP"] + conf["FP"])
        * (conf["TP"] + conf["FN"])
        * (conf["TN"] + conf["FP"])
        * (conf["TN"] + conf["FN"])
    )
    if den_sq <= 0:
        return 0.0
    return float(num / math.sqrt(den_sq))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via Binomial test on discordant pairs (B, C)."""
    total = b + c
    if total <= 0:
        return 1.0
    return float(binomtest(b, total, 0.5).pvalue)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact two-sided Clopper-Pearson binomial confidence interval."""
    lo = float(beta.ppf(alpha / 2.0, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)) if k < n else 1.0
    return [lo, hi]


def fit_oof_isotonic_by_corpus(
    y: np.ndarray,
    prob_uncal: np.ndarray,
    dataset: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Fits 5-fold stratified out-of-fold IsotonicRegression per corpus on uncalibrated probabilities.

    Eliminates in-sample isotonic probability coupling during CAL_ROUTER parameter selection.
    """
    oof = np.zeros(len(y), dtype=float)
    for corpus in sorted(set(dataset)):
        idx = np.where(dataset == corpus)[0]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(prob_uncal[idx], y[idx]):
            iso = IsotonicRegression(out_of_bounds="clip").fit(prob_uncal[idx][tr], y[idx][tr])
            oof[idx[te]] = iso.predict(prob_uncal[idx][te])
    return oof


def paired_bootstrap_f1_cis(
    y: np.ndarray,
    y_spec: np.ndarray,
    y_llm: np.ndarray,
    cascades: Mapping[str, np.ndarray],
    *,
    point_f1s: Mapping[str, float],
    f1_spec: float,
    f1_full: float,
    n_boot: int = 10_000,
    seed: int = 42,
    chunk: int = 250,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Computes vectorized paired bootstrap 95% percentile CIs for (F1_selective - F1_spec)
    and (F1_selective - F1_full). Preserves chunked RNG draw order (`chunk=250`, `seed=42`).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    acc: dict[str, dict[str, list[np.ndarray]]] = {
        k: {"vs_spec": [], "vs_full": []} for k in cascades
    }

    done = 0
    while done < n_boot:
        bs = min(chunk, n_boot - done)
        ii = rng.integers(0, n, (bs, n))
        yy = y[ii]
        fs = compute_f1(yy, y_spec[ii])
        ff = compute_f1(yy, y_llm[ii])
        for k, casc_pred in cascades.items():
            fc = compute_f1(yy, casc_pred[ii])
            acc[k]["vs_spec"].append(fc - fs)
            acc[k]["vs_full"].append(fc - ff)
        done += bs

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for k in cascades:
        out[k] = {}
        for vs_key, base_f1 in (("vs_spec", f1_spec), ("vs_full", f1_full)):
            vals = np.concatenate(acc[k][vs_key])
            out[k][f"dF1_{vs_key}"] = {
                "point": float(point_f1s[k] - base_f1),
                "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            }
    return out


def evaluate_cascade_cohort(
    y: np.ndarray,
    y_spec: np.ndarray,
    y_llm: np.ndarray,
    scores_bits: np.ndarray,
    taus: Mapping[str, float],
    *,
    fallback_rows: int = 0,
    boot: int = 10_000,
    boot_keys: tuple[str, ...] = ("b0.25", "tauBE"),
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluates full-escalation breakeven 2x2 counts and selective entropy routing points
    (using strict inequality `scores_bits > tau`), plus optional paired bootstrap CIs.
    """
    n = len(y)
    sc = y_spec == y
    lc = y_llm == y

    a_cnt = int((sc & lc).sum())
    b_cnt = int((~sc & lc).sum())
    c_cnt = int((sc & ~lc).sum())
    d_cnt = int((~sc & ~lc).sum())

    e_cnt = b_cnt + d_cnt
    rho = b_cnt / e_cnt if e_cnt else float("nan")
    eps = c_cnt / (a_cnt + c_cnt)
    eps_star = rho * e_cnt / (n - e_cnt)

    f1_spec = float(compute_f1(y, y_spec))
    f1_full = float(compute_f1(y, y_llm))

    out: dict[str, Any] = {
        "N": n,
        "llm_fallback_rows": int(fallback_rows),
        "spec_acc": (a_cnt + c_cnt) / n,
        "A": a_cnt,
        "B": b_cnt,
        "C": c_cnt,
        "D": d_cnt,
        "rho": rho,
        "eps": eps,
        "eps_star": eps_star,
        "full_net": b_cnt - c_cnt,
        "full_mcnemar_p": mcnemar_exact_p(b_cnt, c_cnt),
        "sign_matches_breakeven": bool((b_cnt - c_cnt > 0) == (eps < eps_star)),
        "F1_spec": f1_spec,
        "F1_full": f1_full,
    }

    cascades: dict[str, np.ndarray] = {}
    point_f1s: dict[str, float] = {}
    for k, tau in taus.items():
        esc = scores_bits > tau
        casc_pred = np.where(esc, y_llm, y_spec)
        b1 = int((esc & ~sc & lc).sum())
        c1 = int((esc & sc & ~lc).sum())
        b0 = int((~esc & ~sc & lc).sum())
        c0 = int((~esc & sc & ~lc).sum())
        f1_sel = float(compute_f1(y, casc_pred))
        out[k] = {
            "tau": tau,
            "escalated": int(esc.sum()),
            "esc_rate": float(esc.mean()),
            "top_B": b1,
            "top_C": c1,
            "top_net": b1 - c1,
            "top_mcnemar_p": mcnemar_exact_p(b1, c1),
            "bottom_net": b0 - c0,
            "F1_selective": f1_sel,
        }
        if k in boot_keys:
            cascades[k] = casc_pred
            point_f1s[k] = f1_sel

    if boot and cascades:
        cis = paired_bootstrap_f1_cis(
            y,
            y_spec,
            y_llm,
            cascades,
            point_f1s=point_f1s,
            f1_spec=f1_spec,
            f1_full=f1_full,
            n_boot=boot,
            seed=seed,
        )
        for k, ci_dict in cis.items():
            out[k].update(ci_dict)

    return out
