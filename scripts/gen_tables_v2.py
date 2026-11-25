#!/usr/bin/env python3
"""
Generates LaTeX table rows and TikZ/pgfplots coordinate strings from frozen v2 JSONs.

Default output: results/v2_runs/gen_out.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    POLICIES_DIR,
    REPO_ROOT,
    RESULTS_V2_DIR,
    read_json,
    verify_json_file,
    write_json,
)

DEFAULT_OUT = RESULTS_V2_DIR / "gen_out.json"


def _fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def _fmt_signed(x: int) -> str:
    prefix = "+" if x > 0 else ("-" if x < 0 else "")
    return prefix + _fmt_int(abs(x))


def _fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{100 * x:.{decimals}f}\\%"


def _fmt_pval(p: float) -> str:
    if p == 0:
        return "0"
    exp = math.floor(math.log10(p))
    mant = p / 10**exp
    if round(mant, 2) >= 10:
        mant /= 10
        exp += 1
    return f"{mant:.2f}\\times10^{{{exp}}}"


def run(
    v2_dir: Path = RESULTS_V2_DIR,
    policies_dir: Path = POLICIES_DIR,
    *,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds exact LaTeX table fragments and figure coordinates from v2 JSONs."""
    art = artifacts or {}
    m_data = art.get("v2_main_metrics") or read_json(v2_dir / "v2_main_metrics.json")
    p_data = art.get("v2_policies") or read_json(v2_dir / "v2_policies.json")
    f_data = art.get("v2_frontier_and_error") or read_json(v2_dir / "v2_frontier_and_error.json")
    x_data = art.get("v2_paper_extras") or read_json(v2_dir / "v2_paper_extras.json")
    _ = art.get("oos_sample_size") or read_json(v2_dir / "oos_sample_size.json")
    _ = art.get("tau_BE_v2") or read_json(policies_dir / "tau_BE_v2.json")

    out: dict[str, Any] = {}

    def _breakeven_row(
        name: str,
        rec: dict[str, Any],
        top25: int,
        bot25: int,
        topbe: int,
        esc25: float,
        escbe: float,
    ) -> str:
        p_val = rec["full_mcnemar_p"] if "full_mcnemar_p" in rec else rec["p"]
        return (
            f"{name} & {_fmt_int(rec['N'])} & {_fmt_pct(rec['spec_acc'])} & "
            f"{_fmt_pct(rec['rho'])} & {_fmt_pct(rec['eps_star'])} & {_fmt_pct(rec['eps'])} & "
            f"${_fmt_signed(rec['full_net'])}$ \\ (${_fmt_pval(p_val)}$) & "
            f"${_fmt_signed(top25)}$ \\ ({100 * esc25:.1f}\\%) & ${_fmt_signed(bot25)}$ & "
            f"${_fmt_signed(topbe)}$ \\ ({100 * escbe:.1f}\\%) \\\\"
        )

    b_rows: list[str] = []
    for key, label in (
        ("PWD2021", "PWD2021"),
        ("EBBU2017", "EBBU2017"),
        ("URLPhish2025", "URL-Phish 25"),
    ):
        rec = m_data["E3"][key]
        b_rows.append(
            _breakeven_row(
                label + " (Gemini)",
                rec,
                rec["b0.25"]["top_net"],
                rec["b0.25"]["bottom_net"],
                rec["tauBE"]["top_net"],
                rec["b0.25"]["esc_rate"],
                rec["tauBE"]["esc_rate"],
            )
        )
    rec_pool = m_data["E3"]["pooled"]
    b_rows.append("\\midrule")
    b_rows.append(
        _breakeven_row(
            "\\textbf{Pooled E3 \\texttt{TEST}}",
            rec_pool,
            rec_pool["b0.25"]["top_net"],
            rec_pool["b0.25"]["bottom_net"],
            rec_pool["tauBE"]["top_net"],
            rec_pool["b0.25"]["esc_rate"],
            rec_pool["tauBE"]["esc_rate"],
        )
    )
    e4_chk = x_data["E4_check"]
    b_rows.append(
        _breakeven_row(
            "E4 \\texttt{CAL} check$^{\\ddagger}$",
            e4_chk | {"full_mcnemar_p": e4_chk["p"]},
            e4_chk["b0.25"]["top_net"],
            e4_chk["b0.25"]["bottom_net"],
            e4_chk["tauBE"]["top_net"],
            e4_chk["b0.25"]["esc_rate"],
            e4_chk["tauBE"]["esc_rate"],
        )
    )
    b_rows.append("\\midrule")
    for key, label in (
        ("gemini-3.5-flash-lite (composite E3)", "E9: Gemini 3.5 Flash-Lite"),
        ("gpt-5.6-luna", "E9: GPT-5.6 Luna"),
        ("gpt-6-luna", "E9: GPT-6 Luna"),
        ("deepseek-v4.1-flash", "E9: DeepSeek-V4.1-Flash"),
        ("claude-haiku-5-5", "E9: Claude Haiku 5.5"),
    ):
        rec = m_data["E9"][key]["pooled"]
        b_rows.append(
            _breakeven_row(
                label,
                rec,
                rec["b0.25"]["top_net"],
                rec["b0.25"]["bottom_net"],
                rec["tauBE"]["top_net"],
                rec["b0.25"]["esc_rate"],
                rec["tauBE"]["esc_rate"],
            )
        )
    out["BREAKEVEN"] = "\n".join(b_rows)

    # ---- frontier table
    fr = {q["label"]: q for q in f_data["frontier"]}
    keep_frontier = (
        ("entropy b=0.0", "0.00 (Spec)"),
        ("entropy b=0.05", "0.05"),
        ("entropy b=0.1", "0.10"),
        ("entropy tau*_BE", "$\\tau^{\\ast}_{\\mathrm{BE}}$"),
        ("entropy b=0.15", "0.15"),
        ("entropy b=0.2", "0.20 (peak)"),
        ("entropy b=0.25", "\\textbf{0.25 ($\\tau$)}"),
        ("entropy b=0.35", "0.35"),
        ("entropy b=0.5", "0.50"),
        ("entropy b=0.8", "0.80--0.95"),
        ("entropy b=1.0", "1.00 (Full)"),
    )
    fl: list[str] = []
    for key, lab in keep_frontier:
        q = fr[key]
        bold = "0.25" in lab and "textbf" in lab
        nrr = (
            "---"
            if q["NRR_esc"] is None
            else f"${'+' if q['NRR_esc'] >= 0 else '-'}{100 * abs(q['NRR_esc']):.1f}\\%$"
        )
        cells = [
            lab,
            f"{q['esc_pct']:.2f}",
            _fmt_int(q["B"]),
            _fmt_int(q["C"]),
            f"${_fmt_signed(q['net'])}$" if q["net"] else "0",
            nrr,
            f"{q['F1']:.4f}",
            f"${'+' if q['dF1_vs_spec'] >= 0 else '-'}{abs(q['dF1_vs_spec']):.4f}$".replace("0.", "."),
        ]
        if bold:
            cells = [cells[0]] + [
                f"\\textbf{{{c}}}" if not c.startswith("$") else f"$\\mathbf{{{c.strip('$')}}}$"
                for c in cells[1:]
            ]
        fl.append(" & ".join(cells) + " \\\\")
    q_rand = fr["random 25% (seed 42)"]
    fl.append("\\midrule")
    fl.append(
        " & ".join(
            [
                "Random 0.25",
                f"{q_rand['esc_pct']:.2f}",
                _fmt_int(q_rand["B"]),
                _fmt_int(q_rand["C"]),
                f"${_fmt_signed(q_rand['net'])}$",
                f"$-{100 * abs(q_rand['NRR_esc']):.1f}\\%$",
                f"{q_rand['F1']:.4f}",
                f"$-{abs(q_rand['dF1_vs_spec']):.4f}$".replace("0.", "."),
            ]
        )
        + " \\\\"
    )
    out["FRONTIER"] = "\n".join(fl)

    # ---- policies table
    p_rows = p_data["rows"]

    def _policy_row(lab: str, key: str, bold: bool = False) -> str:
        rec = p_rows[key]
        cells = [
            lab,
            f"{rec['queried_pct']:.2f}",
            _fmt_int(rec["B"]),
            _fmt_int(rec["C"]),
            f"{rec['F1']:.4f}",
        ]
        if bold:
            cells = [f"\\textbf{{{x}}}" for x in cells]
        return " & ".join(cells) + " \\\\"

    pl = [
        _policy_row("Full replace-on-escalation", "full_replace"),
        _policy_row(
            f"Full + LLM conf.\\ gate ($c_L{{=}}{p_rows['full_conf_gate']['c_L']:.2f}$)",
            "full_conf_gate",
        ),
        _policy_row(
            f"Full + soft fusion ($\\alpha{{=}}{p_rows['full_soft_fusion']['alpha']:.2f}$)",
            "full_soft_fusion",
        ),
        "\\midrule",
        _policy_row("Uncalibrated entropy (matched)", "uncalibrated_entropy_matched_tauBE"),
        _policy_row("Temp.\\ Scaling entropy (matched)", "temp_scaling_entropy_matched_tauBE"),
        _policy_row("Entropy $\\tau^{\\ast}_{\\mathrm{BE}}$ (Isotonic)", "entropy_tau_tauBE", True),
        _policy_row("Learned LogReg gate ($P(B){>}P(C)$)", "learned_logreg_gate"),
        _policy_row("Learned HGB gate ($P(B){>}P(C)$)", "learned_hgb_gate"),
        "\\midrule",
        _policy_row("Entropy $\\tau$, $b{=}0.25$ (Isotonic)", "entropy_tau_b0.25"),
        _policy_row(
            f"$\\tau$ + soft fusion ($\\alpha{{=}}{p_rows['tau_soft_fusion_b0.25']['alpha']:.2f}$)",
            "tau_soft_fusion_b0.25",
        ),
    ]
    out["POLICIES"] = "\n".join(pl)

    # ---- cell C composition
    comp = f_data["error_composition_full_escalation"]
    cl: list[str] = []
    for key, lab in (
        ("PWD2021", "PWD2021"),
        ("EBBU2017", "EBBU2017"),
        ("URLPhish2025", "URL-Phish 25"),
    ):
        c = comp[key]
        sh = _fmt_pct(c["missed_phish_share_of_C"], 1)
        sh_fmt = (
            f"$\\mathbf{{{sh.replace(chr(92) + '%', '')}\\%}}$"
            if key == "URLPhish2025"
            else f"${sh}$"
        )
        cl.append(
            f"{lab} & ${_fmt_signed(c['delta_missed_phishing'])}$ & "
            f"${_fmt_signed(c['delta_false_alarms'])}$ & {_fmt_int(c['C'])} & {sh_fmt} \\\\"
        )
    c_pool = comp["pooled"]
    cl.append("\\midrule")
    cl.append(
        f"Pooled E3 \\texttt{{TEST}} & $\\mathbf{{{_fmt_signed(c_pool['delta_missed_phishing'])}}}$ & "
        f"$\\mathbf{{{_fmt_signed(c_pool['delta_false_alarms'])}}}$ & "
        f"{_fmt_int(c_pool['C'])} & ${_fmt_pct(c_pool['missed_phish_share_of_C'], 1)}$ \\\\"
    )
    out["CELLC"] = "\n".join(cl)

    # ---- fig 3 points (x = pi/1.5, y = eps/3), curve rho pooled
    def _fig3_pt(rec: dict[str, Any]) -> list[float]:
        return [100 * (1 - rec["spec_acc"]) / 1.5, 100 * rec["eps"] / 3]

    fig3 = {d: _fig3_pt(m_data["E3"][d]) for d in ("PWD2021", "EBBU2017", "URLPhish2025")}
    for k in m_data["E9"]:
        fig3[k] = _fig3_pt(m_data["E9"][k]["pooled"])
    fig3["rho_pooled"] = m_data["E3"]["pooled"]["rho"]
    out["FIG3"] = fig3

    # ---- fig 4 coordinates
    coords: list[str] = []
    for key in (
        "entropy b=0.0", "entropy b=0.05", "entropy b=0.1", "entropy b=0.15",
        "entropy b=0.2", "entropy b=0.25", "entropy b=0.3", "entropy b=0.35",
        "entropy b=0.4", "entropy b=0.45", "entropy b=0.5", "entropy b=0.6",
        "entropy b=0.7", "entropy b=0.8", "entropy b=1.0",
    ):
        q = fr[key]
        coords.append(f"({q['esc_pct'] * 0.064:.3f}, {(q['F1'] - 0.930) * 100:.3f})")
    out["FIG4"] = " ".join(coords)
    out["FIG4_RANDOM"] = (
        f"(0.000, {(f_data['F1_specialist'] - 0.93) * 100:.3f}) "
        f"({q_rand['esc_pct'] * 0.064:.3f}, {(q_rand['F1'] - 0.93) * 100:.3f}) "
        f"({6.4:.3f}, {(fr['entropy b=1.0']['F1'] - 0.93) * 100:.3f})"
    )
    q_prim = fr["entropy b=0.25"]
    out["FIG4_PRIMARY"] = [q_prim["esc_pct"] * 0.064, (q_prim["F1"] - 0.93) * 100]
    q_be = fr["entropy tau*_BE"]
    out["FIG4_BE"] = [q_be["esc_pct"] * 0.064, (q_be["F1"] - 0.93) * 100]
    out["FIG4_BASE"] = (f_data["F1_specialist"] - 0.93) * 100
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify against committed frozen JSON without writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run()
    if args.verify:
        diffs = verify_json_file(result, DEFAULT_OUT)
        if diffs:
            print("FAIL gen_tables_v2 verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS gen_tables_v2 matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=1)
    for k, v in result.items():
        print(f"===== {k}\n{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
