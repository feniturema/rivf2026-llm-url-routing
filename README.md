# Selective Uncertainty-Aware LLM Routing for Phishing URL Triage

Official code, frozen thresholds, zero-shot prompts, request ledgers, and anonymized response caches for the **IEEE RIVF 2026** paper:

> **Selective Uncertainty-Aware LLM Routing for Phishing URL Triage: The Base-Rate Breakeven Regime and Advantage Partitioning**  
> Jiaju Ma, Faculty of Information Science and Technology, National University of Malaysia (UKM)

The paper PDF is not distributed in this repository because copyright has been transferred to IEEE. It will be available from IEEE Xplore after publication, and its DOI will be added here. The LaTeX source in `paper/main.tex` is kept because the number-binding checks verify it against the result files.

---

## Key Results at a Glance

Replacing a high-accuracy local specialist classifier (>95% accuracy) with a commercial cloud LLM degrades pooled detection unless the LLM's intrusion rate $\varepsilon$ on specialist-correct inputs stays below the exact **base-rate breakeven threshold** $\varepsilon^\ast = \rho\,E/(N - E)$. Confining escalation to the calibrated high-uncertainty head yields net gains for every corpus and vendor:

| Policy (Pooled E3 `TEST`, $N = 26{,}886$) | Escalated | Rescues ($B$) | Regressions ($C$) | Net ($\Delta_{\mathrm{net}}$) | $F_1$ | List-Price Spend / 1M URLs |
|---|---:|---:|---:|---:|---:|---:|
| Local specialist (`Char TF-IDF + LogReg`) | 0.00% | 0 | 0 | 0 | 0.9537 | \$0 |
| Full replace-on-escalation (`Gemini 3.5 Flash-Lite`) | 100.00% | 971 | 1,439 | −468 | 0.9360 | \$271 |
| **Selective entropy routing ($\tau^\ast_{\mathrm{BE}} = 0.4567$)** | **12.45%** | **759** | **235** | **+524** | **0.9736** | **\$33.7** |
| Fixed-budget entropy routing ($\tau = 0.1997$, $b=0.25$) | 25.30% | 899 | 414 | +485 | 0.9721 | \$68.5 |
| Fixed-budget $\tau$ + soft fusion ($\alpha = 0.50$) | 25.30% | 776 | 218 | +558 | 0.9748 | \$68.5 |

All five evaluated commercial LLMs across four vendors (**Google Gemini 3.5 Flash-Lite**, **OpenAI GPT-5.6 Luna**, **OpenAI GPT-6 Luna**, **DeepSeek-V4.1-Flash**, **Anthropic Claude Haiku 5.5**) lose under pooled full escalation yet recover strictly positive net gains at $\tau^\ast_{\mathrm{BE}}$ across all 20 model–corpus cells on the shared cross-vendor benchmark (`E9`, $N = 3{,}000$).

---

## One-Command Offline Reproduction (No API Keys Needed)

Every table, metric, figure coordinate, and frozen SHA-256 integrity seal in the paper reproduces deterministically offline in under 20 seconds using [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13:

```bash
# 1. Create the pinned environment (numpy==2.2.6, scikit-learn==1.7.2, xgboost, scipy, pandas, pytest)
uv sync --extra dev

# 2. Verify all 13 offline reproduction stages + frozen SHA-256 seals + paper number binding
.venv/bin/python experiments/reproduce_v2_all.py --verify

# 3. Run the full test suite (46 unit, invariance, evidence-binding, and v2 reproduction tests)
.venv/bin/pytest -q tests/
```

To regenerate all v2 result JSONs into a scratch directory without overwriting the committed frozen artifacts:

```bash
.venv/bin/python experiments/reproduce_v2_all.py --output-dir /tmp/rivf2026_repro
```

---

## Repository Structure

```text
.
├── configs/
│   ├── price_table.yaml                   # Per-model list-price table ($/1M tokens)
│   ├── production_experiments_freeze.json # Frozen manifest & prompt hashes
│   └── prompts/                           # Content-addressed v1/v2 zero-shot prompts & JSON schemas
├── data/
│   └── manifests/                         # Canonicalized 60/10/10/20 splits (PWD2021, EBBU2017, URL-Phish 2025)
├── experiments/
│   ├── reproduce_v2_all.py                # 13-stage end-to-end camera-ready offline reproduction runner
│   ├── eval_v2_main.py                    # E3/E9 breakeven, partitioning, E8 stability, bootstrap CIs (Table IV)
│   ├── eval_v2_policies.py                # OOF-tuned policy-space & calibration comparison (Table V)
│   ├── eval_v2_supplement.py              # Budget frontier, error decomposition, single-corpus check (Tables VI–VII)
│   ├── eval_v2_paper_extras.py            # E4 calibration check, confusion matrices, cost calculation
│   ├── eval_v2_oos_samplesize.py          # Out-of-sample sign-accuracy vs. calibration sample size
│   ├── eval_e9_v2_interim.py              # Five-model four-vendor E9 summary
│   └── run_*harvest_v2.py                 # v2 API harvesters (--dry-run supported; no live calls needed for replay)
├── paper/
│   └── main.tex                           # IEEEtran camera-ready LaTeX source (checked by check_number_binding_v2.py)
├── results/
│   ├── E3/, E4/, E9/, pilot/, E5/         # August 2026 response caches/ledgers (inputs to Option C / E4) & v1 archive
│   ├── artifacts/                         # Single-feature shortcut audit (Table II) & cross-corpus lineage matrix
│   ├── policies/                          # Frozen thresholds (policy_freeze_v2.json, tau_BE_v2.json)
│   ├── specialist_baselines/              # 10-seed first-tier specialist benchmarks (Table III)
│   └── v2_runs/                           # Authoritative camera-ready v2 JSON metrics, E3 backfill, E8/E9 caches, gen_out.json
├── scripts/
│   ├── freeze_policy_thresholds_v2.py     # Strict quantile threshold freezer (DEV_POLICY)
│   ├── compute_tau_be_v2.py               # 5-fold OOF breakeven threshold estimator (CAL_ROUTER)
│   ├── make_option_c_selections.py        # Deterministic Option C E3 selection partitioner
│   ├── gen_tables_v2.py                   # Generates LaTeX table rows & TikZ coordinates from JSON
│   ├── check_number_binding_v2.py         # Binds prose & table numbers in paper/main.tex to JSONs
│   └── probe_champion_artifacts.py        # Single-feature shortcut & cross-corpus overlap audit (Table II)
├── src/
│   ├── data/                              # Canonical URL hashing & leakage-controlled splits
│   ├── llm/                               # Shared harvester, cache, schema parser, cost ledger
│   ├── replay/                            # Pure offline cascade replay engine
│   ├── routing/                           # Calibrated uncertainty & policy routers
│   └── v2/                                # Shared offline evaluation I/O & statistical metrics
└── tests/                                 # 46 automated tests covering replay, binding, harvesters, and v2 parity
```

> **Note on `v2` vs. archived `v1` files:** All camera-ready numbers in the paper come from `results/v2_runs/`, `results/policies/{policy_freeze_v2,tau_BE_v2}.json`, `results/artifacts/`, and `results/specialist_baselines/` via `experiments/reproduce_v2_all.py`. Earlier non-`_v2` scripts and `results/{E3,E4,E5,E9}/*.json` are retained for initial-submission provenance and regression tests (`tests/test_evidence_binding.py`).

---

## Dataset Note

Some phishing URLs (`y = 1`) from the URL-Phish 2025 corpus in `data/manifests/*.csv` contain third-party (attacker) Telegram bot tokens verbatim inside the captured URL strings. They are retained unmodified to preserve the frozen SHA-256 manifest hashes and are dataset artifacts rather than credentials of this project (see `.github/secret_scanning.yml`). Response caches and request ledgers under `results/` store only `canonical_url_sha256` digests rather than raw URLs.

---

## Implementation Notes / Errata

- The 15 Claude Haiku 5.5 `E9` rows reported as schema failures were responses with 4 signals rejected by the pre-fix validator (fixed in `src/llm/schema.py`); published numbers keep the original fallback-to-specialist behavior.
- `tau*_BE` ($\tau^\ast_{\mathrm{BE}}$) ranks escalation candidates by out-of-fold isotonic entropy, while the net-gain term $B - C$ uses the manifest (in-sample calibrated) specialist verdicts on `CAL_ROUTER`, as implemented in `scripts/compute_tau_be_v2.py`.

---

## Citation

```bibtex
@inproceedings{ma2026selective,
  author    = {Ma, Jiaju},
  title     = {Selective Uncertainty-Aware {LLM} Routing for Phishing {URL} Triage: The Base-Rate Breakeven Regime and Advantage Partitioning},
  booktitle = {Proc. 2026 RIVF International Conference on Computing and Communication Technologies (RIVF)},
  year      = {2026}
}
```
