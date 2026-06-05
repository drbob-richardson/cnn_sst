# Paper Roadmap — Interpretable Spatial Masking for SST/ENSO Forecasting

**Target venue:** *Environmetrics* (statistical framing).
**Core thesis:** Spatial dependence makes attribution ill-posed (credit is ambiguous among
correlated neighbouring cells); a **total-variation + sparsity penalty on a learned,
in-model spatial mask** (= structured-sparsity attribution; TV ↔ fused-lasso ↔ MRF prior)
yields more **faithful** attributions. Demonstrated on **synthetic ground truth** (method
comparison) and validated on **ENSO** against a **fused-lasso + correlation statistical
reference** and known physics.
**Contributions (the claims we stand on):**
1. **Faithful interpretability** — the TV+sparsity in-model mask recovers true drivers more faithfully than post-hoc and non-TV learned baselines (synthetic ground truth).
2. **Tiling = efficiency knob** — tiled masks match pixel/no-gate **predictive accuracy and AUROC at ~250× less training cost**.
**Explicit NON-goal:** improving predictive accuracy. The mask is for interpretability + efficiency, *not* better forecasts — do not make or imply a prediction-improvement claim.

**Decided:** fused lasso, **no Bayes**. Bayesian/MCMC dropped (intractable at SST resolution;
fused lasso is the tractable, conceptually-matched sibling of the TV mask).

## Owner legend
- ✅ **Claude** — straight-through: I implement/run/draft autonomously; you spot-check after.
- ⏸ **Approval** — I stop for your expert sign-off (before and/or after).
- 🚦 **Gate** — go/no-go that depends on results; **your call** whether to proceed.

---

## Phase 0 — Lock framing & claims  *(DECIDED 2026-06-04)*
- [x] ⏸ Target = **Environmetrics** + structured-sparsity framing. *(AIES / Env. Data Science remain fallbacks.)*
- [x] ⏸ Thesis direction approved as the route to explore (the spatial-dependence + TV-faithfulness message above). Formal **claim list** still to be drafted/signed off (revisited in Phase 5 claims audit).
- [x] ⏸ Scope = **one paper** (sims + ENSO triangulation). No split.

### Future follow-ups — OUT OF SCOPE for this paper (do not pull in)
- **3D CNNs on environmental models** to uncover important *temporal nonlinear functionals* (separate paper, author already working on it).
- **Health longitudinal forecasting** (separate paper).
- *Implication:* keep this paper's masking strictly **spatial** (2D); resist scope-creep toward temporal-functional or longitudinal extensions.

## Phase 1 — Literature & positioning
- [ ] ✅ Run a **focused, verified** deep-research pass on geoscience XAI SOTA (confirm Toms & Barnes 2020 JAMES, Barnes 2020, **Mamalakis et al. 2022/23 fidelity benchmarks**, McGovern 2019; surface 2023–25 work; check whether TV/spatial-smoothness attribution already exists in geoscience).
- [ ] ✅ Assemble the **must-cite + distinguish** list: TVmax (Martins 2020), STG (Yamada 2020), L2X, INVASE, L0 (Louizos), Fong–Vedaldi, Mamalakis, Toms&Barnes.
- [ ] ⏸ Approve the **"how we differ" positioning** paragraph (esp. vs. TVmax and vs. Mamalakis).

## Phase 2 — Code consolidation & reproducibility
- [ ] ⏸ **Locate or authorize reconstruction of the missing El Niño driver** `learn_pixel_masks_loop_serial.py` (produced your Figs 5–7; not on disk). *Biggest risk in the plan* — see Risks below.
- [ ] ✅ Integrate the masking code into `masking/` (sim_1 done; reconstruct/clean the El Niño trainer; config-driven, seeded).
- [x] ✅ Aggregate existing ENSO results (`write_up/results/_aggregated/`). **Finding (2026-06-04):** results are *weak* — F1 collapses to **0.000 for all gates at leads ≥9** (degenerate all-negative classifiers), accuracy is base-rate (0.67–0.79), AUROC mostly 0.6–0.7. Only **lead 3** (F1 ~0.53–0.67, AUROC ~0.85) and partially **lead 6** discriminate. **The mask does not improve prediction** (mean F1: none 0.142, pixel 0.119, tile_2x2 0.133, tile_4x4 0.074); pixel gate costs ~250–300× more for no gain. Single run each (no seeds). *(Note: `collect_elnino_results.py` leaderboard has an F1 parse-key bug — reads `f1`/prints nan; patch later.)*
- [ ] ⏸ Review **text-vs-metrics discrepancies** — CONFIRMED: §6 numbers don't match files (e.g., pixel lead-6 text 0.709 vs file 0.718; tile_4x4 lead-6 text 0.728 vs file 0.786) and the "pixel improves accuracy" claim is unsupported. Decide handling.
- [x] 🚦 **EARLY GATE — RESOLVED (2026-06-04):** prediction-improvement is *not* a claim, so no heavy ENSO rehab. **Decision: option (B+light).** ENSO becomes (i) an interpretability illustration at the **working leads (3, partly 6)** where the model discriminates, and (ii) the **tiling efficiency demonstration** (accuracy + AUROC + train-time by gate, at working leads, with a few seeds for error bars). Simulations carry the primary evidence. Do NOT headline accuracy averaged over degenerate long leads.

## Phase 3 — Simulations: our method vs. existing methods  *(primary evidence)*
- [x] ⏸ **Baseline set APPROVED** (no fused-lasso in sims): post-hoc = saliency/IG/Grad-CAM/gradient-SHAP; learned = L0/STG + CBAM-style spatial attention; ablation = our mask ±TV.
- [x] ✅ **Spurious-correlation regime** (`masking/sim_spurious.py`, `results/sim_spurious.csv`, figs in `write_up/sim_spurious_*.png`). **NEGATIVE / IMPORTANT FINDING (2026-06-04):** our TV+sparsity mask is **captured by the cheap spurious shortcut** (IoU_causal **0.057**, IoU_spurious 0.409) while post-hoc **IG/SHAP correctly localize the causal region** (0.71). **This is the OPPOSITE of the paper's current §4 claim** ("integrated gates better for causal interpretability"). Mechanism: sparsity rewards concentrating on the *cheapest* predictive region = the spurious blob; the causal XOR (two blobs) is more expensive. ⇒ **The "robust to spurious correlation / causal interpretability" claim does NOT hold as tested and must be re-examined or dropped.** (Design-dependent on spurious strength — see fork below.)
- [x] 🚦 **FORK — RESOLVED → (b) (2026-06-04):** **drop the spurious-correlation / "causal interpretability" claim.** The paper does NOT claim robustness to spurious correlation. Keep `sim_spurious.py` + figures as documentation; mention the shortcut-capture as a **one-paragraph limitation** in §Discussion. Sim-1 (modular TV + frontier) carries the simulation evidence. *User may revisit option (a) — the flip_p characterization — later while thinking about WHY the capture happens.*
- **§4 ACTION:** the current paper's §4 (Strip-and-Noise / spurious-correlation study, Tables 2–3, Figs 2–4) asserts the now-refuted claim → **rewrite or remove §4** when we get to writing; do not carry its "integrated gates resist spurious correlation" takeaway.
- [ ] ⏸ Approve adding a **linear-signal simulation regime** — *deferred*; first run uses the existing nonlinear XOR regime.
- [x] ✅ Implemented in `masking/sim_attribution_benchmark.py` (IoU + saliency-mass vs ground truth; **insertion/deletion deferred** — ground-truth IoU/mass is the cleaner primary metric).
- [x] ✅ Ran seeded study (15 seeds × 60 epochs) → `results/sim_attribution_benchmark.csv`.
- [x] ✅ Significance: paired Wilcoxon (ours+TV vs each) built into the runner.
- [x] 🚦 **GATE — PASSED (core hypothesis), with a nuance (2026-06-04):**
  - **TV ablation is decisive:** signal-IoU 0.453 (no-TV) → **0.747 (TV), p=0.0001**, distractor-IoU stays 0. TV improves faithful localization. ✅
  - **BUT ours is *not* the top localizer:** L0 (0.836) and STG (0.840) beat ours on signal-IoU (p=0.002) — *however they tank prediction* (STG acc 0.579/F1 0.276). Ours has the **best distractor suppression + best predictive accuracy** (0.716/0.62).
  - **Honest claim = trade-off**, not a clean sweep: "TV improves faithfulness; among distractor-suppressing methods ours uniquely preserves predictive skill." Do **not** claim best-localization.
- [x] ✅ **Modular ±TV factorial run** (`results/sim_attribution_benchmark_tv.csv`). **Finding (2026-06-04):** TV is **NOT** a universal modular prior as tested — it decisively helps only our sigmoid mask (IoU_sig 0.435→0.753, p=1e-4); marginal/ns for L0 (+0.02) & STG (+0.005), slightly negative for attention. Caveats: L0/STG near a **ceiling** (~0.84), single **untuned** λ_tv=5e-2, attention-TV formulation. ⇒ **Do NOT claim "TV improves all attribution methods."** Surviving claims: (1) TV essential to our mask (decisive ablation); (2) ours = best **faithfulness×accuracy** frontier point (IoU_sig 0.75, IoU_dis 0, acc 0.72; L0/STG match IoU but lose predictive skill).
- [x] 🚦 **FORK — RESOLVED via λ sweep** (`results/sim_lambda_sweep.csv`, 10 seeds):
  - **(a) Modular CONFIRMED (measured):** the factorial null was a **ceiling artifact**. Off the ceiling (weak sparsity) TV significantly improves L0 (+0.07, p=0.012) and STG (+0.13, p=0.008); benefit saturates at high sparsity. ⇒ **"TV is a general faithfulness prior across in-model gates (ours, L0, STG)" is now a supportable, measured claim.**
  - **(b) Frontier:** at matched F1, L0/STG still out-localize ours (IoU 0.84 vs 0.68) — *no* "we localize best" claim. BUT ours reaches **F1≈0.61** (IoU 0.73–0.80, distractor 0) which **L0/STG never exceed (~0.46 ceiling)**. ⇒ honest **Pareto** story: ours owns the high-accuracy region; sparsity-only gates own high-IoU/low-F1.
  - Caveats for limitations: single untuned λ_tv=5e-2 on the +TV side; modest absolute effects; one synthetic regime.
- **Does "ours" have an advantage? YES — significant (best-config, paired, 10 seeds):** ours F1 **0.633** vs L0 0.483 / STG 0.457 (**p=0.002**), and ours F1 > ungated baseline 0.539 (soft mask = helpful denoising; L0/STG can't exceed baseline). Cost: ours peak IoU 0.797 vs 0.845 (**p=0.043**, real deficit, must report). ⇒ **ours = faithful attribution at NO predictive cost; L0/STG = marginally better localization but collapse the predictor below baseline.** Keep ours.
- **Two contributions to write:** (1) **TV as a modular faithfulness prior** (helps L0/STG/ours, significant off the ceiling); (2) **our TV+sparsity sigmoid gate is the frontier sweet spot** — faithful attribution while keeping the predictor intact. (+ ENSO application + tiling efficiency.)
- [ ] ⏸ (then) revisit the **linear-signal regime** + insertion/deletion metric.

## Phase 4 — ENSO application: rehab + statistical reference
- [ ] ⏸ Approve the **evaluation protocol**: data choice (`sst.mon.mean.nc` vs `sst_data.nc`), class-imbalance handling (weighting/threshold/report AUPRC+F1), multiple seeds + error bars.
- [x] ✅ **Statistical reference** (`masking/sst_statistical_reference.py` → `write_up/sst_statistical_reference.png`, `results/sst_statistical_reference.npz`). **Lagged-correlation map is clean & physical** (equatorial band at lead 3 broadening off-equator by lead 6). **Fused-lasso logistic is limited by collinearity** (p≫n; even with strong L1+TV it can't cleanly localize — itself an illustration of the attribution-under-dependence problem). ⇒ **use correlation as the primary reference; footnote the fused-lasso.**
- [x] ✅ **Masking driver RECONSTRUCTED** (`masking/sst_masking.py`, replaces the lost script): DeepCNN (5-conv) + pixel/2×2/4×4 gates + TV/sparsity + τ-anneal, on the 30×90 field. **Running** leads {3,6} × 4 seeds → `results/sst_masking.csv`, `results/sst_masking_maps.npz`, `write_up/sst_masks_vs_reference.png` (acc/F1/AUROC + mask↔correlation agreement). *Note: downsampled reconstruction; not byte-identical to the lost original.*
- [~] ✅ Agreement metric = Pearson(mask, |correlation|) on the common 30×90 grid (in the run above).
- [x] ✅ **Fixed methodology** (temporal holdout + real per-seed variation). Honest numbers: **lead 3 AUROC ~0.87 / F1 ~0.69** (good), **lead 6 AUROC ~0.65** (marginal — out-of-time 6-mo ENSO is hard).
- [x] ✅ **TV-weight sweep** (`masking/sst_tv_sweep.py`, `results/sst_tv_sweep.csv`, `write_up/sst_tv_sweep.png`). TV smooths masks as intended; **λ_tv=0.2 selected** (best mask↔correlation agreement at flat AUROC; λ=0.5 starts hurting AUROC). Final masks re-run at 0.2.
- [x] 🚦 **ENSO interpretability — honest verdict:** masks agree with the correlation reference **moderately at lead 6 (~0.5)** but **~0 at lead 3, robustly across all λ_tv.** *Why:* at lead 3 the ENSO signal is so strong/persistent across the equatorial Pacific that attribution is genuinely ill-posed — the mask need not (and does not) pin to the "true" band. **This is on-thesis** (it *demonstrates* attribution ambiguity under spatial dependence). **Decision: interpretability claim rests on the SIMULATIONS; ENSO = predictive result + statistical-reference comparison + honest discussion of the lead-3 ambiguity.** Do NOT claim "masks cleanly recover ENSO physics."
- **§ENSO ACTION:** rewrite §5–6 to this honest framing (predictive AUROC w/ error bars + temporal holdout; correlation-reference agreement at lead 6; lead-3 ambiguity as illustration). Drop old prediction-improvement claims + 0.25° figures (or cite 0.25° only for the tiling-runtime point). Tiling-efficiency: cite the **0.25°** timings (pixel ~600s vs tiled ~3s); at 30×90 all gates ~23s.
- [x] ✅ **DEFENSIBLE MODEL (3-way temporal split + selection)** (`masking/sst_model_select.py`; `results/sst_model_select_*.csv`). Train 60% / val 20% (selection) / **untouched test 20%** (report). **Selected a ~5k-param compact CNN; the inherited 288k DeepCNN generalizes WORST** (val AUROC 0.890 vs 0.874) — over-parameterization confirmed.
- [x] 🚦 **ENSO verdict CORRECTED (supersedes the lead-3-ambiguity claim above):** with the properly-sized model the **masks RECOVER the equatorial signal** — agreement **0.57–0.63 at lead 3** (was ~0 with the deep model!), 0.26–0.50 at lead 6. The earlier lead-3 non-agreement was a **model artifact** (over-param net fitting noise), NOT intrinsic ambiguity. New point: *faithful attribution presupposes a well-specified predictor.* Honest test-block prediction: AUROC lead 3 ≈ 0.82, lead 6 ≈ 0.62 (threshold F1/acc degraded by temporal shift → report AUROC). **§ENSO + Discussion + Abstract rewritten accordingly; paper compiles.**
- [ ] ⏸ **Expert physics check:** do the masks recover credible ENSO precursors, and are agreements/divergences with the statistical reference defensible? *(I can't supply this judgment.)*
- [ ] 🚦 **GATE:** Are the ENSO masks scientifically credible and is predictive performance honestly reportable (mask "at no predictive cost" is fine; degenerate classifiers are not)? **Your call to proceed.**
- [ ] ⏸ Decide: add a **robustness-to-missing-inputs / occlusion** experiment, or **drop that claim** from the abstract (currently unsupported).

## Phase 5 — Writing  *(main.tex in write_up/)*
- [x] ✅ Reframed **Abstract + Intro** drafted (structured-sparsity / spatial-dependence angle; positions vs TVmax/L0/STG/Fong-Vedaldi). `write_up/main.tex` compiles → `main.pdf`.
- [x] ⏸ **Methods**: added the structured-sparsity paragraph (TV ↔ fused-lasso ↔ MRF). *You should sanity-check the MAP/MRF claim wording.*
- [x] ✅ **New simulation section** written from real results (Table + 3 figures `sim1_attributions/faithfulness/frontier.png`) replacing old §3; **old §4 (spurious) removed**. Honest limitation paragraph included.
- [x] ✅ **§ENSO rewritten** (honest framing: temporal-holdout AUROC + error bars; correlation reference; lead-6 agreement / lead-3 ambiguity; tiling cost). Old §5–6 overclaims + 0.25° result-PNG figures removed; new figures `sst_statistical_reference.png`, `sst_masks_vs_reference.png` + Table.
- [x] ✅ **§Discussion + §Conclusion written** (2 contributions + 2 honest limitations: spurious capture, attribution ambiguity under redundancy). Empty §Interpretability stub removed. Whole paper compiles → `main.pdf`.
- [ ] ⏸ Remaining writing polish: **claims audit**, **reformat to Environmetrics (Wiley) template/style**, verify new bib details, polish figures, tidy supplementary/title-page template artifacts.
- [x] ✅ Added must-cite refs (TVmax, STG, L2X, INVASE, Grad-CAM, fused lasso, Mamalakis, Toms&Barnes, Ham, Captum, Dabkowski). *Verify a few bib details (esp. TVmax venue/authors).*
- [ ] ⏸ **Claims audit**: every claim traces to a result; cut overclaims.
- [ ] ✅ Reformat to **Environmetrics (Wiley)** template + bibliography style; add the must-cite refs.

## Phase 6 — Pre-submission
- [ ] ✅ Repo reproduction pass (README "to reproduce"; the repo is already organized).
- [ ] ⏸ Final read-through + go/no-go to submit.
- [ ] ⏸ Cover letter + suggested reviewers (I draft, you finalize).

---

## Decision gates (the three that can change the project)
1. **Phase 3 GATE** — TV improves faithfulness on ground truth? (makes/breaks the thesis)
2. **Phase 4 GATE** — ENSO masks credible + honestly reportable? (makes/breaks the application)
3. **Phase 0/5** — final claim set you're willing to stand behind.

## Key risks / honest flags
- **Missing El Niño driver** (`learn_pixel_masks_loop_serial.py`): the Fig 5–7 results exist in `write_up/results/`, but the script that made them is gone. Reconstruction from the paper spec may not byte-reproduce the figures; if exact reproduction matters, we need the original or a re-run we both trust.
- **Linearity of the statistical reference**: fused-lasso/correlation capture *linear* predictable signal. Sims carry the nonlinear-faithfulness claim; the ENSO reference is **convergent validity**, not primary proof. Keep roles distinct.
- **TVmax (Martins 2020)** scoops the pure-ML "TV-on-learned-attention" claim — the statistical framing + application are why this still works at Environmetrics. Must cite & distinguish.
- **Compute**: everything runs locally on the M3 Max; only the pixel-gate ENSO sweep is overnight-scale.

## Suggested starting point
Phase 1 (verified SOTA search) + Phase 2 (aggregate existing ENSO results, surface discrepancies) are **cheap, high-value, and mostly straight-through** — they sharpen the framing and tell us whether the existing numbers even support the story before we build baselines. Recommend starting there.
