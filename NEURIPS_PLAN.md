# NeurIPS 2026 Position Paper — Plan

**Title (locked):** *Identifiability for Brain Dynamics is Untestable Without Intervention: A Case for Synthetic-with-Known-Ground-Truth Benchmarks*

**Authors:** Mahta Ramezanian-Panahi, Pooneh Mousavi, Guillaume Dumas (assumed same as ICLR blog post).

**Track:** NeurIPS 2026 Position Paper Track. 9 pp, position-stating title, bold position in intro, optional Alternative Views section in main text, no checklist.

---

## 0. Status of inputs

| Input | Status |
|---|---|
| ICLR 2026 blog post + reviews | ✅ |
| NeurIPS Position Paper call (assumed similar to 2025) | ✅ |
| Position statement | ✅ locked: 3 claims (unfalsifiability + benchmark prescription + constructive paths) |
| Synthetic generator | ✅ locked: trained shPLRNN, used as **testbed** (not as a solution) |
| Latent dim `M` for the generator | ✅ M = 4 (or 8 if needed for richer dynamics) |
| Mixing | ✅ shPLRNN's `obs_model=linear` provides learned linear mixing |
| Multi-channel EEG dataset | 🟡 **default = BCI Competition IV-2a (22 channels, motor imagery)** — confirm or override |
| Interventions in experiments | ✅ **dropped** — not needed empirically; appears only in the "constructive paths" prescription |
| Biophysical sanity check (Jansen-Rit etc.) | ✅ deferred to future work |
| HierarchicalDSR reproduction (92.6%) | ✅ off the critical path |

---

## 1. Position statement

> **Identifiability claims for latent brain sources from purely observational scalp-level recordings are not falsifiable. The community should require synthetic-with-known-ground-truth or interventional benchmarks as a prerequisite for publishing such claims. Passive-paradigm recordings cannot deliver causal inference; intervention-capable paradigms (BCI, intracranial recording, TMS) and generative models that admit synthetic ground-truth construction are the legitimate paths forward.**

This is three claims:
1. **Negative.** Observational identifiability for brain dynamics is not falsifiable.
2. **Methodological.** Synthetic-with-known-ground-truth or interventional benchmarks should be required.
3. **Constructive.** BCI / iEEG / TMS, and generative models used as synthetic testbeds, are the way forward.

NeurIPS-required form: this paragraph (or a tightened version) appears in **bold** in the introduction; abstract opens with *"This position paper argues that…"*.

---

## 2. Crucial framing point (must be consistent across the paper)

**The shPLRNN is not a solution to the source-recovery problem.** It has no identifiability guarantees of its own. We use it strictly as a *testbed*: it is trained on real EEG to produce realistic-looking dynamics, and we then *define* its learned latents as the ground-truth sources for the synthetic benchmark. This framing must be consistent throughout the paper — never claim or imply that the shPLRNN "finds the true brain sources."

### What we can and cannot claim (with citations)

**What we CAN claim — three tiers of defense:**

1. **"Formal surrogate" (Tier 1, strongest).** Durstewitz, Koppe, Thurm (2023, *Nature Reviews Neuroscience*) — the group behind shPLRNN — explicitly define trained RNNs on neural data as "formal surrogates" of the experimental system. That is the published language. They also explicitly disclaim that latent units are "identified biological sources," which *helps* us because we make the same limited claim.

2. **Simulation-based benchmark precedent (Tier 2).** The EEG source localization community uses simulated data with known dynamics as accepted ground-truth testbeds (EEGSourceSim; Neural Mass Modelling framework, bioRxiv 2026). shPLRNN fits this paradigm and is *stronger* because it learns its generative process from real EEG rather than assuming a parametric forward model. Schneider et al. (ICLR 2024) did exactly this — applied LFADS to EEG, validated on synthetic EEG with known latent dynamics. Direct precedent for our pipeline.

3. **Dipole-topography check (Tier 3, will be run).** Because `obs_model=linear`, the shPLRNN has a learned mixing matrix W (latents → channels). Each column of W is a scalp topography. We fit equivalent dipoles using MNE/DIPFIT. Residual variance < 15% is the ICA community's standard biological plausibility claim (Delorme et al. 2012, *PLOS ONE*, 10k+ citations). Cheap (~1 afternoon) and gives us the same biological correspondence language ICA papers use.

**What we CANNOT claim:** anatomically identified biological sources. No direct validation against known dipole generators. Reviewers who know the EEG field will accept this caveat — the inverse problem is irreducibly underdetermined even for ICA, and every simulation-based EEG benchmark in the literature has the same limitation. The reviewer attack ("no biological correspondence") applies equally to EEGSourceSim, neural mass model testbeds, and every ICA-on-EEG evaluation pipeline in existence.

**Key citations to have on hand:**
- Durstewitz, Koppe, Thurm (2023). Reconstructing computational system dynamics from neural data with recurrent neural networks. *Nature Reviews Neuroscience* 24, 693–710.
- Koppe, G. et al. (2019). Identifying nonlinear dynamical systems via generative recurrent neural networks with applications to fMRI. *PLOS Computational Biology* 15(8): e1007263.
- Schneider et al. (ICLR 2024). A latent variable modeling approach for cognitive EEG data.
- Delorme, A. et al. (2012). Independent EEG Sources Are Dipolar. *PLOS ONE*.
- Pandarinath et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders. *Nature Methods*.
- Zhu et al. (2023). Unsupervised representation learning of spontaneous MEG data with nonlinear ICA. *NeuroImage*.

The paper's logic:
1. Real-EEG identifiability claims have no oracle ⇒ unfalsifiable.
2. To test methods fairly, we need data with known sources.
3. The shPLRNN, trained on real EEG, generates realistic *and* ground-truth-defined synthetic data.
4. We run candidate methods (PCA, FastICA, TCL, iVAE) on this synthetic data; measure MCC against the known sources.
5. Methods fail at biophysically realistic SNR / mixing / coupling.
6. Therefore, claims about real EEG cannot be substantiated.
7. Constructive answer: synthetic benchmarks should be required; interventional data is the gold standard for the real-world version.

---

## 3. How the plan addresses the ICLR rejection

| Critique | Addressed by |
|---|---|
| AC: "what would a good solution be?" | §1 claim 2 + §1 claim 3 — concrete benchmark protocol + paradigm pivot. |
| AC: "what is missing from the proposed methods?" | §6 — controlled failure-axis sweeps quantify which assumption breaks first. |
| LjQ5: "no mechanism-level explanation" | §6 — analytic + empirical decomposition of failure modes. |
| LjQ5: "no actionable guidance / what next" | §1 claim 2 + §1 claim 3. |
| LjQ5: "translate results into broader lessons" | The position itself is the lesson; every section visibly serves it. |
| ACaF: "background on causality / TCL / nonlinear ICA" | §4 of the paper — primer subsection. |
| ACaF: "figures missing, equations broken" | LaTeX submission resolves this. Figure quality is a hard requirement; replicates the rendering issue from the blog must not recur. |
| ACaF: "if assumptions hold, does it work?" | §6 directly tests this on synthetic data where assumptions are made to hold by construction. |

---

## 4. Paper structure (9 pp)

```
1. Introduction                                  (1.5 pp) — bold position
2. Background                                    (1.5 pp) — TCL, iVAE,
                                                            interventional identifiability
3. Why scalp-level identifiability is            (1.0 pp) — the unfalsifiability argument
   untestable
4. The synthetic testbed: shPLRNN as generator   (1.0 pp) — what it does and does NOT do
5. Mechanism: where does identifiability         (2.0 pp) — failure-axis sweeps; the empirical core
   actually break?
6. Real-EEG re-analysis (illustrative)           (0.5 pp) — TCL ≯ PCA on real EEG, framed as
                                                            "we cannot tell if this is method
                                                            failure or unfalsifiable evaluation —
                                                            exactly our point"
7. Constructive paths: synthetic benchmarks,     (0.5 pp) — community standards + BCI/iEEG/TMS
   BCI, intracranial, TMS
8. Alternative views                             (0.5 pp) — required by NeurIPS, in main text
9. Conclusion                                    (0.5 pp)
References                                       — no page limit
Appendix                                          — full pipeline, ablations, intervention-test
                                                    sketch as future work
```

---

## 5. Background (paper §2)

Concise (~1.5 pp). Must cover:
- Linear ICA → time-contrastive learning (TCL, Hyvärinen & Morioka 2016) → auxiliary-variable identifiability (iVAE, Khemakhem 2020) → interventional / nonparametric identifiability (Kügelgen 2023).
- For each: state the identifiability theorem and the assumption it rests on.
- 1-row-per-method table: **assumption** vs. **whether real EEG plausibly satisfies it**. (Spoiler: no row is "yes.")

This addresses ACaF directly.

---

## 6. The empirical core (paper §4 + §5)

### 6.1 Pipeline

1. **Train multi-channel shPLRNN.** `latent_size = 4`, `obs_size = 22` (matched to BCI IV-2a's 22 EEG channels), `obs_model = linear`. The shPLRNN's output channels are the "synthetic EEG"; its latents are the "synthetic sources." No formal generator validation step (skipped to save time); the model's training loss is sufficient. The 92.6% paper-reproduction target is **not** a goal here.

2. **Add nuisances.** On top of the generator's output, add (a) configurable 1/f pink-noise background, (b) Gaussian sensor noise, both with controllable SNR.

3. **Run methods on the noisy generated data.** PCA, FastICA, TCL, iVAE.

4. **Evaluate.** Mean correlation coefficient (MCC) between recovered components and the known generator latents, with optimal permutation matching. (This is the standard metric in the nonlinear-ICA literature.)

### 6.2 Failure-axis sweeps (paper §5 — the empirical core)

| Failure axis | What we vary | What we predict |
|---|---|---|
| **1/f background amplitude** | SNR ∈ {high, medium, low, EEG-like} | All methods degrade monotonically; TCL fastest |
| **Mixing nonlinearity** | linear → mild nonlinear → strong nonlinear projection from latent to channels | TCL collapses past mild nonlinearity |
| **Source coupling strength** | independent → weakly coupled → strongly coupled latents | TCL/iVAE break when independence violated |
| **Nonstationarity informativeness** | stationary ↔ source-aligned NS ↔ confounder-aligned NS | TCL works only when NS is source-aligned |
| **Sources vs sensors** | M < D, M = D, M > D | All methods fail when M > D |

Methods: PCA, FastICA, TCL, iVAE. Each (axis × config × method × seed) cell → MCC. Plot MCC vs. axis variable per method.

This is the **mechanism analysis** asked for by AC + LjQ5.

### 6.3 Intervention sketch (appendix only)

Brief subsection in the appendix describing what an intervention-based test of identifiability would look like (clamp source `k` in the generator, check whether the recovered `s_k` shifts accordingly), framed as future work / a community challenge. Does not need to be implemented.

---

## 7. Real-EEG re-analysis (paper §6)

Reuse the existing TCL-vs-PCA figure(s) from the original blog post (regression coefficient matrices, c=4/10/25 patient panels). No new experiments. The figures are already in `/Users/mahta/Projects/iclr_2026/submission/ICLR 2026/`. Caption is reframed: *"We cannot distinguish whether this is method failure or unfalsifiable evaluation — that is the point of this paper."*

Must render correctly in the LaTeX submission (the ICLR blog reviewers reported figures didn't display — that was a markdown/blog-platform issue, won't recur in LaTeX).

---

## 8. Alternative views (paper §8, NeurIPS-required, main text)

Address (not strawmen):

1. **"Synthetic benchmarks oversimplify biology."** Counter: any benchmark is a falsifiability tool under specified assumptions; the alternative is no falsifiability at all. We make the synthetic *easier* than real EEG (fewer subjects, no participant motion artefacts, controlled noise). If methods fail on the easy version, they cannot succeed on the hard one.
2. **"Interventional data already exists in BCI / TMS / iEEG, so the problem is solved."** Counter: it exists, but identifiability papers don't currently use it as the publication standard. This is a community-norms argument.
3. **"Identifiability theorems are theoretical, not empirical, so they don't need empirical validation."** Counter: every paper running an experiment makes an empirical claim that the theoretical assumptions hold on the test data. Current practice elides this, especially in neuro-AI.
4. **"shPLRNN is itself a learned model, not biophysics; your synthetic isn't real ground truth."** Counter: (a) The shPLRNN group itself calls trained RNNs "formal surrogates" of the experimental system (Durstewitz et al., NRN 2023) — that is the operational definition of ground truth we are using. (b) Every simulation-based EEG benchmark in the source-localization literature uses modelled rather than directly observed sources as ground truth (EEGSourceSim; Neural Mass Modelling framework); if those are acceptable, so is a data-driven surrogate trained on real EEG. (c) We can run the ICA community's standard biological plausibility check (dipole-fit the obs_model columns; RV < 15% = biologically plausible, Delorme et al. 2012) as a post-hoc sanity check. The reviewer attack applies to the entire EEG simulation benchmark literature, not uniquely to us.

---

## 9. Open questions for Mahta (short list)

1. **Multi-channel EEG dataset:** default = BCI Competition IV-2a. OK or do you want TUH EEG / CHB-MIT / something else? The choice mostly determines what the trained shPLRNN has *learned*; for the source-recovery benchmark itself, any reasonable multi-channel dataset works.
2. **Confirm M = 4** as the default latent dim, with M = 8 as a fallback if richer dynamics are needed.
3. **Confirm reframing** of the shPLRNN as "testbed, not solution" (per §2). This is critical for the paper's coherence.
4. **Methods list:** locked at PCA, FastICA, TCL, iVAE. Add anything? (CDIB? SimCLR-style contrastive baselines for time series? A "supervised upper bound" using true source labels?)

---

## 10. Workplan (max rigor, not max compute)

Mostly sequential; some parallel possible.

1. **Lock §9 decisions** (dataset, M, methods list).
2. **Multi-channel dataset prep.** Download BCI IV-2a (or chosen alternative), pre-process, split into segments.
3. **Train multi-channel shPLRNN.** Single training run, possibly a few seeds.
4. **Dipole-fit the obs_model columns** (~1 afternoon, MNE/DIPFIT, see §2 Tier 3).
5. **Build mixing-and-noise pipeline.** Configurable 1/f, sensor noise; deterministic so ground truth is preserved.
6. **Method implementations.** PCA / FastICA from sklearn; TCL from existing repo or Hyvärinen reference; iVAE from Khemakhem reference.
7. **Failure-axis grid (§6.2).** ~5 axes × ~5 config points × 4 methods × 5 seeds = 500 runs. CPU-tractable.
8. **LaTeX skeleton + drafting.** Order: §2 (background) → §3 (unfalsifiability) → §4 (testbed framing) → §5 (sweeps) → §1 + §9 → §6 + §7 + §8. Reuse existing blog figures for paper §6 (no re-rendering needed).
9. **Internal review** with Pooneh and Guillaume.
10. **Submit.**

Compute footprint: shPLRNN training is cheap (single model, M=4, ~1 GPU-day). iVAE on 500 configs is the only meaningful GPU cost (~30–50 GPU-hours total on L40S/A100). Everything else runs on CPU.

---

## 11. Internal expected outcomes (planning use, not for the paper)

These are our predictions before running anything, kept here so we plan the writing around what we expect to find. Not a public pre-registration.

| Axis | Expected curve | Mechanism |
|---|---|---|
| **1/f SNR (clean → EEG-like)** | TCL drops below PCA at low SNR | TCL discriminator latches onto the largest NS = 1/f drift, not source dynamics |
| **Mixing nonlinearity (linear → strong)** | Linear methods (PCA/FastICA) collapse first; TCL/iVAE break later | Linear methods cannot represent nonlinear inverse |
| **Source coupling (indep → strong)** | TCL drops sharply | Independence assumption violated |
| **Nonstationarity (stationary → source-aligned → confounder-aligned)** | TCL: chance → high → ~chance | TCL recovers whichever NS is dominant |
| **Sources vs sensors (M ≤ D → M > D)** | Sharp cliff at M = D for all methods | Underdetermined inverse problem; not method-specific |

**Headline story we expect:** all methods work in easy regimes; under simultaneous EEG-realistic conditions (low SNR + nonlinear mixing + coupled sources + confounder-dominated NS), all methods drop to near-chance, with TCL specifically dropping below PCA. This reproduces the original blog finding under controlled conditions with mechanistic explanation.

**If predictions are wrong:** the position survives — it depends on real-EEG unfalsifiability, not on TCL failing. Empirical narrative would pivot to "TCL works on synthetic; we still cannot verify on real EEG."

---

## 12. Risks

1. **shPLRNN at M=4 doesn't produce convincingly realistic dynamics.** Mitigation: bump to M=8 or fall back to a simpler generator (a coupled-oscillator system with explicit nonstationarity).
2. **Reviewers say "this is just a research paper, not a position paper."** Every section visibly serves the position; benchmark protocol (§6.1) is itself a community contribution.
3. **Reviewers complain about generator circularity** (shPLRNN fitted on real EEG, then used as a "ground truth" for testing). Pre-empted in Alternative View #4 (§8.4) and §2 framing. We are not claiming biophysical fidelity — we are constructing a controlled testbed.
4. **Reviewers ask "but did you try interventions?"** Pre-empted by §6.3 (sketch in appendix) + the position's "synthetic-with-known-ground-truth **or** interventional" wording. The empirical contribution is the first half; the second half is constructive prescription.
5. **Negative-results bias.** Position is constructive (benchmark + paradigm pivot), not destructive. Defensive framing in writing is essential.
6. **Scope creep.** Hard caps: 4 methods, 5 failure axes, 1 dataset. Anything beyond → appendix or future work.
