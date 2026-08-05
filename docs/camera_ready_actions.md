# Camera-Ready Action List — Submission 951

Consolidated from all 18 Claude Code rebuttal-drafting sessions (HierarchicalDSR project), 2026-08-05. Method: 9 parallel read-agents, one per session/cluster, each paginating the full transcript and extracting camera-ready commitments + open decisions.

**Scope note:** covers only the 18 audited sessions. Other HierarchicalDSR sessions exist outside that set (e.g. "item#1", "item#2-3", "item#6", "item#7", "Review todo rebuttal preparations") — not swept here.

---

## 🔴 CRITICAL #1 — Figure 2 caption: the "drops to near-chance" claim is false

This is the single most important camera-ready fix. Current caption (`writing/sections/05_failure_axes.tex:23`):

> "Every method drops from near-perfect to near-chance on at least one axis, at conditions plausible for real EEG."

It is unsupported **two independent ways**, both visible in the figure:

1. **Not every method reaches chance.** Per the paper's own Table `tab:full`:
   - **TCL** — the *only* method that genuinely spans near-perfect → near-chance: 0.944 (clean) → **0.199** (0dB 1/f) / **0.052** (stationary NS).
   - **iVAE** — reaches near-chance on source-coupling (**0.04**) and M→D (**0.09**), but its ceiling is **0.769**, so it never *starts* near-perfect. Fails the "from near-perfect" half.
   - **FastICA** — floor **0.615** (tanh mixing). Never approaches the 0.25 chance line. Fails the claim entirely.
   - **PCA** — range **0.417 → 0.927**. Floor 0.42 never nears chance; satisfies neither extreme.
   - So the blanket "every method" holds for **TCL only** (and half-holds for iVAE). The two linear baselines never drop to chance.

2. **The chance line itself is a fixed guess, not a null.** The dashed line is a hard-coded MCC = 0.25, never defined in text (T2SC Q1, still unanswered). Against a fixed 0.25, several quoted "failures" — e.g. confounder-aligned NS at **0.26–0.29** — are *barely above* chance, not clean drops to it. A per-condition **shuffled null** would set an honest, axis-specific reference and is needed before any "near-chance" wording is defensible.

**Why this doesn't hurt the thesis (keep this framing):** the position never needed all four methods to crater. It needs each method that *claims* nonlinear identifiability (TCL, iVAE) to have an EEG-plausible regime where its guarantee fails and can't be checked on real data — and those two are exactly the ones that collapse. The linear baselines (PCA, FastICA) make no such claim and staying mediocre-but-stable is consistent with the argument, not a counterexample.

**The fix (do these together):**
- Replace the blanket sentence with per-method, per-axis wording using the numbers above. Anchor the "overclaim" correction on FastICA/iVAE/PCA (**not** TCL — the old draft justified the fix via mRZt's TCL-based reasoning, which is now stale since TCL is no longer broken).
- Compute a per-condition shuffled null (~20 lines reusing the existing `mcc()`), replace the fixed 0.25 line, and state its construction in the caption + appendix.
- **Rebuild the figure from `failure_axis_final`** — the current render (`figures_corrected/`) is built from the corrupted `failure_axis_corrected` dataset (PCA ≡ FastICA on every axis, impossible; see Data-Integrity below).
- Disclose that Fig 2 applies a linear-invariant metric to TCL only and strict MCC to the other three.
- **Figure-number check:** the float file is `fig1_failure_axes.pdf` but sessions/§5 refer to it as "Figure 2" — confirm the label/number is consistent between the caption source and the body text.

---

## ⚠ Timing check

Per your own `review_todo` dashboard, **Phase 2 (author/reviewer/AC discussion) closed Aug 3, 2026.** Today is Aug 5 — two days into **Phase 3 (reviewer/AC only)**. Items below left as "should we post/run X for the reviewer" are likely moot as public rebuttal moves, unless a private author→AC channel is still open (worth a manual OpenReview check). Everything is organized manuscript-action-first; reviewer-facing actions are called out separately where they still might matter (e.g. correcting the posted number that's wrong).

## Reviewer snapshot (latest state across sessions)

| Reviewer | Score | Confidence | Status |
|---|---|---|---|
| **11Xm** | 3 — Reject | 5 | Core novelty objection **unanswered anywhere in 18 sessions**. Highest-priority open thread after Fig 2. |
| **xW1o** | 4 — Borderline reject | 4 | Faithful TCL rerun posted (segments 8→512, real config); prevalence denominator still owed; unconvinced as of last activity. |
| **T2SC** | 6 (dropped from 7) | 3 | Score drop cites "alignment with other reviewers," not a new technical reason — flagged for a possible private AC note. Small items still owed (title, chance-level, latent-source definition). |
| **mRZt** | 6 — Weak accept | 3 | **Locked/satisfied**, explicitly conditional on the camera-ready manuscript actually containing the promised experiments + corrected Fig. 2 caption. High stakes even with no more back-and-forth. |

---

## PART A — Decisions only you can make

Recur across multiple independent sessions, never resolved on-record.

1. **Morlet/alpha-band TCL — DECIDED: will not run** (Mahta, 2026-08-05). Ship the mechanistic argument only (synthetic sources are broadband/non-oscillatory in code, so there is no alpha rhythm on the testbed to filter). **Action:** make sure no drafted/shipped reviewer text still *commits* to a Section 6 Morlet run — one "Identifiability claims" fork drafted xW1o-facing text that promised exactly that ("it is now a testable promise"). That promise must be removed from any draft before camera-ready.

2. **Coupled vs. decoupled TCL segmentation — the record contradicts itself.** The "xW1o (fork)" session (Jul 31) chose a *decoupled* rerun; the "coupled run" session (Aug 1) says you explicitly rejected decoupled and that **a sibling session overwrote `response_xW1o.md`'s C1 with the decoupled numbers you'd rejected**; the latest xW1o session (Aug 2) treats decoupling as accepted and flags no conflict. Looks like forked sessions racing on one file. **Check the current on-disk state of `writing/Rebuttal/response_xW1o.md` directly** — no transcript alone can tell you which numbers are in it now.

3. **11Xm's core novelty objection has no drafted answer anywhere.** "What's new vs. a well-known limitation?" — your only Reject, at confidence 5. Candidate content exists (falsifiability framing + the five-axis benchmark + "iVAE's success is generator-specific, not the failures" + the three-claim separation) but was never turned into a response.

4. **Camera-ready title — two candidates drafted, neither confirmed:**
   - *"Do Not Claim Identifiability Without Ground Truth or Intervention: A Position on Causal Representation Learning from Observational Data"*
   - *"...from Observational Scalp EEG"* (breadth narrowed from "Data" without visible sign-off).

5. **Three open calls from the meta-reviewer/AC session:** confirm the "biological sources" (not "independent sources") wording revert; insert-or-not the drafted closing synthesis paragraph; post-or-not a confidential AC note about the 11Xm/T2SC score-text mismatches.

---

## PART B — Data-integrity flags

Currently wrong, inconsistent, or unverified — some touch numbers already posted publicly on OpenReview.

- **Corrupted intermediate dataset:** `failure_axis_corrected` scores PCA and FastICA *identically* on every axis (impossible). `figures_corrected/` (current Fig. 2 render) was built from it. Rebuild from `failure_axis_final`. **[gates the Fig 2 fix above]**
- **A number already posted to OpenReview is wrong.** xW1o's C2 claims TCL agrees "within 0.01" between learned-W and random-W mixing — the actual max discrepancy is **0.206** (verified against the `.npy`). Public. Needs a corrective comment if Phase 3 still allows author comments; camera-ready must use the true number regardless.
- **Wrong bib entry:** `references.bib` key `mutnuri2023scalp` has the right title but attributes it to an unrelated personal-site PDF instead of Krakovská, Rošťáková, Chvosteková, Maslíková (2023), IEEE Proc. 14th Int. Conf. on Measurement, pp. 92–95. Used once, `writing/sections/06_real_eeg.tex:24`. Rename key + fix metadata; check the Slovak co-author diacritics before compiling.
- **Possible OpenReview visibility bug:** all four posted rebuttals show reader scope "Program Chairs, Authors" only. Confirm manually that reviewers/AC could see them — if not, it may explain reviewer silence.
- **Cross-response numbers never reconciled to one logged source:**
  - iVAE learned-generator MCC: **0.80** (11Xm draft) vs **0.71** (T2SC/mRZt) vs **0.73** (submitted Fig. 2). Same cell, likely seed variance — pick one.
  - TCL clean-condition MCC: **0.92** (mRZt) vs **0.94** (T2SC/xW1o) vs **0.65/0.87/0.52** (generator-driven teacher-forced — different construction, needs a labeling clause not a fix).
  - FastICA worst-case quoted three ways across the paper's own table vs. body: **0.615** (Table `tab:full`) vs "≈0.98" vs "≈0.71".
- **The abstract contradicts your own positive control.** It still says the position covers "exactly the axes on which current identifiability methods break" — TCL's faithful-protocol positive control now recovers 0.83–0.98 on 3 of 5 axes (clean, tanh, coupling). Narrow it.
- **iVAE's collapse to ~0.10 under random linear mixing is theoretically odd** — Khemakhem et al. (2020) predicts iVAE should identify under linear mixing. Possibly undertraining, not a genuine limitation; nobody checked.
- **The "insurance run" gap:** mRZt's own M≫D sweep tests FastICA/PCA/iVAE but omits **TCL**, the flagship method, from the flagship regime. TCL has never been run under leadfield mixing / M≫D with the *faithful* (fixed) implementation — only pre-fix numbers (0.04–0.15) exist there, and those must not be cited against the segment-sweep fix.

---

## PART C — Manuscript edits: agreed and drafted, just need to land

Content/numbers settled and verified — the "go write it into the .tex" bucket, modulo inline flags.

1. **Three-generator appendix** (learned shPLRNN / random linear / MNE leadfield), all five stress axes. Conclusion: linear methods generator-invariant; **iVAE's apparent success — not the failures — is what's generator-specific.** *[11Xm Q3, T2SC C2, mRZt C2]* — verified on disk. ⚠ Don't cite TCL numbers from these cross-generator runs against the segment sweep — they used pre-fix scoring.
2. **TCL segment-count sweep** (8→512, faithful protocol: LayerNorm+dropout, final linear ICA, T=128,000) as a table + figure in the appendix. *[xW1o C1, T2SC C1]* — verified on Mila; write-up + figure pending (.npy files Mila-only, see Part D).
   - **Manuscript-story change, not just a number fix:** under the faithful protocol, tanh (0.95), source-coupling (0.88), and M=22 (~0.80–0.81) now recover fine. **Stop calling coupling/nonlinearity/M=D "TCL failures"** — only strong 1/f (0dB) and confounder-aligned NS remain valid failure claims.
   - Mild 1/f (30dB) is implementation-sensitive (0.72 old vs 0.24–0.30 faithful) — don't headline that specific number.
3. **M≫D / source-count sweep** via the leadfield generator: FastICA 0.91→0.065, PCA collapses similarly; linear methods generator-invariant. *[mRZt C2]* — delivered, mRZt satisfied. (TCL's version is the "insurance run" gap in Part B.)
4. **Latent-dimension wording fix:** separate §4's native M=4 construction from §5's sweep-to-22 (generator not retrained per source count); "approaches or exceeds" → "approaches." Files `04_testbed.tex:11,17,18,24`, `05_failure_axes.tex:17,23`. *[Meta, mRZt]* — delivered.
5. **Fig. 2 caption rewrite** — see CRITICAL #1 at the top; this is the same item, elevated.
6. **Generator-realism wording:** "dynamics resemble those of cortex" → "output reproduces the statistical structure of real EEG (spectral profile, nonstationarity, channel covariance)... without any claim about biophysical source identity." File `04_testbed.tex:11`. *[Meta, T2SC]* — delivered.
7. **Section 6 addition:** standard EEG cleaning doesn't remove the 1/f confound (broadband, survives RANSAC/notch/ICA); segment-sweep shows the failure deepens with finer segmentation. *[mRZt]* — delivered. Gap: whether the real-EEG analysis actually used a cleaning pipeline was never confirmed — still gates exact wording.
8. **New citation:** Krakovská et al. (2023). *[mRZt]* — resolved in principle, but bib entry is wrong (Part B) and citation never independently verified.
9. **FastICA discussion point:** using FastICA isn't a refutation of the position, it's an instance of it — no real-EEG ground truth exists to validate any method. *[T2SC]* — delivered.
10. **Three-claim separation** (mathematical identifiability / assumptions-hold-on-real-EEG / biological-source correspondence) stated in Section 3, every strong claim scoped to the third. *[Meta, 11Xm]* — drafted, wording not finalized; tied to the unresolved 11Xm novelty answer (Part A #3).
11. **Foreground the instrumentalist alternative** at the start of Section 8 — current §8 ¶2 ("interventional data already exists") reads as a straw-man 11Xm likely reacted to; fix regardless. *[11Xm]*
12. **Section 6 / Figure 3 presentation cleanup.** *[T2SC]* — mostly resolved via caption-only rewrite (your proposed MDD/ODER metrics table was dropped — it contradicted §6's own PCA≈TCL claim). One small item: soften "PCA closer to a permutation" → "tendency" at §6 L22–23.
13. **Operational definition of "latent source"** at first use, Section 1. *[T2SC]* — open; one session proposed shrinking to a one-line pointer to existing §1/§6 text, unconfirmed.
14. **Camera-ready title stating the position.** *[T2SC]* — see Part A #4.

---

## PART D — Experiments / build work still needed

- **Per-condition shuffled null for Fig. 2** (CRITICAL #1). Highest-risk gap, most-corroborated across sessions. Do first.
- **§2 classification table** grounding the prevalence claim (papers by identifiability-claim type / real-data biological claim / input space / validation). The underlying systematic review (5 sub-agents, citation-graph search) found 0 direct offenders and 4 partial/self-hedged among modern scalp-EEG papers — but **you haven't verified that review yourself yet**. Verify before it anchors a reviewer-facing claim.
- **TCL under leadfield / M≫D** with the faithful implementation — the "insurance run," never executed (Part B).
- **Cross-generator TCL row:** currently shows pre-fix numbers (0.06–0.08) contradicting the faithful positive control (0.94–0.97). Either rescore with the faithful pipeline or drop TCL from that comparison (restrict to PCA/FastICA/iVAE).
- **Pull Mila-only result files locally** before building final figures — `tcl_segment_sweep_faithful/*.npy`, `generator_driven_tf/*`, `forward_model`/`xgen` outputs aren't in the repo, so the appendix isn't reproducible from what's checked in.
- **Rebuild Fig. 2** from `failure_axis_final` (Part B / CRITICAL #1).
- Optional: swap the iVAE citation for Turco & Houghton (2024) to avoid iVAE playing double duty (failure case in C2 + prevalence exemplar in §2).

## PART E — Unanswered reviewer asks (beyond Part A)

- **xW1o:** full prevalence denominator answer, plus independent verification of the Krakovská citation before it ships anywhere reviewer-facing.
- **T2SC:** chance-level definition (= the shuffled-null item, CRITICAL #1), title (Part A #4), latent-source definition (Part C #13).
- **11Xm:** beyond the novelty question (Part A #3), the "straw-man alternatives" charge and the significance/presentation weaknesses are unaddressed in every session.
- Whether the consolidated AC-facing synthesis (drafted in the meta-reviewer session) ever gets posted — likely moot in Phase 3; check venue rules.

## PART F — Housekeeping

- **Uncommitted code:** `simulate_generator.py` and `generator_driven_sweep.py` sit on the Mila `tcl` branch with local `.bak` backups, not committed. Commit before camera-ready or the appendix numbers aren't reproducible from repo history.
- **Canonical-file drift:** the fuller `review_todo` document (rules, verbatim reviews, SWEEP/GEN/CLARIFY/DIM/DRAFTS panels) survives only in a published artifact; `docs/review_todo.html` is just the action-table mirror. Decide whether to reconstruct the full version into the repo.
- **Typos already live on OpenReview**, fix for camera-ready too: "scam-" for scalp, "representive," "our paper argue," empty template placeholders, reviewer ID misspelled "MRzt" inside mRZt's own response.
- AI-use disclosure wording ("paraphrasing, presenting results, finding the right words") sits close to the position-track's copy-edit-only line. Submission metadata is locked — awareness-only, be ready if the AC raises it.
