# Camera-Ready Execution Plan — Submission 951

Companion to [`camera_ready_actions.md`](camera_ready_actions.md). That file is the *what* (consolidated action list from 18 rebuttal sessions); this is the *how* (sequenced plan). Written 2026-08-05.

## Grounding (verified against repo/data, not just session transcripts)

- Paper `writing/`, authoritative results, and new scripts all live under the main checkout `/Users/mahta/Projects/HierarchicalDSR/` on branch `tcl`.
- **`failure_axis_corrected` is genuinely corrupt** — confirmed by loading the `.npy`: PCA and FastICA arrays are byte-identical on 4 of 5 axes (`1f_snr`, `nonlin`, `coupling`, `m_vs_d`), which is impossible for two different methods. PCA is inflated to ~0.97+ there.
- **`failure_axis_final` is clean** — PCA/FastICA properly distinct; its per-method numbers match the audit's Table `tab:full` exactly (no fabrication).

---

## 🚨 Phase 0 — Save the work FIRST (data-loss risk) — Part F

The camera-ready paper and all rebuttal evidence are **untracked in git**:

```
tcl branch:  writing/                              → 0 files tracked (paper is in git nowhere)
             results/failure_axis_final/           → untracked (??)
             results/tcl_segment_sweep_faithful/,
             generator_driven/, generator_validation/ → untracked
             simulate_generator.py, generator_driven_sweep.py,
             tcl_segment_sweep_faithful.py, validate_generator.py,
             tclfaith.sbatch                        → untracked
             failure_axis_sweep.py, plot_failure_sweep.py → modified, uncommitted
```

One `git clean -fdx`, `rm`, or disk fault wipes the paper AND the evidence. Fix before any editing.

1. Commit paper + authoritative artifacts to `tcl` (or a `camera-ready` branch off it):
   ```bash
   cd /Users/mahta/Projects/HierarchicalDSR
   git add writing/ results/failure_axis_final/ results/tcl_segment_sweep_faithful/ \
           results/generator_driven/ results/generator_validation/ \
           scripts/*.py scripts/*.sh *.sbatch
   git status   # review BEFORE commit — jobs/, tmp/, .DS_Store, the -4.pdf stay out
   ```
2. Quarantine the corrupt set so nothing replots from it:
   `git rm -r --cached results/failure_axis_corrected results/figures_corrected`, add both to `.gitignore`.
3. Confirm Mila-only `.npy` (faithful sweep, generator_driven_tf) are in the add above, not still on the cluster.
4. `.gitignore`: `jobs/ tmp/ .DS_Store .agents/`.

Effort: ~20 min. Blocks everything downstream.

---

## Phase 1 — Figure 2 (CRITICAL #1) — Part C5, Part B abstract, T2SC chance-level

Highest leverage: mRZt's weak-accept is *conditional* on the corrected caption shipping.

### New finding — iVAE looks undertrained, not "failed"

From `failure_axis_final`, iVAE on the source-count axis:
```
M=2 → 0.094   M=4 → 0.050   M=8 → 0.090   M=16 → 0.094   M=22 → 0.109
```
iVAE ≈ 0.1 even at M=2 (easiest cell, 2 sources / 22 channels, near-linear). Khemakhem et al. 2020 says iVAE *should* identify under linear mixing. iVAE ≈ chance on the easy cell reads as a broken/undertrained baseline, not a guarantee failing. iVAE is one of only two methods carrying the "drops to chance" story — **if it's undertrained, the near-chance narrative rests on TCL alone.** Diagnose iVAE training (epochs, linear-cell sanity check) BEFORE any iVAE failure claim ships. High priority.

### Steps

1. Point the plot script at `failure_axis_final`; remove all references to the corrupt set.
2. Add per-condition **shuffled null** (~20 lines reusing `mcc()`), replace the fixed 0.25 line. Simultaneously answers T2SC Q1 (chance level undefined).
3. Resolve the NS-stationary wrinkle: TCL's worst cell is **stationary** NS = 0.052 — the *easiest* condition. A reviewer will ask why it fails hardest on the easy case. Real answer: TCL keys on variance-nonstationarity; with none present it recovers `|s|`, not signed `s`. Either fix the NS base-config (memory flagged this) or add an honest caption note. Decide.
4. Diagnose iVAE (above); re-run the linear cell if undertrained.
5. Rebuild the figure. Fix the label mismatch: float file is `fig1_failure_axes.pdf`, text calls it "Figure 2" — reconcile.
6. Rewrite the caption per-method using the verified numbers:

   | method | best | worst (axis) | reaches chance? |
   |---|---|---|---|
   | FastICA | 0.999 | **0.615** (tanh) | no — floor 0.62 |
   | PCA | 0.927 (M=2) | **0.417** (M=22) | no — floor 0.42 |
   | TCL | 0.944 | **0.199** (1/f 0dB), **0.052** (stationary NS) | yes |
   | iVAE | 0.769 (never near-perfect) | **0.043** (coupling), **0.050** (M=4) | yes — but see undertrain flag |

   Anchor the overclaim correction on FastICA/PCA (never near chance) + iVAE (never near-perfect). Only TCL truly spans the claimed near-perfect→near-chance range.

   Per-axis honest story (mean MCC at the stress endpoint):
   - **1/f 0dB:** TCL 0.199, iVAE 0.351, PCA 0.666, FastICA 0.889 — TCL fails, linear robust.
   - **tanh:** iVAE 0.170, FastICA 0.615, PCA 0.683, TCL 0.825 — iVAE fails, FastICA dips.
   - **coupling 0.9:** iVAE 0.043, PCA 0.744, TCL 0.854, FastICA 0.980 — iVAE-only failure.
   - **stationary NS:** TCL 0.052, iVAE 0.577, FastICA 0.707, PCA 0.656 — TCL-only (the "easy" cell; needs explaining).
   - **M=22:** iVAE 0.109, PCA 0.417, TCL 0.444, FastICA 0.914 — iVAE fails, FastICA robust.

7. Fix the abstract sentence "exactly the axes on which current identifiability methods break" to match (Part B abstract contradiction) — same edit session.

Effort: 1–2 days (iVAE diagnosis is the unknown).

---

## Phase 2 — Data-integrity + one-number-one-source — Part B

- **Posted-number error (public):** xW1o C2 "TCL agrees within 0.01" (learned vs random mixing) — real max diff **0.206**. Fix the camera-ready number; corrective OpenReview comment only if Phase 3 still permits author comments (**manual check**).
- **bib:** `mutnuri2023scalp` → `krakovska2023scalp`; fix authors (Krakovská, Rošťáková, Chvosteková, Maslíková 2023, IEEE Proc. 14th Int. Conf. on Measurement, pp. 92–95); repoint `06_real_eeg.tex:24`; eyeball Slovak diacritics.
- **Reconcile floats to one logged run each:** iVAE learned-gen (0.71/0.73/0.80 → pick one), TCL clean (0.92/0.94), FastICA worst = **0.615** authoritative. grep `writing/` + response files.
- **OpenReview visibility bug** — all 4 rebuttals scoped "Program Chairs, Authors" only (**manual check** on OpenReview).

Effort: ~half day + two manual checks.

---

## Phase 3 — Settled `.tex` edits, batch — Part C 4,6,7,9,12,13

Low-risk, no new decisions, all in `writing/sections/`. Independent of Phase 1 — parallelizable.

- `04_testbed.tex`: "resemble those of cortex" → statistical-structure wording (C6); separate §4 M=4 from §5 sweep, "approaches or exceeds" → "approaches" (mRZt C1). Lines 11,17,18,24.
- `06_real_eeg.tex`: 1/f-survives-cleaning paragraph (mRZt C4) + bib repoint.
- soften "PCA closer to a permutation" → "tendency".
- FastICA-is-an-instance discussion point (T2SC).
- one-line "latent source" definition, §1 (T2SC).

Effort: ~half day.

---

## Phase 4 — Remaining experiments — Part D

- Shuffled null (folded into Phase 1).
- **TCL faithful leadfield / M≫D "insurance run"** — the gap mRZt/11Xm can attack (mRZt's own M≫D sweep omitted TCL). Mila job. Do if time.
- Cross-generator TCL row shows pre-fix 0.06–0.08 vs faithful 0.94 positive control — **rescore faithful or drop TCL from that table.**
- §2 prevalence table — **verify the systematic review yourself first** (unverified, prior fabrication-scar) before it anchors a claim.

---

## Phase 5 — Decisions + reviewer-facing — Part A, Part E

Mostly author calls / gated on whether the Phase-3 window is open:

- **A#1 Morlet — DONE, will not run.** Only action: scrub any draft that still *promises* a Section 6 Morlet run (one xW1o fork did). grep response files.
- **A#2 coupled/decoupled race** — check `response_xW1o.md` on disk: which numbers are in C1? Can be settled immediately.
- **A#3 11Xm novelty** — no answer exists anywhere. Draft: falsifiability + benchmark + "iVAE success is generator-specific" + three-claim split, into §3; post comment if window open. Biggest score-risk after Fig 2.
- **A#4 title** — pick one candidate.
- **A#5 meta** — biological-sources revert (confirm), closing synthesis (in/out), confidential AC note (send/not).

---

## Recommended order

Phase 0 now (data-loss). Then Phase 1 ∥ Phase 3 (independent). Phase 2 alongside. Phases 4/5 after, gated on the discussion window + author decisions.

## Reversible, ready to start with no new decisions
- Phase 0 commit (after eyeballing the `git add` set).
- Read `response_xW1o.md` → settle the coupled/decoupled race (A#2).
- Diagnose the iVAE training config (the Phase-1 blocker).
