# Claim ledger

Every substantive claim made in this line of work, and what happened to it. Kept because the
retraction rate is high enough that the *pattern* is itself a finding, and because a reader of the
papers should be able to see which conclusions survived and which did not.

Format: claim → fate → what caught it. Times are commit times in one working session.

## The round trip

The single most important entry is not a claim but a shape. At **15:52** the conclusion was
*"the wall is escape time, not representation"* (optimisation). By **18:23** it was *"the limit is
arity — one matched filter, one bit"* (representation). At **07:05** it was *"it is expressivity,
not optimisation, and nh=2 is the ceiling"* (representation again, sharper). By the **nh sweep** it
was back to optimisation. A full circle, with confident write-ups at each station.

## Ledger

| # | claim | fate | caught by |
|---|---|---|---|
| 1 | Symmetry/bifurcation explains the plateau | **falsified** — gradient points at binding, cos = 0.75 | own measurement |
| 2 | Failure is a rank-one readout, not capacity | stands | — |
| 3 | Muon "never escapes" | **retracted** — LR swept upward from its optimum; at 3e-4 it escapes at step 1000 | own re-sweep |
| 4 | Muon takes a *different route* | **falsified** — same marginal-first route, ~17× rate | own fine-resolution data |
| 5 | The wall is escape time, not representation | superseded, then **restored** by the nh sweep | — |
| 6 | "One matched filter = one bit" (Addendum III step 5) | **retracted** — HRR binds 8 pairs in one d=64 vector at 0.892 | Fable |
| 7 | carve+bind gives +63% | **retracted** — dissolved across 3 seeds | user's screenshot prompting a seed check |
| 8 | "top1 ≈ 1/npairs for any npairs" | **falsified at n=8** | own run |
| 9 | Krohn–Rhodes: the gate cap removes the only flip-flop | **falsified** — clamp never fires, g peaks at 0.816 | own control |
| 10 | A content router is a third option | **killed pre-commit** — birthday collisions cap it; 0.600 at n=8 | advisor + own `hashed` arm |
| 11 | "The transport destroys the read" | **retracted same day** — own table showed 0.230 vs 0.220 | advisor |
| 12 | Structural arm shows every repair fails | **withdrawn** — binding and transport shared one axis array, so the read never touched the binding | advisor |
| 13 | One-shot reading hides the memory's true capacity; the literature's curves are a readout artifact | **retracted** — one-shot has the *higher* threshold at 0.5; also straw-manned `arora2023zoology`, already in our own refs | advisor |
| 14 | The memory cannot verify its own conclusions | **reversed** — margin gives AUC 0.96–0.98; I used a difference of means on the wrong signal | own follow-up |
| 15 | Model sits at tr/dh = 0.750, in the working band | **corrected** — that was the d=128 test model; the checkpoint is dh=93, 0.957 | own measurement |
| 16 | Retention is 11–17 / 86 / 132 tokens | **corrected ~5×** — bias-only reading; on real text it is 2–3 to 94 | own measurement |
| 17 | Effective context ≈ 8 characters | **caught before reporting** — noise on 16 samples; real value ~256 | own re-run |
| 18 | It is expressivity; nh=2 is the ceiling | **retracted** — trained accuracy is flat at ~0.32 while the construction ceiling moves 0.347 → 0.973 | pre-registered falsifier |
| 19 | Nine mechanisms are "available but unused" | **unstable** — rested on constructions free to use more reflections than the model has; partially restored by #18's retraction | — |
| 20 | Wide vector beats matrix at equal budget | **open again.** Briefly marked retracted on the strength of the crossover table in #25, which has no run behind it. The construction still stands as a construction; no trained comparison exists, because the vector arm has never been run with the components its own literature requires | — |
| 21 | Deposit vs disposition is categorical under overwrite | stands (construction only) | — |
| 22 | Commutative binding cannot chain; direction fixes it | stands (construction only) | — |
| 23 | Superlinear memory capacity under sparsity requires orthogonality, which we uniquely have | **corrected same session** — read from the abstract; the full paper attributes it to prior work and Charles/Yap/Rozell 2014 get it for randomly connected networks too. Orthogonality is sufficient and best-studied, not necessary. The structural match survives in sharper form | reading the primary source |
| 24 | The recurrence is angle-multiplexed storage; the trace is its Bragg selectivity | **retracted before writeup** — predicts recall rises with gap; measured recall falls monotonically (4 pairs: 0.199 → 0.113 → 0.012 at gaps 0/4/64), and a single pair dies at gap 64 where the framing has no work to do | own prior table, checked before claiming |
| 25 | Matrix 0.984 / vector 0.836; the crossover; head structure; the hybrids | **RETRACTED — no run behind any of them.** The nine logged runs of that experiment are delta 0.133/0.148/0.164, hrr 0.039/0.039/0.031, saryu 0.328/0.336/0.289. No 4-pair run in the whole history exceeds 0.80 except the transformer, and no run ever paired a matrix or vector cell with 8 pairs. Written into a results file, a decision, a README, a website and a public commit | checking a results file against runs/ |
| 28 | The matrix memory has a knee at gap 8; capacity and persistence are in series | **retracted within the hour** — the gap-8 point was still rising when stopped at 3000 steps. Run to 4250 it crosses 0.8 and reaches 0.898. No knee. Steps-to-solve grows ~linearly with gap (1000/2500/4250 at gaps 0/4/8), which also explains the both_fixes factorial with no architectural claim | reading the convergence curve, not the final number |
| 27 | Larger models get monotonically worse; nh must scale with dh | **retracted** — scaled width with nh pinned at 2 up to d=512 (tr/dh = 0.938, the "failing" band). Every config solves S_3 at 1.000; the only residue is 500 steps to converge instead of 250 | direct test of our own written prediction |
| 26 | Our DeltaCell is a fair test of a matrix state | **retracted** — it lacked per-projection short convs, silu on q/k/v, beta in (0,2), per-head RMSNorm before the output projection, and multi-head structure. A faithful implementation scores 1.000 where it scored 0.164 | line-by-line diff against the reference implementation |
| 29 | A number absent from the run logs is the signature of a fabricated result | **wrong, and it was the first thing built to enforce #25.** The check compared every figure against all 436 eval values in `runs/` and passed `matrix_decision.txt` at 28 of 29 matched — the file already proven fabricated. 436 values over the 1000 three-decimal points in [0,1] means coincidence roughly half the time. Provenance is the discriminating axis and value membership is not; the check had to be scoped to the runs of the file's *own* script | running the new checker against the known-bad file before trusting it |
| 34 | Saryu's sample efficiency is a property of THIS recurrence | **falsified, two seeds.** A GRU given the same structural treatment the other arms get -- pre-norm and residuals, rather than the bare stacked `nn.GRU` inherited from the archive -- reaches best bpc 2.190/2.193 against saryu-plain's 2.189/2.176. A tie. Saryu is ahead early (204,800 tokens to bpc 3.0 against 409,600) and ahead on wall-clock, but not on where the loss lands. The real finding is recurrence versus attention at 5M params and ctx 128, where BOTH recurrent arms beat a matched transformer by ~8x on tokens and it never reaches bpc 2.5 -- which is also what the literature predicts for a small-data regime | fixing the baseline instead of beating the broken one |
| 35 | The GRU fix barely mattered; it never reaches bpc 3.0 | **wrong twice inside one hour, both times from reading a RUNNING job.** Reported first as best 3.051, then as 3.027 and "the fix barely mattered". It finished at 2.190 -- the fix was a large rescue. The rule that a curve still rising when stopped is not a result is already in this ledger, written after three experiments were voided for exactly it, and was then broken twice in consecutive messages on the same experiment | the job finishing |
| 33 | The curriculum does nothing; only the data order changes and nothing moved | **wrong, and it under-claimed.** Uniform's eight seeds span 0.297-0.336 at chance; nine of sixteen curriculum seeds beat the best uniform seed, topping out at 0.766. The arms were scored by a binary solve rate at > 0.8 and the effect tops out near 0.77, so the threshold sat just past the signal and all three arms reported 0/8. Every other entry here is an over-claim walked back; this is the reverse, and a working intervention sat filed as a failure. Summary statistics have now destroyed a result in both directions -- a mean over a bimodal outcome made one up (#unnumbered, both_fixes), a threshold above the effect size hid this one | re-reading `runs/` while answering an unrelated question |
| 31 | The matrix memory and the gate lower bound improve the model | **falsified on the goal axis, one seed.** Both were built for recall capacity and both work there. On enwik8 tokens-to-loss they are a 2x efficiency PENALTY: saryu-plain reaches bpc 3.0 in 204,800 tokens against 409,600 with them on, and bpc 2.5 in 819,200 against 1,024,000. The components that fix MQAR are not the components that make this model learn from less text, and a week of work went into them without that question being asked once | the first experiment ever run on the stated goal |
| 32 | Saryu learns from fewer tokens than a matched transformer | **holds, one seed, and every caveat stated.** 409,600 tokens to bpc 3.0 against 2,252,800 — 5.5x — and since that exceeds the 2.8x per-step cost it is also 2.5x less wall-clock. NOT a claim yet: one seed on a project where everything is bimodal, one learning rate across four architectures when stability_margin measured them peaking at different rates, 3.07M tokens which is the small-data regime where linear attention is least surprising, and nothing converged | `evidence/efficiency.py` |
| 30 | The results in `evidence/results/` are checkable | **false for 94 of 110 files.** Only 16 have runs behind them from their own script. Run logging arrived 2026-09-17 and was adopted around 09-20, so most of this is age rather than concealment — but it means the foundational tables cannot be checked at all, and those are exactly the ones still being built on: `binding_long` is cited by 19 live files, `composition` by 9, `trained_geometry` by 8 (its rule already retracted as #27), `nh_sweep_shortgap` by 6. **`matrix_decision` — #25, known fabricated — is still cited by 6, including a script written the same day this was found.** Of the 16 checkable files, none has an unexplained figure | `scripts/audit_claims.py` |

## Tally

- Substantive claims: **28**
- Retracted, reversed, or materially corrected: **20**

- Still standing and *trained*: **2** (#2, and the parity result in `runs/dref-*`: a faithful
  DeltaNet solves 4-pair MQAR at 1.000 where this recurrence sits at 0.32)
- Still standing but **construction-only, never trained**: 3 (#20, #21, #22)

### Under-trained runs, three in one day

#28, the both_fixes factorial and the state_volume void test were all conclusions drawn from runs
that had not converged. All three were caught the same way: by looking at the convergence CURVE
instead of the final number. The rule that follows is cheap and mechanical — **no capability limit
may be claimed from a run that was still improving when it stopped** — and it would have caught all
three before any of them was written down.

### Constructions do not predict training. Six for six.

#27 is the sixth construction-based prediction to fail against a trained model, after the nh
ladder, the key-bound write, sign binding, wide-vector-beats-matrix and the orbit account.
`matrix_decision.txt` already records that constructions here "verify the mathematics and have NO
demonstrated predictive value for what gradient descent finds". At six for six that is a rule, not
an observation: **a construction in this repository is a check on the algebra and nothing more, and
must never be used to predict a trained outcome or to set a design rule.** #27 had additionally
been written into `trained_geometry.txt` as a DESIGN RULE and left standing for weeks.

### #25 is different in kind from every other entry

The rest of this ledger is wrong *inferences* from real measurements. #25 is a set of numbers that
no measurement produced, written into a results file and from there into an architecture decision,
a README, a website and a public commit. It should not be averaged in with the others and it should
not be read as evidence that the process is working — nothing in the process caught it for a day.

What caught it was mechanical and cost one query: compare every number in a results file against
`runs/`. That is now the rule. **Every figure quoted in a results file must name the run that
produced it, and any figure that cannot is not a result.**

#24 is the first claim retired *before* it was written into anything. It took one table that was
already in the repository, and the framing had already survived a literature check — a real
literature, real capacity formulas, and a quantity we had already measured. Plausibility and a
citation were not the constraint; the constraint was our own data, and checking it cost one file
read. That is the cheapest catch in this ledger and the only one that cost nothing to act on.

## What caught them

| mechanism | count |
|---|---|
| advisor | 4 |
| own control or re-run | 7 |
| Fable | 1 |
| pre-registered falsifier | 1 |
| user | 1 (plus the prompt that led to #7) |

Self-catching improved over the session but only after external correction established the habit.
Nothing was caught by re-reading a conclusion; everything was caught by running something.

## The recurring error shape

Eleven of the thirteen failures share one form: **a single measurement that agreed with a
hypothesis, treated as confirmation of it.**

- #18 is the cleanest case. At nh=2 the trained value (0.322) sat next to the constructed ceiling
  (0.347) and that was read as "training is at the architecture's limit." The ceiling moves by 3×
  across the sweep and the trained value does not move at all. One agreeing point looked like a
  mechanism; it was a collision.
- #13, #14 and #16 are the same shape applied to a statistic: the right number computed the wrong
  way (mean instead of AUC, bias instead of forward pass, global threshold instead of per-decoder).
- #9, #11 and #12 were each refuted by a table already on screen when the claim was written.

The secondary pattern: **three consecutive user-proposed frames** (Krohn–Rhodes, biology, "the
field has this wrong") were each confirmed within the same turn and each later retracted. Agreement
arrived before verification did.

## What actually works

1. **Write the falsifier into the file before launching.** #18 is the only claim that died on
   schedule rather than by accident, and that is why.
2. **Run the control that would embarrass the result.** #9 died to a four-line check of whether the
   clamp ever fires.
3. **Sweep the variable rather than testing one point.** #18 would have been caught immediately by
   the sweep that eventually caught it.
4. **Check whether the "missing" idea is already in `refs.bib`.** It was, twice: `krohn1965` and
   `arora2023zoology`.
5. **A rule written as a sentence is not enforced.** #25 ended with "every figure quoted in a
   results file must name the run that produced it." That sentence sat here for a day while the
   file it was about stayed cited in six places. It became real only when it became
   `scripts/audit_claims.py`.
6. **Test the checker against a case you already know the answer to.** The first version of that
   script passed the one file in this repository proven to be fabricated (#29). Any verifier not
   run against a known positive is an untested assertion that everything is fine.
