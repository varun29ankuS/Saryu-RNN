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
| 20 | Wide vector beats matrix at equal budget | stands (construction only) | — |
| 21 | Deposit vs disposition is categorical under overwrite | stands (construction only) | — |
| 22 | Commutative binding cannot chain; direction fixes it | stands (construction only) | — |

## Tally

- Substantive claims: **22**
- Retracted, reversed, or materially corrected: **13**
- Still standing and *trained*: **1** (#2)
- Still standing but **construction-only, never trained**: 3 (#20, #21, #22)

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
