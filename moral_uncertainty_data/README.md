# SaveMe coursework dataset (simple)

This is a small, interpretable regression dataset taken from the SaveMe rescue world. It
is used to train a ReLU MLP, attack it with masked PGD, explain it with SHAP and LIME, and
later verify it with Vehicle. 

| File | Contents |
|---|---|
| `data/train.csv` | 14,000 rooms |
| `data/val.csv` | 3,000 rooms |
| `data/test.csv` | 3,000 rooms |
| `data/meta.json` | τ, feature order and ranges, the `rs` mapping, the perturbation mask, row counts |

## The world

A rescue agent must decide whether a room is worth attempting. Each room contains an
estimated `n_estimated` children, whose urgency is 5 (critical), 3 (injured) or 1
(healthy). An **open** room has been seen inside, so its children are known to be
present (`p_alive = 1`). A **closed** room has not, so `p_alive ∈ [0, 1]` is only an
estimate. Harder cases are also harder to rescue: the rescue-success probability `rs` is
0.50, 0.70 or 0.95 for urgency 5, 3 or 1. Three of SaveMe's ethical principles score the
room. P1 favours high expected value, P7 penalises uncertainty, and P9 penalises likely
failure. The agent attempts the room when the score reaches the threshold τ.

## Formulas

These are the formulas and constants of the ([SaveMe](https://www.research.ed.ac.uk/en/publications/interpretable-moral-decision-making-under-epistemic-uncertainty/)) code

```
P1 = 3 · n_estimated · p_alive · urgency                           maximise expected utility
P7 = 8 · (1 − p_alive)          if closed,               else 0    defer to certainty
P9 = 30 · (0.3 − p_alive · rs)  if p_alive · rs < 0.3,   else 0    avoid likely failure

score    = P1 − P7 − P9
decision = 1 if score ≥ τ else 0          τ = 12.68 (exact value in meta.json)
```

P9 never fires for an open room, because `rs ≥ 0.5 > 0.3` when `p_alive = 1`. The score
is piecewise linear in `p_alive`, with a kink where `p_alive · rs = 0.3`.

## Sampling

| `sample_source` | share | how |
|---|---|---|
| `uniform` | 70% | `is_closed ~ Bernoulli(0.5)`; `urgency` uniform over {5,3,1}; `n_estimated ~ U[1,5]`; `p_alive = 1` if open, else `U(0,1)` |
| `p9_kink` | 15% | closed rooms with `p_alive · rs ~ U[0.2, 0.4]` |
| `boundary` | 15% | closed rooms with `|score − τ| < 1` |

τ is the median over the `uniform` rows, i.e. over the environment distribution. It is
not the median of the oversampled mix, so the overall `decision` rate is a little below
0.5. Rows are shuffled and split 70/15/15 with seed 0.

## Columns

| column | kind | range | notes |
|---|---|---|---|
| `p_alive` | input | [0, 1] | exactly 1.0 for open rooms |
| `n_estimated` | input | [1, 5] | continuous |
| `urgency` | input | {1, 3, 5} | also fixes `rs` |
| `is_closed` | input | {0, 1} | |
| `p1`, `p7`, `p9` | true terms | ≥ 0 | unsigned magnitudes; `score = p1 − p7 − p9` |
| `score` | regression target | ≈ [−17, 75] | |
| `decision` | label | {0, 1} | `score ≥ τ` |
| `sample_source` | meta | | `uniform` / `p9_kink` / `boundary` |

The network's inputs are **only the four input columns**, in the order listed. The other
columns are ground truth for evaluating and explaining your model. Feeding them in as
inputs gives the answer away.

## Perturbation mask

PGD, and later the Vehicle robustness property, may change **only `p_alive`, and only for
closed rooms**, within `|δ| ≤ ε` and clipped to [0, 1]. Everything else is fixed.

* **Open rooms stay at `p_alive = 1`.** That is what "open" means. A perturbed open room
  is a state the world cannot produce.
* **`urgency` and `is_closed` are discrete.** A small continuous change to them does not
  describe any real room.
* **`n_estimated` is held fixed.** The question is robustness to uncertainty about
  survival, not about the head-count.

In Vehicle this is roughly: `is_closed = 1`, `|p_alive' − p_alive| ≤ ε`,
`0 ≤ p_alive' ≤ 1`, and every other input unchanged ⇒ `(f(x') ≥ τ) = (f(x) ≥ τ)`.

## Attacking a regression model

The network outputs a score, not a class, so an attack succeeds when it changes the
**decision**. The perturbed prediction must land on the other side of τ from the original
prediction, `(f(x') ≥ τ) ≠ (f(x) ≥ τ)`. So if f(x) ≥ τ, push the prediction down;
otherwise push it up. Compare against the model's own prediction f(x), not the
`decision` column.

* **ε is in `p_alive` units.** ε = 0.05 means `p_alive` may move by ±0.05. If you
  normalise inputs, convert ε accordingly, or attack the raw inputs.
* **Only closed rooms can flip.** Report flip rates over closed rooms, because open rooms
  can't be perturbed at all.
* **Some flips are correct.** Near τ, even the true score crosses the threshold within ε.
  Recompute the true score at the perturbed input with the formulas above to tell a
  network flaw from a decision that is genuinely fragile.
