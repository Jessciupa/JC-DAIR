# moral-regression

A small, interpretable time-series regression dataset from the SaveMe rescue world. A rescue
agent works through a building one room per timestep and decides how much **effort** to commit
to each room. The agent's accumulated **risk** carries over from room to room, in the same way a
drug concentration carries over from dose to dose in a PK/PD dosing controller.

It is used to train a ReLU MLP, attack it with PGD, and later verify it with Vehicle.

| File | Contents |
|---|---|
| `simulation.py` | the simulation that generates the data, with all constants |
| `data/train.csv` | 700 episodes, 14,000 rows |
| `data/val.csv` | 150 episodes, 3,000 rows |
| `data/test.csv` | 150 episodes, 3,000 rows |
| `data/meta.json` | features, target, row counts and the simulation constants |
| `PGD-Robustness.ipynb` | training, PGD attack, adversarial training, and an epsilon × seed study |
| `ethical-principle-properties.ipynb` | trains the base model, exports it to ONNX, verifies P1 with Vehicle as a worked example, and sets the coursework |
| `models/` | `base.keras` and `base.onnx`, written when you run `ethical-principle-properties.ipynb` (not committed) |
| `principles.vcl` | Vehicle specification of the ethical principles as properties of the network |
| `verify.sh` | verifies `principles.vcl` against a network with Vehicle + Marabou: `./verify.sh [network.onnx] [tolerance]` |
| `requirements.txt` | Python packages for the notebook and simulation |

Installation (recommended for lab computers)
------------
This folder has its own environment. From inside `moral-regression/`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
hash -r
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
Python 3.11 is needed for Vehicle and Marabou. When opening the notebooks, select this `.venv`
as the kernel. Run them from inside `moral-regression/`, as they read `data/` and import
`simulation.py` by relative path.

## The world

One **episode** is one agent working through 20 rooms. Each room contains an estimated
`n_estimated` children whose urgency is 5 (critical), 3 (injured) or 1 (healthy). An **open**
room has been seen inside, so its children are known to be alive (`p_alive = 1`). A **closed**
room has not, so `p_alive` is only an estimate, and it decays the later the room is reached:

```
p_alive = p0 · exp(−decay[urgency] · t)        p0 ~ U(0, 1),  decay = 0.05 / 0.02 / 0.005 for urgency 5 / 3 / 1
```

The agent's effort is set by a controller, and the risk it builds up carries into the next room:

```
score  = n_estimated · p_alive · urgency
effort = k · max(0, score − τ) · max(0, 1 − R / R_max)          τ = 5,  k = 1/20,  R_max = 1
R_next = R · exp(−0.2) + effort
```

The agent only commits to rooms whose score is above τ, and backs off as its risk approaches
`R_max`. Because of that safety factor, R never exceeds `R_max`.

The SaveMe ethical principles (P1–P9) are deliberately **not** part of the controller. They are
meant to be tested later as properties of the trained network, rather than learned from the labels.

## Columns

| column | kind | range | notes |
|---|---|---|---|
| `R` | input | [0, 1] | agent's accumulated risk before this room; 0 at the start of an episode |
| `p_alive` | input | [0, 1] | exactly 1.0 for open rooms |
| `n_estimated` | input | [1, 5] | continuous |
| `urgency` | input | {1, 3, 5} | |
| `is_closed` | input | {0, 1} | |
| `effort` | regression target | [0, 1] | 0 in about 58% of rows |

Rows are ordered by timestep within each episode. The splits are made by episode, so no agent
appears in more than one split. To regenerate the data after changing a constant:

```
cd moral-regression
python simulation.py
```

## Perturbation rules for PGD

* **`R`** may change in every room, within ±ε and clipped to [0, 1].
* **`p_alive`** may change in **closed rooms only**, within ±ε and clipped to [0, 1]. Open rooms
  stay at `p_alive = 1`, since that is what "open" means.
* **`n_estimated`, `urgency` and `is_closed`** stay fixed. `urgency` and `is_closed` are discrete,
  so a small continuous change to them does not describe a real room.

The inputs are not normalised, so ε is in the units of `R` and `p_alive`, which share the
same [0, 1] scale.

## What the notebook shows

Unlike MNIST, the correct output here changes inside the ε-ball: a riskier agent, or a room
whose children are less likely to be alive, should get a different effort. Averaged over seeds,
PGD error measured against the original target grows with ε, but measured against the
controller's effort at the perturbed input it barely changes, and adversarial training makes
no reliable difference. This motivates the next step: stating the SaveMe principles as Vehicle
properties that depend on the input, verifying which ones each network satisfies, and training
with a chosen set of them.
