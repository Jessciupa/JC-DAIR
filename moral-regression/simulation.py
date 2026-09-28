"""SaveMe rescue simulation — generates state-action training data.

One episode is one rescue agent working through a building, one room per timestep.
The agent's accumulated risk R carries over from room to room, in the same way the
drug concentration carries over from dose to dose in the PK/PD simulation. Each row
is the state the agent sees at one timestep and the effort the controller commits:

    X = [R, p_alive, n_estimated, urgency, is_closed]      y = effort in [0, 1]

Run `python simulation.py` from this folder to regenerate data/.
"""

import json
import os

import numpy as np

# --- Simulation parameters ---
NUM_EPISODES = 1000
TIMESTEPS = 20              # rooms per episode
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, split by episode
SEED = 0

# --- Room sampling ---
P_CLOSED = 0.5              # share of rooms nobody has seen inside
N_RANGE = (1.0, 5.0)        # estimated number of children
URGENCY_LEVELS = (1, 3, 5)  # healthy / injured / critical

# --- Survival decay (closed rooms) ---
# p_alive = p0 * exp(-DECAY[urgency] * t): the later a room is reached, the lower the
# chance its children are still alive, and critical children deteriorate fastest.
DECAY = {1: 0.005, 3: 0.02, 5: 0.05}  # per timestep

# --- Risk dynamics (≈ PK curve) ---
# R_next = R * exp(-RISK_RECOVERY) + RISK_PER_EFFORT * effort
RISK_RECOVERY = 0.2         # recovery rate per timestep (≈ Ke, elimination)
RISK_PER_EFFORT = 1.0       # risk added by full effort (≈ dose / Vd)
R_MAX = 1.0                 # risk at which the controller stops committing (≈ C_max)

# --- Effort controller (≈ compute_dose) ---
TAU = 5.0                   # score the room must exceed before any effort (≈ T_norm upper bound)
SCORE_MAX = N_RANGE[1] * 1.0 * max(URGENCY_LEVELS)
K_EFFORT = 1.0 / (SCORE_MAX - TAU)  # scales effort to [0, 1]

COLUMNS = ['R', 'p_alive', 'n_estimated', 'urgency', 'is_closed', 'effort']


def sample_room(rng, t):
    """Sample one room as the agent reaches it at timestep t."""
    n_estimated = rng.uniform(*N_RANGE)
    urgency = rng.choice(URGENCY_LEVELS)
    is_closed = int(rng.rand() < P_CLOSED)

    # Open rooms have been seen inside, so the children are known to be alive
    if is_closed:
        p_alive = rng.uniform(0, 1) * np.exp(-DECAY[urgency] * t)
    else:
        p_alive = 1.0
    return p_alive, n_estimated, urgency, is_closed


def compute_score(p_alive, n_estimated, urgency):
    """Expected benefit of a rescue: children likely alive, weighted by urgency.

    The SaveMe principles (P1-P9) are deliberately NOT part of the controller, so that
    later they can be tested as properties of the trained network rather than learned.

    Alternative for the future: expected children saved, n_estimated * p_alive * rs,
    with rs = 0.50 / 0.70 / 0.95 for urgency 5 / 3 / 1. It weights urgency by rescue
    success instead of need, so P1 would not hold by construction.
    """
    return n_estimated * p_alive * urgency


def compute_effort(p_alive, n_estimated, urgency, R):
    """Effort controller: commit to rooms above TAU, backing off as risk builds up."""
    score = compute_score(p_alive, n_estimated, urgency)
    effort_from_score = K_EFFORT * max(0.0, score - TAU)

    safety_factor = max(0.0, 1 - R / R_MAX)
    return float(np.clip(effort_from_score * safety_factor, 0.0, 1.0))


def next_risk(R, effort):
    """Risk carried into the next room: past risk recovers, new effort adds to it."""
    return R * np.exp(-RISK_RECOVERY) + RISK_PER_EFFORT * effort


def simulate_episode(rng):
    """Run one agent through TIMESTEPS rooms. Returns an array of shape (TIMESTEPS, 6)."""
    R = 0.0
    rows = []

    for t in range(TIMESTEPS):
        p_alive, n_estimated, urgency, is_closed = sample_room(rng, t)
        effort = compute_effort(p_alive, n_estimated, urgency, R)

        rows.append([R, p_alive, n_estimated, urgency, is_closed, effort])

        R = next_risk(R, effort)

    return np.array(rows)


def simulate(num_episodes=NUM_EPISODES, seed=SEED):
    """Simulate all episodes. Returns a list of per-episode arrays."""
    rng = np.random.RandomState(seed)
    return [simulate_episode(rng) for _ in range(num_episodes)]


def save_splits(episodes, out_dir='data', seed=SEED):
    """Split by episode (so no agent appears in two splits) and write the CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    order = np.random.RandomState(seed).permutation(len(episodes))
    n_train = int(SPLIT[0] * len(episodes))
    n_val = int(SPLIT[1] * len(episodes))
    splits = {
        'train': order[:n_train],
        'val': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }

    counts = {}
    for name, idx in splits.items():
        data = np.concatenate([episodes[i] for i in idx], axis=0)
        np.savetxt(os.path.join(out_dir, f'{name}.csv'), data, delimiter=',',
                   header=','.join(COLUMNS), comments='', fmt='%.10g')
        counts[name] = len(data)

    meta = {
        'seed': seed,
        'episodes': len(episodes),
        'timesteps': TIMESTEPS,
        'features': COLUMNS[:-1],
        'target': COLUMNS[-1],
        'rows': counts,
        'TAU': TAU,
        'K_EFFORT': K_EFFORT,
        'R_MAX': R_MAX,
        'RISK_RECOVERY': RISK_RECOVERY,
        'RISK_PER_EFFORT': RISK_PER_EFFORT,
        'DECAY': DECAY,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return counts


if __name__ == '__main__':
    print(save_splits(simulate()))
