"""Generate the moral drift dataset: how the Moral Machine's cultural clusters' preferences, and
so their stay/swerve choices, could drift over a 20-year deployment.

    python generate.py

The pipeline has three stages:

1. FEATURES. Every dilemma of moral-classification is described by a vector phi: the attributes
   of the characters swerving saves minus those it kills, along the dimensions Awad et al.
   (2018) report (more lives, young, female, fit, high status, human, lawful, pedestrians).

2. FIT (from data). Each cluster c gets a logistic model of its share of swerve votes,

       P(swerve) = sigmoid(theta_c . [1, phi])

   fitted to the real votes in moral-classification/dataset/votes.csv.gz. theta_c are the
   cluster's present-day preference weights, with standard errors.

3. DRIFT (assumed). The weights evolve under Friedkin-Johnsen opinion dynamics with noise,

       d theta_i = gamma * (-theta_i + sum_j a_ij theta_j + (1 - sum_j a_ij) u_i) dt + sigma dW

   and every year each cluster's majority choice is recorded on all 2,295 dilemmas.

Stage 2 is estimated from data. The parameters of stage 3 (a_ij, gamma, sigma) are
ASSUMPTIONS: placeholders to be replaced with values from the literature or longitudinal data.

Writes to data/:
    preferences.json   fitted theta, standard errors and accuracy per cluster; phi of every dilemma
    trajectories.csv   scenario, run, year, cluster and its weights
    choices.npz        p_swerve and choice (1 = swerve), shape [scenario, run, year, cluster, dilemma]
    cora_params.json   the same model as an uncertain linear system, for cora/moral_drift_FJ.m
    meta.json          every parameter and the seed
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MORAL_CLASSIFICATION = ROOT.parent / 'moral-classification'
sys.path.insert(0, str(MORAL_CLASSIFICATION / 'dataset'))
from moral_machine import CLUSTERS, decode_side  # noqa: E402

VOTES = MORAL_CLASSIFICATION / 'dataset' / 'votes.csv.gz'
DILEMMAS = MORAL_CLASSIFICATION / 'data' / 'dilemmas.jsonl'


# =================================================================================================
# 1. FEATURES: a dilemma as attribute differences
# =================================================================================================

# The first weight is the ACTION BIAS: its feature is always 1, so it shifts every dilemma
# towards stay (negative) or swerve (positive) regardless of who is involved. The other eight
# weights say how strongly a cluster spares each attribute.
DIMENSIONS = ['action', 'more', 'young', 'female', 'fit', 'status', 'human', 'lawful', 'pedestrians']

# How much each character counts towards an attribute (+1 or -1). Characters not listed count 0.
# E.g. sparing a boy instead of an old man scores +2 on 'young'.
YOUNG = {'Boy': 1, 'Girl': 1, 'Stroller': 1, 'OldMan': -1, 'OldWoman': -1}
FEMALE = {'Woman': 1, 'Girl': 1, 'OldWoman': 1, 'LargeWoman': 1, 'Pregnant': 1, 'FemaleExecutive': 1,
          'FemaleAthlete': 1, 'FemaleDoctor': 1,
          'Man': -1, 'Boy': -1, 'OldMan': -1, 'LargeMan': -1, 'MaleExecutive': -1,
          'MaleAthlete': -1, 'MaleDoctor': -1}
FIT = {'FemaleAthlete': 1, 'MaleAthlete': 1, 'LargeWoman': -1, 'LargeMan': -1}
STATUS = {'FemaleExecutive': 1, 'MaleExecutive': 1, 'FemaleDoctor': 1, 'MaleDoctor': 1,
          'Homeless': -1, 'Criminal': -1}
PETS = {'Dog', 'Cat'}
SIGNAL = {0: 0, 1: 1, 2: -1}  # no light, crossing on green (lawful), crossing on red (unlawful)


def side_attributes(passengers, signal, counts):
    """Attribute scores of the characters on one side of a dilemma."""
    def total(weights):
        return sum(weights.get(c, 0) * n for c, n in counts.items())

    n = sum(counts.values())
    pets = sum(n_c for c, n_c in counts.items() if c in PETS)
    return np.array([
        n,
        total(YOUNG),
        total(FEMALE),
        total(FIT),
        total(STATUS),
        n - 2 * pets,          # humans minus pets
        SIGNAL[signal],
        0 if passengers else 1,
    ], dtype='float64')


def features(stay, swerve):
    """[1, phi]. Staying kills the stay side, so swerving SAVES it and KILLS the swerve side:
    phi > 0 on a dimension means swerving spares more of that attribute."""
    return np.concatenate([[1.0], side_attributes(*stay) - side_attributes(*swerve)])


def load_votes():
    """[1, phi] for every scenario in votes.csv.gz, and its votes per cluster."""
    votes = pd.read_csv(VOTES)
    X = np.array([features(decode_side(a), decode_side(b)) for a, b in zip(votes['stay'], votes['swerve'])])
    return X, votes


def load_dilemmas():
    """[1, phi] for the 2,295 dilemmas of moral-classification, with their ids and choices."""
    rows = [json.loads(line) for line in open(DILEMMAS)]

    def side(s):
        return s['passengers'], s['signal'], s['characters']

    X = np.array([features(side(r['stay']), side(r['swerve'])) for r in rows])
    return X, rows


# =================================================================================================
# 2. FIT: present-day preferences from the votes
# =================================================================================================

RIDGE = 1e-6       # keeps IRLS stable for dimensions that rarely vary
IRLS_STEPS = 50


def fit_logistic(X, successes, trials):
    """Binomial logistic regression by IRLS (Newton's method). Returns weights and standard errors.

    Each scenario counts once per vote, so scenarios answered by many users weigh more. The
    standard errors come from the inverse Hessian. They measure how precisely we know TODAY'S
    preferences, which is why they become the radius of CORA's initial set below. They are tiny
    because there are millions of votes, and they say nothing about how fast preferences change.
    """
    theta = np.zeros(X.shape[1])
    for _ in range(IRLS_STEPS):
        p = 1 / (1 + np.exp(-X @ theta))
        w = trials * p * (1 - p)
        H = X.T @ (w[:, None] * X) + RIDGE * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (successes - trials * p))
        theta += step
        if np.abs(step).max() < 1e-10:
            break
    return theta, np.sqrt(np.diag(np.linalg.inv(H)))


def fit_clusters(X, votes, X_dilemmas, dilemmas):
    """theta and standard errors per cluster, and how often the fitted model's majority matches
    the cluster's real choice in dilemmas.jsonl. The model is linear in phi, so it cannot get
    every dilemma right; about one in ten is wrong, mostly near 50/50."""
    clusters = {}
    for c in CLUSTERS:
        keep = votes[f'votes_{c}'] > 0
        theta, se = fit_logistic(X[keep], votes[f'swerve_{c}'][keep].to_numpy(float),
                                 votes[f'votes_{c}'][keep].to_numpy(float))

        predicted = np.where(X_dilemmas @ theta > 0, 'swerve', 'stay')
        actual = np.array([r[c]['choice'] for r in dilemmas])
        decided = actual != None  # noqa: E711, ties have no choice
        accuracy = float((predicted[decided] == actual[decided]).mean())

        clusters[c] = {'theta': theta.tolist(), 'se': se.tolist(), 'accuracy': accuracy,
                       'votes': int(votes[f'votes_{c}'].sum())}
    return clusters


# =================================================================================================
# 3. DRIFT: Friedkin-Johnsen opinion dynamics
# =================================================================================================
#
#     d theta_i = gamma * (-theta_i + sum_j a_ij theta_j + (1 - sum_j a_ij) u_i) dt + sigma dW
#
# u_i     cluster i's ANCHOR: its preferences at deployment. Friedkin-Johnsen agents are
#         "stubborn": they never forget where they started, so the clusters settle on a
#         compromise rather than full consensus.
# a_ij    how much cluster i listens to cluster j. Its row sum is i's SUSCEPTIBILITY: 0 means
#         i keeps its anchor, close to 1 means i is carried along by the others.
# gamma   how fast the clusters move towards that compromise, per year.
# sigma   random year-to-year change in preferences that the model does not explain.
#
# The drift is linear in theta, which is what makes it verifiable by reachability in CORA:
# a_ij uncertain in an interval gives a linear system with an uncertain matrix (linParamSys).

YEARS = 20
STEPS_PER_YEAR = 12
RUNS = 50
SEED = 0

# ASSUMPTION. Row i listens to column j. Each a_ij lies in A_CENTRE +- A_RADIUS; each run draws
# one value from that interval and CORA covers all of them at once. Row sums of
# A_CENTRE + A_RADIUS must stay below 1, or a cluster would forget its anchor completely.
# To calibrate: cultural distance between clusters (e.g. Muthukrishna et al., 2020), or
# media, trade and migration flows as proxies for influence.
A_CENTRE = np.array([[0.30, 0.10, 0.10],
                     [0.15, 0.20, 0.15],
                     [0.10, 0.10, 0.35]])
A_RADIUS = 0.3 * A_CENTRE

# ASSUMPTION. Drift rate in 1/years: deviations from the compromise halve in about
# ln 2 / gamma = 3.5 years. To calibrate: change between waves of the World Values Survey or
# European Values Study on related items.
GAMMA = 0.2

# ASSUMPTION. Noise per dimension, in log-odds per sqrt(year). Around its equilibrium a weight
# then wobbles with a standard deviation of about SIGMA / sqrt(2 * GAMMA) = 0.016.
# It is set directly, NOT from the fit's standard errors: those shrink with the number of votes,
# and more votes do not make a society change its mind more slowly. CORA has no stochastic
# noise; it would enter there as a bounded disturbance (see cora_params.json, 'sigma').
SIGMA = np.full(len(DIMENSIONS), 0.01)

# baseline:         no influence (a = 0); each cluster only wobbles around its own anchor
# social_influence: the clusters pull towards each other, held back by their anchors
SCENARIOS = ['baseline', 'social_influence']


def sample_coupling(rng, scenario, runs):
    """Influence matrices A for each run, shape (runs, N, N)."""
    if scenario == 'baseline':
        return np.zeros((runs, len(CLUSTERS), len(CLUSTERS)))
    return A_CENTRE + A_RADIUS * rng.uniform(-1, 1, size=(runs, len(CLUSTERS), len(CLUSTERS)))


def drift(theta, u, A):
    """The Friedkin-Johnsen drift term for all runs at once: theta, u (runs, N, D), A (runs, N, N).

    -theta + A @ theta + (1 - susceptibility) * u: each cluster moves towards a weighted mix of
    the others' current preferences and its own anchor."""
    susceptibility = A.sum(axis=2, keepdims=True)
    return GAMMA * (-theta + A @ theta + (1 - susceptibility) * u)


def simulate(rng, scenario, theta0, se, runs=RUNS):
    """Euler-Maruyama trajectories with monthly steps. Returns theta per year, (runs, YEARS + 1, N, D)."""
    # Today's preferences are only known up to their standard errors, so each run starts from
    # a different draw. CORA's initial set is the box theta0 +- se that these draws come from.
    u = theta0 + se * rng.standard_normal((runs, *theta0.shape))
    A = sample_coupling(rng, scenario, runs)
    dt = 1 / STEPS_PER_YEAR

    theta = u.copy()
    yearly = [theta.copy()]
    for step in range(1, YEARS * STEPS_PER_YEAR + 1):
        noise = SIGMA * np.sqrt(dt) * rng.standard_normal(theta.shape)
        theta = theta + drift(theta, u, A) * dt + noise
        if step % STEPS_PER_YEAR == 0:
            yearly.append(theta.copy())
    return np.stack(yearly, axis=1)


def swerve_share(theta, phi):
    """Each cluster's predicted share of swerve votes per dilemma: (..., N, D) -> (..., N, K).
    The majority choice is swerve when this is above one half, i.e. when theta . phi > 0."""
    return 1 / (1 + np.exp(-theta @ phi.T))


def sample_votes(rng, p, votes):
    """Optional: synthetic vote counts, Binomial(votes, p), for data shaped like the real votes."""
    return rng.binomial(votes, p)


# =================================================================================================
# Outputs
# =================================================================================================

def trajectories_table(thetas):
    rows = []
    for s, scenario in enumerate(SCENARIOS):
        runs, years, n, _ = thetas[s].shape
        for r in range(runs):
            for t in range(years):
                for i, c in enumerate(CLUSTERS):
                    rows.append([scenario, r, t, c, *thetas[s][r, t, i]])
    return pd.DataFrame(rows, columns=['scenario', 'run', 'year', 'cluster', *DIMENSIONS])


def save_cora_params(dilemma_ids, phi, theta, se):
    """The model as an uncertain linear system for cora/moral_drift_FJ.m.

    x0 +- x0_rad is the initial set, a_c +- a_r the interval coupling. The action bias is the
    first dimension with phi = 1 in every dilemma, so each dilemma's offset b is 0. sigma is
    exported for a bounded disturbance; the script does not use it yet."""
    params = {
        'clusters': CLUSTERS,
        'dimensions': DIMENSIONS,
        'x0': theta.tolist(),
        'x0_rad': se.tolist(),
        'a_c': A_CENTRE.tolist(),
        'a_r': A_RADIUS.tolist(),
        'gamma': GAMMA,
        'sigma': SIGMA.tolist(),
        'tFinal': YEARS,
        'dilemmas': [{'name': i, 'phi': row, 'b': 0} for i, row in zip(dilemma_ids, phi.tolist())],
    }
    with open(DATA / 'cora_params.json', 'w') as f:
        json.dump(params, f)


def main():
    assert np.all((A_CENTRE + A_RADIUS).sum(axis=1) < 1), 'susceptibility must stay below 1'
    DATA.mkdir(exist_ok=True)

    # 1-2. Features and present-day preferences
    X, votes = load_votes()
    phi, dilemmas = load_dilemmas()
    dilemma_ids = [r['id'] for r in dilemmas]
    clusters = fit_clusters(X, votes, phi, dilemmas)
    with open(DATA / 'preferences.json', 'w') as f:
        json.dump({'dimensions': DIMENSIONS, 'clusters': clusters,
                   'dilemma_ids': dilemma_ids, 'phi': phi.tolist()}, f)

    theta0 = np.array([clusters[c]['theta'] for c in CLUSTERS])
    se = np.array([clusters[c]['se'] for c in CLUSTERS])
    print(f'{len(votes)} scenarios, {len(dilemmas)} dilemmas\n')
    print('Preference weights (log-odds of swerving per unit of phi):')
    print(pd.DataFrame(theta0.T, index=DIMENSIONS, columns=CLUSTERS).round(3).to_string())
    print('Majority accuracy on dilemmas.jsonl:', {c: round(clusters[c]['accuracy'], 3) for c in CLUSTERS})

    # 3. Drift
    rng = np.random.RandomState(SEED)
    thetas = [simulate(rng, scenario, theta0, se) for scenario in SCENARIOS]
    p = np.stack([swerve_share(th, phi) for th in thetas])   # (scenario, run, year, N, K)

    trajectories_table(thetas).to_csv(DATA / 'trajectories.csv', index=False, float_format='%.6g')
    np.savez_compressed(DATA / 'choices.npz', p_swerve=p.astype('float16'), choice=(p > 0.5).astype('int8'),
                        scenarios=SCENARIOS, clusters=CLUSTERS, dilemma_ids=dilemma_ids)
    save_cora_params(dilemma_ids, phi, theta0, se)

    meta = {'seed': SEED, 'runs': RUNS, 'years': YEARS, 'steps_per_year': STEPS_PER_YEAR,
            'scenarios': SCENARIOS, 'clusters': CLUSTERS, 'dimensions': DIMENSIONS,
            'A_CENTRE': A_CENTRE.tolist(), 'A_RADIUS': A_RADIUS.tolist(), 'GAMMA': GAMMA,
            'SIGMA': SIGMA.tolist(), 'choices_shape': ['scenario', 'run', 'year', 'cluster', 'dilemma']}
    with open(DATA / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{len(SCENARIOS)} scenarios x {RUNS} runs x {YEARS + 1} years x {len(CLUSTERS)} clusters '
          f'x {phi.shape[0]} dilemmas written to {DATA}/')


if __name__ == '__main__':
    main()
