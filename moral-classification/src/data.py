"""Load the Moral Machine dilemmas and each cultural cluster's choice.

Each dilemma is a text description of a self-driving car with brake failure that must
either stay on course or swerve, killing different characters either way. Its label is
the choice most users in a cultural cluster made (Awad et al., 2018).
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'

CLASSES = ['stay', 'swerve']
CLUSTERS = ['western', 'eastern', 'southern']


def load_dilemmas():
    with open(DATA / 'dilemmas.jsonl') as f:
        return [json.loads(line) for line in f]


def load_split(split):
    """Dilemmas of the 'train' or 'test' split.

    Returns a dict with the sentences, dilemma ids and scenario types, and for every
    cluster its labels (0 stay, 1 swerve, -1 for a tie), vote counts and swerve shares.
    """
    rows = [r for r in load_dilemmas() if r['split'] == split]
    data = {'sentence': np.array([r['natural'] for r in rows]),
            'id': np.array([r['id'] for r in rows]),
            'scenario_type': np.array([r['scenario_type'] for r in rows])}
    for c in CLUSTERS:
        data[f'y_{c}'] = np.array([CLASSES.index(r[c]['choice']) if r[c]['choice'] else -1 for r in rows])
        data[f'votes_{c}'] = np.array([r[c]['votes'] for r in rows])
        data[f'swerve_share_{c}'] = np.array([r[c]['swerve_share'] for r in rows], dtype='float32')
    return data
