"""Build the Moral Machine dilemma dataset from the vote counts.

    python dataset/build.py

Reads dataset/votes.csv.gz (written by dataset/moral_machine.py) and writes
data/dilemmas.jsonl: one row per dilemma, with its text description, the majority choice
of each cultural cluster, and that choice as an LTL formula saying who is protected.
"""

import json
import random
from pathlib import Path

import pandas as pd

from ltl import ap, op, to_canonical, to_raw
from moral_machine import CLUSTERS, decode_side

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'

MIN_VOTES = 100    # votes a dilemma needs in every cluster to be kept
TEST_FRAC = 0.2
SPLIT_SEED = 0

# Character type -> (singular, plural), in the order they are listed in a description
NAMES = {
    'Stroller': ('baby in a stroller', 'babies in strollers'),
    'Boy': ('boy', 'boys'),
    'Girl': ('girl', 'girls'),
    'Pregnant': ('pregnant woman', 'pregnant women'),
    'Man': ('man', 'men'),
    'Woman': ('woman', 'women'),
    'MaleAthlete': ('male athlete', 'male athletes'),
    'FemaleAthlete': ('female athlete', 'female athletes'),
    'MaleExecutive': ('male executive', 'male executives'),
    'FemaleExecutive': ('female executive', 'female executives'),
    'MaleDoctor': ('male doctor', 'male doctors'),
    'FemaleDoctor': ('female doctor', 'female doctors'),
    'LargeMan': ('large man', 'large men'),
    'LargeWoman': ('large woman', 'large women'),
    'OldMan': ('elderly man', 'elderly men'),
    'OldWoman': ('elderly woman', 'elderly women'),
    'Homeless': ('homeless person', 'homeless people'),
    'Criminal': ('criminal', 'criminals'),
    'Dog': ('dog', 'dogs'),
    'Cat': ('cat', 'cats'),
}
NUMBERS = ['no', 'one', 'two', 'three', 'four', 'five']
SIGNALS = {0: '', 1: ', who are crossing on a green signal', 2: ', who are crossing on a red signal'}


def describe_characters(counts):
    parts = [f'{NUMBERS[counts[c]]} {NAMES[c][counts[c] > 1]}' for c in NAMES if c in counts]
    return parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


def describe_side(passengers, signal, counts, where):
    """What happens on one side of the dilemma. `where` is 'ahead' or 'in the other lane'."""
    characters = describe_characters(counts)
    if passengers:
        return f'it will crash into a barrier {where}, killing its passengers: {characters}'
    return f'it will drive through the pedestrian crossing {where}, killing {characters}{SIGNALS[signal]}'


def describe(stay, swerve):
    return ('A self-driving car has sudden brake failure. '
            f'If it stays on course, {describe_side(*stay, "ahead")}. '
            f'If it swerves, {describe_side(*swerve, "in the other lane")}.')


def group(side, where):
    """The atomic proposition for killing one side's characters."""
    passengers = side[0]
    return 'kill passengers' if passengers else f'kill pedestrians {where}'


def protected_formula(choice, stay, swerve):
    """The choice as an LTL constraint: never kill the side that is spared.

    Staying kills the characters ahead, so it protects those in the other lane, and vice versa.
    """
    spared = group(swerve, 'in other lane') if choice == 'stay' else group(stay, 'ahead')
    return op('globally', op('not', ap(spared)))


def main():
    votes = pd.read_csv(ROOT / 'votes.csv.gz')
    enough = (votes[[f'votes_{c}' for c in CLUSTERS]] >= MIN_VOTES).all(axis=1)
    votes = votes[enough].sort_values(['stay', 'swerve']).reset_index(drop=True)

    ids = list(range(len(votes)))
    random.Random(SPLIT_SEED).shuffle(ids)
    test = set(ids[:round(TEST_FRAC * len(ids))])

    rows = []
    for i, v in votes.iterrows():
        stay, swerve = decode_side(v['stay']), decode_side(v['swerve'])
        row = {'id': f'D{i:05d}', 'natural': describe(stay, swerve), 'scenario_type': v['scenario_type'],
               'split': 'test' if i in test else 'train',
               'stay': {'passengers': stay[0], 'signal': stay[1], 'characters': stay[2]},
               'swerve': {'passengers': swerve[0], 'signal': swerve[1], 'characters': swerve[2]}}
        for c in CLUSTERS:
            n, s = int(v[f'votes_{c}']), int(v[f'swerve_{c}'])
            choice = 'swerve' if 2 * s > n else 'stay' if 2 * s < n else None  # ties get no label
            formula = protected_formula(choice, stay, swerve) if choice else None
            row[c] = {'votes': n, 'swerve_share': round(s / n, 4) if n else None, 'choice': choice,
                      'formula': to_canonical(formula) if formula else None,
                      'raw_ltl': to_raw(formula) if formula else None}
        rows.append(row)

    DATA.mkdir(exist_ok=True)
    with open(DATA / 'dilemmas.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    print(f'{len(rows)} dilemmas with at least {MIN_VOTES} votes in every cluster '
          f'({len(rows) - len(test)} train, {len(test)} test)')
    for c in CLUSTERS:
        choices = pd.Series([r[c]['choice'] for r in rows])
        print(f'  {c:9s} swerve {int((choices == "swerve").sum()):5d}  stay {int((choices == "stay").sum()):5d}  '
              f'tie {int(choices.isna().sum()):3d}')
    labelled = [r for r in rows if all(r[c]['choice'] for c in CLUSTERS)]
    disagree = sum(len({r[c]['choice'] for c in CLUSTERS}) > 1 for r in labelled)
    print(f'  clusters disagree on {disagree} of {len(labelled)} dilemmas')


if __name__ == '__main__':
    main()
