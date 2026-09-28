"""Count Moral Machine votes per distinct dilemma and cultural cluster.

    python dataset/moral_machine.py path/to/SharedResponses.csv

Reads the raw responses of the Moral Machine experiment (Awad et al., 2018), available at
https://osf.io/3hvt2/ (Datasets / Moral Machine Data / SharedResponses.csv.tar.gz, 3.2 GB),
and writes dataset/votes.csv.gz: for every distinct dilemma that users from all three
cultural clusters answered, how many users in each cluster chose to stay and to swerve.

In the raw data each dilemma (scenario) is two rows sharing a ResponseID, one per outcome:

    Intervention    0: these characters die if the car stays, 1: if it swerves
    Barrier         1: these characters are the car's passengers, 0: they are pedestrians
    CrossingSignal  0: no traffic light, 1: crossing on green, 2: crossing on red
    Saved           1: the user chose to save these characters
    Man ... Cat     how many characters of each of the 20 types are in this outcome

A dilemma is identified by both outcomes: who is on each side, whether they are
passengers or pedestrians, and the crossing signal. votes.csv.gz stores each outcome as one
integer (see encode_side), in the columns 'stay' and 'swerve'.
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# The 20 character types, in the column order of the raw data
CHARACTERS = ['Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman', 'Boy', 'Girl',
              'Homeless', 'LargeWoman', 'LargeMan', 'Criminal', 'MaleExecutive', 'FemaleExecutive',
              'FemaleAthlete', 'MaleAthlete', 'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat']

# Cultural clusters of Awad et al. (2018), from country_cluster_map.csv on the same OSF page
CLUSTERS = ['western', 'eastern', 'southern']

COLUMNS = ['ResponseID', 'Intervention', 'Barrier', 'CrossingSignal', 'Saved', 'UserCountry3',
           'ScenarioType'] + CHARACTERS


def load_cluster_map():
    with open(ROOT / 'country_cluster_map.csv') as f:
        return {row['ISO3']: CLUSTERS[int(row['Cluster'])] for row in csv.DictReader(f)}


def encode_side(rows):
    """Encode one outcome as a single integer: passengers or pedestrians, crossing signal, and
    the 20 character counts (each 0-5) as the digits of a base-6 number."""
    counts = rows[CHARACTERS].to_numpy(dtype='int64')
    code = counts @ (6 ** np.arange(len(CHARACTERS) - 1, -1, -1, dtype='int64'))
    return code + (rows['Barrier'].to_numpy(dtype='int64') * 3 + rows['CrossingSignal'].to_numpy(dtype='int64')) * 6 ** len(CHARACTERS)


def decode_side(code):
    """The inverse of encode_side: (passengers?, crossing signal, {character: count})."""
    structure, code = divmod(int(code), 6 ** len(CHARACTERS))
    barrier, signal = divmod(structure, 3)
    counts = {}
    for c in reversed(CHARACTERS):
        code, n = divmod(code, 6)
        if n:
            counts[c] = n
    return barrier == 1, signal, {c: counts[c] for c in CHARACTERS if c in counts}


def pair_outcomes(rows):
    """Pair the two outcomes of every scenario whose rows are all present.

    Returns one row per scenario: its key, scenario type, the user's cluster, and whether
    the user chose to swerve.
    """
    stay = rows[rows['Intervention'] == 0].drop_duplicates('ResponseID', keep=False).set_index('ResponseID')
    swerve = rows[rows['Intervention'] == 1].drop_duplicates('ResponseID', keep=False).set_index('ResponseID')
    ids = stay.index.intersection(swerve.index)
    stay, swerve = stay.loc[ids], swerve.loc[ids]

    # The user swerves if they saved the characters who would die if the car stayed
    return pd.DataFrame({
        'stay': stay['side'],
        'swerve_side': swerve['side'],
        'scenario_type': stay['ScenarioType'],
        'cluster': stay['cluster'],
        'swerve': stay['Saved'].astype(int),
    })


def main(path, chunksize=2_000_000, n_buckets=64):
    """Count votes in two passes, so memory stays small however the file is ordered.

    The raw file is not sorted by scenario: the two outcomes of a scenario can be far
    apart. Pass 1 streams the file and writes every outcome, reduced to a few columns, to
    one of n_buckets files chosen by its ResponseID, so both outcomes of a scenario land in
    the same bucket. Pass 2 pairs and counts one bucket at a time.
    """
    cluster_map = load_cluster_map()
    # The buckets hold a reduced copy of the whole file, so they go next to it, not in the repository
    with tempfile.TemporaryDirectory(dir=Path(path).resolve().parent) as tmp:
        buckets = [Path(tmp) / f'bucket_{b:02d}.csv' for b in range(n_buckets)]

        # Pass 1: reduce every outcome and write it to its bucket
        n_rows = 0
        for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=chunksize, low_memory=False):
            n_rows += len(chunk)
            # A few chunks hold counts as text such as '0.0', so parse every number explicitly
            numbers = ['Intervention', 'Saved', 'Barrier', 'CrossingSignal'] + CHARACTERS
            chunk[numbers] = chunk[numbers].apply(pd.to_numeric, errors='coerce')
            chunk = chunk.dropna(subset=['ResponseID'] + numbers)
            chunk = chunk.assign(cluster=chunk['UserCountry3'].map(cluster_map)).dropna(subset=['cluster'])
            reduced = pd.DataFrame({'ResponseID': chunk['ResponseID'], 'Intervention': chunk['Intervention'].astype(int),
                                    'side': encode_side(chunk), 'Saved': chunk['Saved'].astype(int),
                                    'ScenarioType': chunk['ScenarioType'], 'cluster': chunk['cluster']})
            bucket = pd.util.hash_pandas_object(reduced['ResponseID'], index=False) % n_buckets
            for b, rows in reduced.groupby(bucket.values):
                rows.to_csv(buckets[b], mode='a', header=not buckets[b].exists(), index=False)
            print(f'pass 1: {n_rows:,} rows read', flush=True)

        # Pass 2: pair the outcomes in each bucket and count the votes
        counts = []
        for bucket_path in buckets:
            if not bucket_path.exists():  # no outcome was assigned to this bucket
                continue
            scenarios = pair_outcomes(pd.read_csv(bucket_path))
            counts.append(scenarios.groupby(['stay', 'swerve_side', 'scenario_type', 'cluster'])['swerve'].agg(['size', 'sum']))
        print(f'pass 2: {sum(int(c["size"].sum()) for c in counts):,} scenarios paired', flush=True)

    votes = pd.concat(counts).groupby(level=[0, 1, 2, 3]).sum().unstack('cluster', fill_value=0)
    votes.columns = [f'{"votes" if stat == "size" else "swerve"}_{cluster}' for stat, cluster in votes.columns]
    columns = [f'{stat}_{c}' for stat in ['votes', 'swerve'] for c in CLUSTERS]
    votes = votes.reindex(columns=columns, fill_value=0).reset_index().rename(columns={'swerve_side': 'swerve'})
    save_votes(votes)


def save_votes(votes):
    """Merge scenario types and keep the dilemmas every cluster voted on.

    Early sessions had their scenario type inferred afterwards (see the Moral Machine
    ReadMe), so the same dilemma can appear under two types: their votes are added up and
    the dilemma keeps its most common type. Only dilemmas with at least one vote from every
    cluster are kept, which keeps the file small enough to commit.
    """
    columns = [f'{stat}_{c}' for stat in ['votes', 'swerve'] for c in CLUSTERS]
    total = votes[[f'votes_{c}' for c in CLUSTERS]].sum(axis=1)
    scenario_type = votes.assign(total=total).sort_values('total').groupby(['stay', 'swerve'])['scenario_type'].last()
    merged = votes.groupby(['stay', 'swerve'])[columns].sum().join(scenario_type).reset_index()
    print(f'{len(merged):,} distinct dilemmas')

    merged = merged[(merged[[f'votes_{c}' for c in CLUSTERS]] > 0).all(axis=1)]
    merged[['stay', 'swerve', 'scenario_type'] + columns].to_csv(ROOT / 'votes.csv.gz', index=False)
    print(f'{len(merged):,} dilemmas with votes from every cluster written to dataset/votes.csv.gz')


if __name__ == '__main__':
    main(sys.argv[1])
