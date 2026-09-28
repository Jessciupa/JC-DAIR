"""Verify the classifier on many hyperrectangles, one `vehicle verify` call per box.

    python src/verify_boxes.py models/western_base.onnx                        # every box, western labels
    python src/verify_boxes.py models/western_base.onnx --limit 50             # the first 50 boxes
    python src/verify_boxes.py models/western_base.onnx --sample 100           # a random sample of 100 boxes
    python src/verify_boxes.py models/eastern_base.onnx --cluster eastern      # eastern labels
    python src/verify_boxes.py --export-box 7 box.idx --cluster western        # write box 7, print its label

Each box is labelled with the choice the cluster made for its dilemma; boxes whose
dilemma the cluster tied on are skipped.

Vehicle 0.28 cannot compile one property over all boxes (see spec.vcl), so every box
is written to its own IDX file and verified separately, as in DAIR-course-NLP.
"""

import argparse
import os
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent.parent
SPEC = FOLDER / 'spec.vcl'
HYPERRECTANGLES = FOLDER / 'embeddings' / 'hyperrectangles.npz'
TIMEOUT = 60  # seconds per box; a few boxes are much harder for Marabou than the rest


def write_idx(path, array):
    """Write an array in the IDX format Vehicle reads datasets from (big-endian float64)."""
    array = np.asarray(array, dtype='>f8')
    with open(path, 'wb') as f:
        f.write(bytes([0, 0, 0x0E, array.ndim]))
        f.write(struct.pack('>' + 'i' * array.ndim, *array.shape))
        f.write(array.tobytes())


def load_hyperrectangles(cluster):
    """The boxes and the cluster's choice for each (0 stay, 1 swerve, -1 tie)."""
    with np.load(HYPERRECTANGLES) as f:
        return f['boxes'], f[f'label_{cluster}']


def verify_box(network, box, label, idx_path, timeout=TIMEOUT):
    """Run `vehicle verify` on one box.

    Returns 'verified', 'counterexample', 'timeout' (Marabou did not finish within `timeout`
    seconds) or the error output.
    """
    write_idx(idx_path, box)
    # vehicle and Marabou are installed next to this Python
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH'])
    command = ['vehicle', 'verify', '-v', 'Marabou', '-s', str(SPEC), '-n', f'classifier:{network}',
               '-d', f'hyperrectangle:{idx_path}', '-p', f'label:{label}']
    # A new session, so that on a timeout Marabou is stopped together with vehicle
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return 'timeout'
    if 'proved no counterexample exists' in stdout:
        return 'verified'
    if 'found a counterexample' in stdout:
        return 'counterexample'
    return (stdout + stderr).strip()[-500:]


def verify_boxes(network, cluster='western', limit=None, sample=None, seed=0, timeout=TIMEOUT, progress=False):
    """Verify the labelled boxes: all of them, the first `limit`, or a random `sample` of them.

    Returns the indices of the boxes checked and one outcome per box.
    """
    boxes, labels = load_hyperrectangles(cluster)
    indices = np.where(labels >= 0)[0]
    if sample is not None:
        indices = np.sort(np.random.default_rng(seed).choice(indices, size=min(sample, len(indices)), replace=False))
    indices = indices[:limit]
    outcomes = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, i in enumerate(indices):
            outcomes.append(verify_box(network, boxes[i], labels[i], Path(tmp) / 'box.idx', timeout))
            if progress and ((k + 1) % 100 == 0 or k + 1 == len(indices)):
                print(f'[{k + 1}/{len(indices)}] verified {outcomes.count("verified")}, '
                      f'counterexample {outcomes.count("counterexample")}, timeout {outcomes.count("timeout")}', flush=True)
    return indices, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('network', nargs='?', help='ONNX network to verify')
    parser.add_argument('--cluster', default='western', choices=['western', 'eastern', 'southern'],
                        help="whose choices label the boxes (default: western)")
    parser.add_argument('--limit', type=int, default=None, help='only verify the first LIMIT boxes')
    parser.add_argument('--sample', type=int, default=None, help='verify a random sample of SAMPLE boxes')
    parser.add_argument('--seed', type=int, default=0, help='random seed for --sample (default: 0)')
    parser.add_argument('--timeout', type=float, default=TIMEOUT,
                        help=f'seconds Marabou may spend on one box before it counts as a timeout (default: {TIMEOUT})')
    parser.add_argument('--export-box', nargs=2, metavar=('INDEX', 'PATH'),
                        help='write one box to an IDX file and print its label, then exit')
    args = parser.parse_args()

    if args.export_box:
        boxes, labels = load_hyperrectangles(args.cluster)
        i = int(args.export_box[0])
        write_idx(args.export_box[1], boxes[i])
        print(labels[i])
        return
    if args.network is None:
        parser.error('give a network to verify, or --export-box')

    n_boxes = args.sample or args.limit
    indices, outcomes = verify_boxes(args.network, args.cluster, args.limit, args.sample, args.seed, args.timeout,
                                     progress=n_boxes is None or n_boxes > 100)
    errors = [(i, o) for i, o in zip(indices, outcomes) if o not in ('verified', 'counterexample', 'timeout')]
    print(f'Boxes checked: {len(outcomes)}')
    print(f'Verified (no counterexample): {outcomes.count("verified")}')
    print(f'Counterexample found: {outcomes.count("counterexample")}')
    print(f'Timeout (no answer within {args.timeout:g}s): {outcomes.count("timeout")}')
    print(f'Errors: {len(errors)}')
    for i, error in errors[:5]:
        print(f'  box {i}: {error}')


if __name__ == '__main__':
    main()
