"""Hyperrectangles in embedding space, following ANTONIO.

A hyperrectangle is the smallest box that contains a sentence's embedding and the
embeddings of its perturbations. Each box has shape [dimensions, 2]: a [lower, upper]
bound for every input dimension of the network. The box takes the sentence's label: a
network is robust on the box if it gives that label to every point inside it.
"""

import numpy as np

# Perturbations whose full sentence embedding is less similar than this to the original
# are not used for the box (ANTONIO's threshold)
COSINE_THRESHOLD = 0.6


def cosine_similarity(a, b):
    """Row-wise cosine similarity between two arrays of embeddings of the same shape."""
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def build_hyperrectangles(X, X_perturbed, source, cosine, threshold=COSINE_THRESHOLD):
    """One box per sentence around its embedding and its kept perturbations.

    Args:
        X: Reduced embeddings of the sentences, shape (n, d).
        X_perturbed: Reduced embeddings of the perturbations, shape (m, d).
        source: For every perturbation, the index of the sentence it came from.
        cosine: For every perturbation, its full-embedding cosine similarity to the original.
        threshold: Minimum cosine similarity for a perturbation to be used.

    Returns:
        boxes: Shape (k, d, 2), one box for every sentence with at least one kept perturbation.
        box_source: For every box, the index of the sentence it was built from.
        n_points: For every box, the number of points it was built from.
    """
    boxes, box_source, n_points = [], [], []
    for i in range(len(X)):
        kept = X_perturbed[(source == i) & (cosine > threshold)]
        if len(kept) == 0:
            continue
        points = np.vstack([X[i:i + 1], kept])
        boxes.append(np.stack([points.min(axis=0), points.max(axis=0)], axis=1))
        box_source.append(i)
        n_points.append(len(points))
    return np.array(boxes), np.array(box_source), np.array(n_points)


def contained(points, boxes):
    """Boolean matrix (n_points, n_boxes): whether each point lies inside each box."""
    lower = boxes[None, :, :, 0]
    upper = boxes[None, :, :, 1]
    p = points[:, None, :]
    return np.all((p >= lower) & (p <= upper), axis=2)


def conflicting_boxes(boxes, labels):
    """For every box, whether it intersects a box with a different label.

    Two boxes intersect if their [lower, upper] ranges overlap in every dimension. A point in
    the intersection of boxes with different labels cannot get both labels, so no network can
    be robust on both boxes.
    """
    lower, upper = boxes[:, :, 0], boxes[:, :, 1]
    conflict = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        intersects = np.all((lower[i] <= upper) & (lower <= upper[i]), axis=1)
        conflict[i] = np.any(intersects & (labels != labels[i]))
    return conflict
