"""Sentence embeddings, reduced to a few dimensions and scaled to [0, 1].

    sentence -> all-MiniLM-L6-v2 (384 dimensions) -> PCA (30 dimensions) -> scale to [0, 1]

The network only sees the 30 scaled dimensions, which keeps it small enough for Vehicle
and Marabou. This module needs sentence-transformers (requirements-embed.txt). The
notebooks do not: they read the precomputed embeddings in embeddings/.
"""

from pathlib import Path

import numpy as np

EMBEDDINGS = Path(__file__).resolve().parent.parent / 'embeddings'
ENCODER = 'all-MiniLM-L6-v2'
N_COMPONENTS = 30

_model = None


def encode(sentences):
    """Full 384-dimensional sentence embeddings."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(ENCODER)
    return _model.encode(list(sentences), batch_size=128, convert_to_numpy=True)


def fit_reduction(E, n_components=N_COMPONENTS):
    """Fit PCA on the training embeddings, and the [0, 1] scaling of its components."""
    mean = E.mean(axis=0)
    _, _, Vt = np.linalg.svd(E - mean, full_matrices=False)
    components = Vt[:n_components]
    Z = (E - mean) @ components.T
    return {'mean': mean, 'components': components, 'low': Z.min(axis=0), 'high': Z.max(axis=0)}


def reduce(E, reduction):
    """Project full embeddings onto the PCA components and scale them to [0, 1].

    Points outside the range seen in training are clipped to [0, 1], the input range of
    the network.
    """
    Z = (E - reduction['mean']) @ reduction['components'].T
    return np.clip((Z - reduction['low']) / (reduction['high'] - reduction['low']), 0, 1).astype('float32')


def load_reduction():
    with np.load(EMBEDDINGS / 'reduction.npz') as f:
        return dict(f)


def embed(sentences):
    """Sentences -> the network's 30 scaled input dimensions, e.g. for LIME."""
    return reduce(encode(sentences), load_reduction())
