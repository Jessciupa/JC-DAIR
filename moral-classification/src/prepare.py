"""Precompute everything the notebooks read from embeddings/.

    python src/prepare.py

1. Embed the train and test dilemmas with all-MiniLM-L6-v2.
2. Fit PCA on the training embeddings and scale to [0, 1] (30 dimensions).
3. Create character perturbations of the train and test dilemmas, and embed them.
4. Build one hyperrectangle per training dilemma from its perturbations whose
   full-embedding cosine similarity to the original is above 0.6.

The labels differ between cultural clusters, so every file keeps one label per cluster
(y_western, y_eastern, y_southern; -1 where the cluster's vote was tied).

Needs sentence-transformers (requirements-embed.txt). The results are committed, so the
notebooks run without it.
"""

import numpy as np

from data import CLUSTERS, load_split
from embed import EMBEDDINGS, encode, fit_reduction, reduce
from hyperrectangles import build_hyperrectangles, cosine_similarity
from perturbations import character_perturbations

N_PER_OPERATION = 1  # as in ANTONIO: one perturbation per operation and sentence
SEED = 42


def perturb_and_embed(sentences, E, reduction, seed):
    """Character perturbations of the sentences, their reduced embeddings and cosine similarities."""
    perturbed, source, operation = character_perturbations(sentences, N_PER_OPERATION, seed)
    E_perturbed = encode(perturbed)
    cosine = cosine_similarity(E[source], E_perturbed)
    return dict(X=reduce(E_perturbed, reduction), sentence=perturbed, source=source,
                operation=operation, cosine=cosine.astype('float32'))


def main():
    EMBEDDINGS.mkdir(exist_ok=True)
    train, test = load_split('train'), load_split('test')

    # 1-2. Embed and reduce; PCA and scaling are fitted on the training dilemmas only
    E_train, E_test = encode(train['sentence']), encode(test['sentence'])
    reduction = fit_reduction(E_train)
    np.savez(EMBEDDINGS / 'reduction.npz', **reduction)

    X_train = reduce(E_train, reduction)
    np.savez_compressed(EMBEDDINGS / 'train.npz', X=X_train, **train)
    np.savez_compressed(EMBEDDINGS / 'test.npz', X=reduce(E_test, reduction), **test)

    # 3. Character perturbations
    train_p = perturb_and_embed(train['sentence'], E_train, reduction, SEED)
    test_p = perturb_and_embed(test['sentence'], E_test, reduction, SEED + 1)
    np.savez_compressed(EMBEDDINGS / 'train_perturbed.npz', **train_p)
    np.savez_compressed(EMBEDDINGS / 'test_perturbed.npz', **test_p)

    # 4. Hyperrectangles, with the label of the dilemma they were built from in every cluster
    boxes, box_source, n_points = build_hyperrectangles(X_train, train_p['X'], train_p['source'], train_p['cosine'])
    labels = {f'label_{c}': train[f'y_{c}'][box_source] for c in CLUSTERS}
    np.savez_compressed(EMBEDDINGS / 'hyperrectangles.npz', boxes=boxes, source=box_source, n_points=n_points, **labels)

    kept = train_p['cosine'] > 0.6
    print(f'train {len(X_train)}, test {len(test["sentence"])} dilemmas')
    print(f'train perturbations {len(kept)} ({kept.mean():.1%} kept by the cosine filter), '
          f'test perturbations {len(test_p["source"])}')
    print(f'hyperrectangles {boxes.shape}, points per box: median {int(np.median(n_points))}')


if __name__ == '__main__':
    main()
