"""Compare two curve metrics on the nonlinear figure checkpoint, without retraining.

First run reproduce_nonlinear_figures.py, then:
    python examples/custom_similarity.py --checkpoint results/nonlinear_figures/checkpoint.pt

The custom symmetric nearest-point score ignores curve traversal order, unlike
Frechet distance. This is an extension demonstration, not a performance study.
"""
import argparse
from pathlib import Path

import torch

from reproduce_revised_table import Batches, build_model, initialize, prepare_panel
from alemtl.similarity import MultiTaskALE, MultitaskSimilarity


def nearest_point_similarity(curve_a, curve_b):
    """Map symmetric nearest-point distance to similarity for (batch, points, 2).

    Both standardized feature coordinates and normalized ALE values contribute.
    Return one score per curve pair; larger values mean greater similarity.
    """
    distances = torch.cdist(curve_a, curve_b)
    forward = distances.amin(dim=-1).mean(dim=-1)
    backward = distances.amin(dim=-2).mean(dim=-1)
    return torch.exp(-0.5 * (forward + backward))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    initialize()
    panel = prepare_panel('clustered_nonlinear', 1101)
    model = build_model('ale_frechet', 6, 5)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    train = panel.splits['train']
    ale = MultiTaskALE(model, Batches(train, 64, 1301), n_tasks=6,
                       num_intervals=8, n_guess=train[0].size(1))
    ale.update()
    comparisons = {
        'Frechet': MultitaskSimilarity(ale, std=1.0),
        'Nearest-point': MultitaskSimilarity(
            ale, similarity_func=nearest_point_similarity, std=1.0),
    }
    peers = {}
    for name, similarity in comparisons.items():
        similarity.compute()
        _, pairs = similarity.tasks_groups()
        peers[name] = pairs[:, 1]
        print(f'{name} peers (one-based): {(pairs[:, 1] + 1).tolist()}')
        print(f'{name} task 1--2 feature scores: '
              f'{similarity.similarity_tasks_features[0, 1].tolist()}')
    changed = (peers['Frechet'] != peers['Nearest-point']).nonzero().flatten() + 1
    print(f'Tasks whose nearest peer changed: {changed.tolist()}')
    print('Scores from different metrics are not calibrated to the same scale.')


if __name__ == '__main__':
    main()
