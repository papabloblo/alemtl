"""Check the documented custom metric's geometric contract."""
import importlib.util
from pathlib import Path
import sys

import torch


def test_nearest_point_metric_is_symmetric_and_ignores_traversal_order():
    examples = Path(__file__).resolve().parents[1] / 'examples'
    sys.path.insert(0, str(examples))
    try:
        spec = importlib.util.spec_from_file_location('custom_similarity', examples / 'custom_similarity.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    score = module.nearest_point_similarity
    a = torch.tensor([[[0., 0.], [1., 1.], [2., 0.]]])
    b = a + torch.tensor([0., 1.])
    assert score(a, b).shape == (1,)
    torch.testing.assert_close(score(a, b), score(b, a))
    torch.testing.assert_close(score(a, a.flip(1)), torch.ones(1))
    assert 0 < score(a, b).item() < 1
