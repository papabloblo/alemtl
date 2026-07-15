"""ALE and curve-similarity utilities."""

from .ale import Intervals, MultiTaskALE
from .similarity import (
    MultitaskSimilarity,
    discrete_frechet_distance_vectorized,
    frechet_distance_vectorized,
    frechet_similarity_vectorized,
)

__all__ = [
    "Intervals",
    "MultiTaskALE",
    "MultitaskSimilarity",
    "discrete_frechet_distance_vectorized",
    "frechet_distance_vectorized",
    "frechet_similarity_vectorized",
]
