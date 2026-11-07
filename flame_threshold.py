"""Otsu's threshold over a list of scores, as `skimage.filters.threshold_otsu` computes it for a float
array with 256 bins, so FLAME needs no scikit-image (which pulls in image libraries)."""

import numpy as np


def threshold_otsu(scores: np.ndarray, nbins: int = 256) -> float:
    """The score that maximizes the variance between the scores below it and above it."""
    values = np.asarray(scores).reshape(-1)
    if np.all(values == values[0]):
        return values[0]
    edges = np.histogram_bin_edges(values, bins=nbins, range=(values.min(), values.max()))
    counts, edges = np.histogram(values, bins=edges)
    counts = counts.astype("float32", copy=False)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * centers) / weight1
    mean2 = (np.cumsum((counts * centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return centers[np.argmax(variance12)]
