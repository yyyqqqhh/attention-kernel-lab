"""Float64 CPU models of dense and streaming attention; no framework required."""

import math
from numbers import Real


def _matrix(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence of rows")
    result = []
    width = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"{name} rows must be nonempty sequences")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"{name} is ragged")
        if any(isinstance(x, bool) or not isinstance(x, Real) for x in row):
            raise TypeError(f"{name} entries must be real numbers, excluding bool")
        converted = [float(x) for x in row]
        if not all(math.isfinite(x) for x in converted):
            raise ValueError(f"{name} entries must be finite")
        result.append(converted)
    return result


def _inputs(q, k, v, causal, scale):
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    q, k, v = (_matrix(x, name) for x, name in ((q, "q"), (k, "k"), (v, "v")))
    if len(k) != len(v):
        raise ValueError("k and v must have the same number of rows")
    d = len(q[0])
    if len(k[0]) != d or len(v[0]) != d:
        raise ValueError("q, k and v must have the same head dimension")
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a real number")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return q, k, v, scale


def _score(q, k, scale):
    value = math.fsum(a * b for a, b in zip(q, k)) * scale
    if not math.isfinite(value):
        raise ValueError("attention score overflowed; reduce input magnitude or scale")
    return value


def dense_attention(q, k, v, *, causal=False, scale=None):
    """Evaluate one [Nq,D] head using a materialized row softmax.

    Causality is upper-left aligned: key j is visible to query i iff j <= i.
    The result is nested Python lists; this is a semantic oracle, not a CPU kernel.
    """
    q, k, v, scale = _inputs(q, k, v, causal, scale)
    result = []
    for i, query in enumerate(q):
        end = min(i + 1, len(k)) if causal else len(k)
        scores = [_score(query, key, scale) for key in k[:end]]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = math.fsum(weights)
        result.append([
            math.fsum(weight * row[d] for weight, row in zip(weights, v[:end]))
            / denominator
            for d in range(len(query))
        ])
    return result


def online_attention(q, k, v, *, causal=False, scale=None, block_size=32):
    """Stream K/V tiles, retaining a maximum, normalizer and output numerator.

    Only the current score tile exists. Rescaling the previous numerator whenever
    the running maximum changes keeps it in the same units as the normalizer.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    q, k, v, scale = _inputs(q, k, v, causal, scale)
    result = []
    for i, query in enumerate(q):
        end = min(i + 1, len(k)) if causal else len(k)
        maximum, denominator = -math.inf, 0.0
        numerator = [0.0] * len(query)
        for start in range(0, end, block_size):
            stop = min(start + block_size, end)
            scores = [_score(query, k[j], scale) for j in range(start, stop)]
            next_maximum = max(maximum, max(scores))
            correction = math.exp(maximum - next_maximum)
            weights = [math.exp(score - next_maximum) for score in scores]
            denominator = correction * denominator + math.fsum(weights)
            numerator = [
                correction * numerator[d]
                + math.fsum(weight * v[j][d] for weight, j in zip(weights, range(start, stop)))
                for d in range(len(query))
            ]
            maximum = next_maximum
        result.append([value / denominator for value in numerator])
    return result
