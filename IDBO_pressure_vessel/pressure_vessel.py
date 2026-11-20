"""Mixed discrete-continuous vessel model with evaluation-time decoding."""
import numpy as np

LOWER = np.array([0., 0., 10., 10.])
UPPER = np.array([100., 100., 200., 240.])
SCALES = np.array([100., 100., 1296000., 240.])
THICKNESS_STEP = 0.0625


def decode(x):
    """Decode a continuous search position without modifying the population.

    The two thickness components are projected to the nearest 0.0625 grid point;
    ties follow NumPy's nearest-even rule. Initialization candidates, opposition
    candidates and every iteration candidate pass through this same mapping.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (4,) or not np.all(np.isfinite(x)) or np.any(x < LOWER) or np.any(x > UPPER):
        raise ValueError('Expected four finite, in-bounds encoded variables')
    physical = x.copy()
    physical[:2] = np.rint(physical[:2] / THICKNESS_STEP) * THICKNESS_STEP
    return physical


def cost(x):
    ts, th, r, length = np.asarray(x, dtype=float)
    return float(.6224*ts*r*length + 1.7781*th*r*r + 3.1661*ts*ts*length + 19.84*ts*ts*r)


def constraints(x):
    ts, th, r, length = np.asarray(x, dtype=float)
    return np.array([.0193*r - ts, .00954*r - th,
                     1296000. - np.pi*r*r*length - (4./3.)*np.pi*r**3, length - 240.])


# All coefficients of the cost function are positive, so this value bounds the
# cost over the whole search box and can serve as the normalization constant.
COST_UPPER = cost(UPPER)  # 57378560.0


def feasible(x):
    """Range, thickness-grid and constraint check on the decoded design."""
    x = np.asarray(x, dtype=float)
    return bool(x.shape == (4,) and np.all(np.isfinite(x)) and
                np.all(x >= LOWER) and np.all(x <= UPPER) and
                np.all(x[:2] / THICKNESS_STEP == np.rint(x[:2] / THICKNESS_STEP)) and
                np.all(constraints(x) <= 0.))


class VesselObjective:
    """Decode a search position, then score the resulting physical design.

    Feasible designs score cost/COST_UPPER within [0,1]; infeasible designs score
    2 plus the sum of positive normalized constraint residuals. This score drives
    every fitness comparison in IDBO, including the mean-score gate of the Levy
    perturbation, while reported costs are the raw objective values.
    """

    def __call__(self, x):
        x = decode(x)
        residuals = constraints(x)
        if np.all(residuals <= 0.):
            return cost(x) / COST_UPPER
        return 2. + float(np.maximum(residuals / SCALES, 0.).sum())
