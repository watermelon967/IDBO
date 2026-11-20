"""IDBO: elite-opposition initialization with staged Levy and Gaussian perturbations."""
from dataclasses import dataclass, asdict
import math
from decimal import Decimal, ROUND_FLOOR
import numpy as np


@dataclass(frozen=True)
class Config:
    population: int = 30
    iterations: int = 300
    expansion: int = 3
    eta: float = 0.2
    narrow_fraction: float = 0.7
    roles: tuple = (0.20, 0.13, 0.25, 0.42)
    beta: float = 1.5
    epsilon: float = 1e-12
    alpha: tuple = (0.06, 0.03, 0.012)
    rho: float = 1.0
    delta: tuple = (0.08, 0.04, 0.01)
    lam: float = 3.0
    rolling_probability: float = 0.9
    positive_sign_probability: float = 0.9
    rolling_k: float = 0.1
    rolling_b: float = 0.3
    stealing_s: float = 0.5


def role_counts(n, ratios):
    """Split n individuals into four roles by floor plus largest remainders."""
    r = np.asarray(ratios, dtype=float)
    if r.shape != (4,) or np.any(r < 0) or not np.isclose(r.sum(), 1):
        raise ValueError('Four nonnegative role proportions must sum to one')
    raw = n * r
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts), kind='stable')
    counts[order[:n - counts.sum()]] += 1
    return counts


def stage(t, total):
    """Three equal stages: 0 for t/T <= 1/3, 1 for <= 2/3, 2 otherwise."""
    return 0 if 3 * t <= total else 1 if 3 * t <= 2 * total else 2


class IDBO:
    def __init__(self, obj_func, dim=10, lower=-100, upper=100, seed=2026, config=None):
        self.obj_func, self.dim = obj_func, int(dim)
        self.config = config or Config()
        self.lower = np.broadcast_to(np.asarray(lower, float), (self.dim,)).copy()
        self.upper = np.broadcast_to(np.asarray(upper, float), (self.dim,)).copy()
        c = self.config
        if self.dim < 1 or not np.all(np.isfinite([self.lower, self.upper])) or np.any(self.lower >= self.upper):
            raise ValueError('Finite lower < upper bounds required')
        if c.population < 4 or c.iterations < 1 or c.expansion < 1 or int(c.expansion) != c.expansion:
            raise ValueError('Invalid population, iterations or expansion')
        if not (0 < c.eta < .5 and 0 <= c.narrow_fraction <= 1 and 0 < c.beta < 2):
            raise ValueError('Invalid initialization or Levy parameters')
        if c.rho <= 0 or c.lam <= 0 or c.epsilon <= 0:
            raise ValueError('Decay exponents and epsilon must be positive')
        if len(c.alpha) != 3 or len(c.delta) != 3 or min(c.alpha + c.delta) < 0:
            raise ValueError('Three nonnegative stage scales required')
        self.counts = role_counts(c.population, c.roles)
        self.seed = seed
        self.evaluations = 0
        self.sigma = (math.gamma(1 + c.beta) * math.sin(math.pi * c.beta / 2) /
                      (math.gamma((1 + c.beta) / 2) * c.beta * 2 ** ((c.beta - 1) / 2))) ** (1 / c.beta)

    def evaluate(self, x):
        if not np.all(np.isfinite(x)) or np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError('Nonfinite or out-of-bounds candidate')
        value = float(self.obj_func(x.copy()))
        self.evaluations += 1
        if not np.isfinite(value):
            raise FloatingPointError('Objective returned a nonfinite value')
        return value

    def initialize(self):
        c, rng = self.config, self.rng
        ne = int(c.expansion * c.population)
        # The narrow-domain count uses decimal arithmetic so that 0.7*90 is 63,
        # not 62 as binary-double rounding would give.
        nn = int((Decimal(str(c.narrow_fraction)) * ne).to_integral_value(rounding=ROUND_FLOOR))
        center = (self.lower + self.upper) / 2
        half = c.eta * (self.upper - self.lower)
        candidates = np.vstack((rng.uniform(center - half, center + half, (nn, self.dim)),
                                rng.uniform(self.lower, self.upper, (ne - nn, self.dim))))
        candidates = np.vstack((candidates, self.lower + self.upper - candidates))
        candidates = np.clip(candidates, self.lower, self.upper)
        fit = np.array([self.evaluate(x) for x in candidates])
        keep = np.argsort(fit, kind='stable')[:c.population]
        # Roles are assigned after a random permutation of the retained individuals.
        keep = keep[rng.permutation(c.population)]
        return candidates[keep].copy(), fit[keep].copy()

    def levy(self, t):
        """Mantegna step scaled by the stage coefficient and the decay (1-t/T)^rho."""
        c = self.config
        u = self.rng.normal(0, self.sigma, self.dim)
        v = self.rng.normal(0, 1, self.dim)
        return (c.alpha[stage(t, c.iterations)] * (self.upper - self.lower) *
                (1 - t / c.iterations) ** c.rho * u / (np.abs(v) + c.epsilon) ** (1 / c.beta))

    def local_bounds(self, anchor, radius):
        """Sort the two local endpoints per component, then clip to global bounds."""
        a, b = anchor * (1 - radius), anchor * (1 + radius)
        local_lower = np.clip(np.minimum(a, b), self.lower, self.upper)
        local_upper = np.clip(np.maximum(a, b), self.lower, self.upper)
        return local_lower, local_upper

    def optimize(self):
        c = self.config
        self.rng = np.random.default_rng(self.seed)
        self.evaluations = 0
        x, fit = self.initialize()
        previous = x.copy()
        best_idx = int(np.argmin(fit))
        best, best_fit = x[best_idx].copy(), float(fit[best_idx])
        history, eval_history = [best_fit], [self.evaluations]
        limits = np.cumsum(self.counts)
        for t in range(1, c.iterations + 1):
            p = t / c.iterations
            old = x.copy()
            current_best = old[np.argmin(fit)].copy()
            worst = old[np.argmax(fit)].copy()
            global_best = best.copy()
            mean_fit = float(np.mean(fit))
            breed_lo, breed_hi = self.local_bounds(current_best, 1 - p)
            forage_lo, forage_hi = self.local_bounds(global_best, 1 - p)
            proposal = np.empty_like(old)
            for i in range(c.population):
                xi = old[i]
                if i < limits[0]:
                    if self.rng.random() < c.rolling_probability:
                        # Eq. (1)-(2): x_i(t) + alpha*k*x_i(t-1) + b*|x_i(t) - X^w|
                        sign = 1 if self.rng.random() < c.positive_sign_probability else -1
                        candidate = xi + sign * c.rolling_k * previous[i] + c.rolling_b * np.abs(xi - worst)
                    else:
                        # Integer degrees in [0,179]; 0 and 90 degrees keep the position.
                        angle = int(self.rng.integers(0, 180))
                        candidate = (xi.copy() if angle in (0, 90) else
                                     xi + math.tan(math.radians(angle)) * np.abs(xi - previous[i]))
                elif i < limits[1]:
                    b1, b2 = self.rng.random(self.dim), self.rng.random(self.dim)
                    candidate = current_best + b1 * (xi - breed_lo) + b2 * (xi - breed_hi)
                    candidate = np.clip(candidate, breed_lo, breed_hi)
                elif i < limits[2]:
                    c1, c2 = self.rng.normal(size=self.dim), self.rng.random(self.dim)
                    candidate = xi + c1 * (xi - forage_lo) + c2 * (xi - forage_hi)
                    if fit[i] > mean_fit:
                        candidate += self.levy(t)
                else:
                    g = self.rng.normal(size=self.dim)
                    gaussian = (c.delta[stage(t, c.iterations)] * (1 - p) ** c.lam *
                                (self.upper - self.lower) * self.rng.normal(size=self.dim))
                    candidate = (global_best + c.stealing_s * g *
                                 (np.abs(xi - current_best) + np.abs(xi - global_best)) + gaussian)
                proposal[i] = np.clip(candidate, self.lower, self.upper)
            proposed_fit = np.array([self.evaluate(z) for z in proposal])
            # Proposals are generated synchronously and accepted per individual
            # only when strictly better than the incumbent.
            accepted = proposed_fit < fit
            x[accepted], fit[accepted] = proposal[accepted], proposed_fit[accepted]
            previous = old
            index = int(np.argmin(fit))
            if fit[index] < best_fit:
                best, best_fit = x[index].copy(), float(fit[index])
            history.append(best_fit)
            eval_history.append(self.evaluations)
        self.global_best_pos, self.global_best_fitness = best, best_fit
        self.history = np.asarray(history)
        return {'best_position': best, 'best_fitness': best_fit, 'history': self.history,
                'evaluation_history': np.asarray(eval_history), 'evaluations': self.evaluations,
                'role_counts': self.counts.copy(), 'config': asdict(c), 'seed': self.seed}
