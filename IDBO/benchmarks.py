"""Unshifted F1-F9; F7 has nonnegative deterministic pseudo-noise vanishing at zero."""
import numpy as np


def f1(x):
    return np.sum(x*x)


def f2(x):
    return np.sum(np.cumsum(x)**2)


def f3(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def f4(x):
    return np.max(np.abs(x))


def f5(x):
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)


def f6(x):
    return np.sum(x*x)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1)))) + 1


def f7(x):
    """Quartic base term; Benchmark adds the fixed nonnegative pseudo-noise."""
    return np.sum(np.arange(1,len(x)+1)*x**4)


def f8(x):
    return np.sum(x*x - 10*np.cos(2*np.pi*x) + 10)


def f9(x):
    return -20*np.exp(-.2*np.sqrt(np.mean(x*x))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e


FUNCTIONS = {
    'F1': (f1, -100, 100, 'Sphere'),
    'F2': (f2, -100, 100, 'Schwefel 1.2'),
    'F3': (f3, -100, 100, 'Schwefel 2.22'),
    'F4': (f4, -100, 100, 'Schwefel 2.21'),
    'F5': (f5, -30, 30, 'Rosenbrock'),
    'F6': (f6, -600, 600, 'Griewank'),
    'F7': (f7, -1.28, 1.28, 'Quartic Function (with pseudo-noise)'),
    'F8': (f8, -5.12, 5.12, 'Rastrigin'),
    'F9': (f9, -32, 32, 'Ackley'),
}

NOISE_AMPLITUDE = 1e-16
NOISE_FREQUENCY = 1234.567


class Benchmark:
    """Fixed unshifted instance; every exact theoretical minimum equals zero.

    noise_seed fixes only the F7 pseudo-noise vector, which is never resampled
    during a run. F7 = sum(j*x_j**4) + 1e-16*sin(1234.567*dot(x,v))**2.
    """

    def __init__(self, name, dim=10, noise_seed=2026):
        if not isinstance(dim, (int, np.integer)) or dim < 2:
            raise ValueError('dim must be an integer >= 2')
        if not isinstance(noise_seed, (int, np.integer)) or noise_seed < 0:
            raise ValueError('noise_seed must be a nonnegative integer')
        self.name, self.dim, self.noise_seed = name, int(dim), int(noise_seed)
        self.base, self.lower, self.upper, self.title = FUNCTIONS[name]
        self.noise_vector = None
        if name == 'F7':
            rng = np.random.default_rng(self.noise_seed + 7*1009)
            self.noise_vector = rng.normal(0, 1, self.dim)
        self.optimum = 0.0

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,) or not np.all(np.isfinite(x)):
            raise ValueError('Expected a finite vector of the configured dimension')
        value = float(self.base(x))
        if self.name == 'F7':
            value += NOISE_AMPLITUDE * np.sin(NOISE_FREQUENCY * np.dot(x, self.noise_vector))**2
        return float(value)

    def record(self):
        return dict(
            name=self.name, title=self.title, dimension=self.dim,
            lower=self.lower, upper=self.upper, transformation='none',
            exact_optimum=self.optimum,
            optimum_position=([1.0]*self.dim if self.name == 'F5' else [0.0]*self.dim),
            noise_seed=(self.noise_seed if self.name == 'F7' else None),
            noise_vector=(self.noise_vector.tolist() if self.noise_vector is not None else None),
            noise_amplitude=(NOISE_AMPLITUDE if self.name == 'F7' else 0.0),
            noise_formula=('sin(1234.567 * dot(x, v))**2' if self.name == 'F7' else None),
        )
