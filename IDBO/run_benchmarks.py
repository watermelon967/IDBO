"""Run seeded IDBO experiments on the unshifted benchmark functions."""
import argparse, csv, json, platform, time, hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from idbo import IDBO, Config
from benchmarks import FUNCTIONS, Benchmark


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dim', type=int, default=10)
    p.add_argument('--population', type=int, default=30)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--noise-seed', type=int, default=2026, help='Seed for the fixed F7 pseudo-noise vector')
    p.add_argument('--base-seed', type=int, default=20261102)
    p.add_argument('--functions', nargs='+', choices=list(FUNCTIONS), default=list(FUNCTIONS))
    p.add_argument('--out', default='results_benchmarks')
    a = p.parse_args()
    if a.dim < 2 or min(a.runs, a.iterations) < 1 or min(a.base_seed, a.noise_seed) < 0:
        p.error('dim >= 2, positive runs/iterations and nonnegative seeds required')
    c = Config(population=a.population, iterations=a.iterations)
    # Validate the configuration before any output directory is created.
    IDBO(lambda x: 0., dim=a.dim, config=c)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    meta = {'implementation_version': '1.0',
            'benchmark_transform': 'none',
            'local_bounds': 'componentwise endpoint sorting followed by global-bound clipping',
            'arguments': vars(a), 'config': asdict(c),
            'python': platform.python_version(), 'numpy': np.__version__,
            'platform': platform.platform(),
            'seed_rule': 'SeedSequence([base_seed, function_number, run_index_zero_based]); first generated uint32',
            'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ['idbo.py', 'benchmarks.py', 'run_benchmarks.py']}}
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    problems = {name: Benchmark(name, a.dim, a.noise_seed) for name in a.functions}
    (out / 'benchmark_instances.json').write_text(
        json.dumps({name: b.record() for name, b in problems.items()}, indent=2), encoding='utf-8')
    raw, summary = [], []
    for name in a.functions:
        fun = problems[name]
        lo, hi = fun.lower, fun.upper
        histories = []; positions = []; evals = []; values = []; times = []
        for run in range(a.runs):
            seed = int(np.random.SeedSequence([a.base_seed, int(name[1:]), run]).generate_state(1)[0])
            start = time.perf_counter()
            r = IDBO(fun, a.dim, lo, hi, seed, c).optimize()
            elapsed = time.perf_counter() - start
            value = r['best_fitness']; values.append(value); times.append(elapsed)
            histories.append(r['history']); positions.append(r['best_position']); evals.append(r['evaluation_history'])
            raw.append({'function': name, 'run': run + 1, 'seed': seed, 'best_fitness': value,
                        'absolute_error': abs(value - fun.optimum),
                        'exact_optimum': fun.optimum,
                        'evaluations': r['evaluations'], 'seconds': elapsed})
        v = np.array(values)
        summary.append({'function': name, 'best': v.min(), 'mean': v.mean(),
                        'std_ddof1': v.std(ddof=1) if a.runs > 1 else '',
                        'median': np.median(v),
                        'IQR': np.percentile(v, 75) - np.percentile(v, 25),
                        'mean_seconds': np.mean(times)})
        np.savez_compressed(out / (name + '_trajectories.npz'),
                            history=histories, best_positions=positions, evaluations=evals)
        write_csv(out / 'raw_results.csv', raw); write_csv(out / 'summary.csv', summary)
        print(name, 'mean=', format(v.mean(), '.6e'), flush=True)
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
