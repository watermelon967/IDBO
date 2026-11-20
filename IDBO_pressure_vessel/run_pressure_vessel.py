"""Seeded IDBO experiment on the mixed discrete-continuous pressure-vessel model."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from idbo import IDBO, Config
from pressure_vessel import (LOWER, UPPER, SCALES, COST_UPPER, THICKNESS_STEP,
                             VesselObjective, cost, constraints, feasible, decode)


def main():
    p = argparse.ArgumentParser(description='Seeded IDBO mixed discrete-continuous pressure-vessel experiment')
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--population', type=int, default=30)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--base-seed', type=int, default=20261103)
    p.add_argument('--out', default='results_pressure_vessel')
    a = p.parse_args()
    if a.runs < 1 or a.base_seed < 0:
        p.error('runs must be positive and base-seed nonnegative')
    cfg = Config(population=a.population, iterations=a.iterations)
    # Validate the configuration before any output directory is created.
    IDBO(lambda x: 0., dim=4, lower=LOWER, upper=UPPER, config=cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    metadata = dict(
        implementation_version='1.0',
        local_bounds='componentwise endpoint sorting followed by global-bound clipping',
        model='discrete thicknesses; continuous radius and length; no shifts',
        variable_types=['discrete', 'discrete', 'continuous', 'continuous'],
        thickness_step=THICKNESS_STEP, length_upper=240.,
        encoding='continuous population; decode before every evaluation; no writeback',
        rounding='nearest grid point, ties to even (numpy.rint)',
        dimension=4, lower=LOWER.tolist(), upper=UPPER.tolist(), config=asdict(cfg),
        runs=a.runs, base_seed=a.base_seed,
        seed_rule='SeedSequence([base_seed, run_index_zero_based])',
        constraint_handling='feasible: cost/COST_UPPER; infeasible: 2+sum(max(g/scales,0))',
        scales=SCALES.tolist(), cost_upper=COST_UPPER, feasibility_tolerance=0.,
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted(root.glob('*.py'))})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    rows, histories, positions, encoded_positions, eval_histories = [], [], [], [], []
    for run in range(a.runs):
        seed = int(np.random.SeedSequence([a.base_seed, run]).generate_state(1)[0])
        obj = VesselObjective()
        start = time.perf_counter()
        result = IDBO(obj, dim=4, lower=LOWER, upper=UPPER, seed=seed, config=cfg).optimize()
        elapsed = time.perf_counter() - start
        encoded = result['best_position']
        x = decode(encoded)
        ok = feasible(x)
        row = dict(run=run + 1, seed=seed, feasible=ok, cost=cost(x), score=result['best_fitness'],
                   **dict(zip(['Ts', 'Th', 'R', 'L'], x.tolist())),
                   **dict(zip(['g1', 'g2', 'g3', 'g4'], constraints(x).tolist())),
                   normalized_violation=float(np.maximum(constraints(x) / SCALES, 0.).sum()),
                   evaluations=result['evaluations'], seconds=elapsed)
        rows.append(row)
        histories.append(result['history'])
        positions.append(x)
        encoded_positions.append(encoded.copy())
        eval_histories.append(result['evaluation_history'])
        print(f'Run {run + 1}/{a.runs}: feasible={ok}, cost={row["cost"]:.10g}', flush=True)

    with (out / 'raw_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    good = np.array([r['cost'] for r in rows if r['feasible']])
    summary = dict(runs=a.runs, feasible_runs=len(good), feasibility_rate=len(good) / a.runs,
                   statistics_scope='FEASIBLE_RUNS_ONLY',
                   best=float(good.min()) if len(good) else None,
                   mean=float(good.mean()) if len(good) else None,
                   std_ddof1=float(good.std(ddof=1)) if len(good) > 1 else None,
                   median=float(np.median(good)) if len(good) else None,
                   IQR=float(np.percentile(good, 75) - np.percentile(good, 25)) if len(good) else None,
                   mean_seconds=float(np.mean([r['seconds'] for r in rows])))
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)

    best = min((r for r in rows if r['feasible']), key=lambda r: r['cost'], default=None)
    (out / 'best_solution.json').write_text(json.dumps(best, indent=2), encoding='utf-8')

    score = np.array(histories)
    # Feasible scores lie in [0,1]; infeasible scores are at least 2.
    np.savez_compressed(out / 'trajectories.npz', score=score,
                        feasible_cost=np.where(score <= 1., score * COST_UPPER, np.nan),
                        best_positions=np.array(positions),
                        best_encoded_positions=np.array(encoded_positions),
                        evaluations=np.array(eval_histories))
    print(json.dumps(summary, indent=2)); print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
