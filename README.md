# IDBO: Benchmark Functions and Pressure Vessel Design Case

This repository provides the Improved Dung Beetle Optimizer (IDBO), the F1–F9 benchmark functions, the pressure vessel design optimization case, and the corresponding execution scripts. It also documents the experimental parameters, random-seed settings, and output files.

The pressure vessel case is located in the `IDBO_pressure_vessel` directory.

## 1. Repository Structure

| Path | Description |
| --- | --- |
| `requirements.txt` | Python dependencies |
| `IDBO/idbo.py` | IDBO implementation and parameter configuration |
| `IDBO/benchmarks.py` | Definitions of benchmark functions F1–F9 |
| `IDBO/run_benchmarks.py` | Entry point for benchmark experiments |
| `IDBO_pressure_vessel/idbo.py` | Same IDBO implementation as `IDBO/idbo.py` |
| `IDBO_pressure_vessel/pressure_vessel.py` | Pressure vessel model, thickness discretization, and constraint-based fitness evaluation |
| `IDBO_pressure_vessel/run_pressure_vessel.py` | Entry point for the pressure vessel experiment |

The benchmark and pressure vessel experiments can be run separately. The benchmark functions and pressure vessel model are defined in the source code; no external data download is required.

## 2. IDBO Algorithm Pseudocode

The concise pseudocode below summarizes the implementation in `IDBO/idbo.py`. The pressure vessel case uses the same optimizer; its objective wrapper decodes the two thickness variables and calculates the constraint-based fitness score.

```text
Algorithm 1: Improved Dung Beetle Optimizer (IDBO)

Input: Objective function f, dimension D, bounds lb and ub,
       population size N, maximum iterations T, run-specific seed,
       IDBO parameters
Output: Global-best solution x_best and fitness f_best

1.  Initialize the random-number generator with the run-specific seed.
2.  Generate kappa*N candidates: floor(q*kappa*N) in the central
    narrow domain and the remainder in the full search domain.
3.  Generate opposition candidates x_opp = lb + ub - x; combine
    both sets, clip to [lb, ub], and evaluate all 2*kappa*N candidates.
4.  Retain the N candidates with the lowest fitness, randomly permute
    them, and assign rolling, breeding, foraging, and stealing roles
    according to the configured proportions.
5.  Initialize the previous population and the global-best solution.
6.  For t = 1,...,T:
      a. Set p = t/T and select stage s:
           s = 1 if p <= 1/3; s = 2 if 1/3 < p <= 2/3;
           otherwise s = 3.
      b. Save the pre-update population; determine its current best,
         worst, mean fitness, and the historical global best.
      c. Construct breeding bounds around the current best and foraging
         bounds around the global best, using radius 1-p; sort each pair
         of endpoints component-wise and clip them to [lb, ub].
      d. Generate one proposal for each individual from the same
         pre-update population:
           Rolling:
             With the configured probability, use the signed previous
             position and distance from the current worst; otherwise,
             use the angle-based update with theta in {0,...,179}.
           Breeding:
             Update around the current best using two independent
             uniform random vectors; clip to the breeding bounds.
           Foraging:
             Update using a normal and a uniform random vector; if the
             individual's fitness exceeds the population mean, add Levy(t).
           Stealing:
             Update around the global best using distances to the current
             best and global best, plus the adaptive Gaussian perturbation.
      e. Clip all proposals to [lb, ub] and evaluate them.
      f. Accept each proposal only if its fitness is strictly lower than
         that of its corresponding individual.
      g. Set the previous population to the saved pre-update population;
         update the historical global best if an improvement is found.
7.  Return x_best and f_best.
```

The four update operations are implemented as follows (all vector operations are component-wise):

- **Rolling:** $x_i'=x_i+s kx_{i,\mathrm{prev}}+b|x_i-x_w|$; otherwise, for $\theta\notin\{0^\circ,90^\circ\}$, $x_i'=x_i+\tan(\theta)|x_i-x_{i,\mathrm{prev}}|$ (with $\theta$ sampled in degrees; at $0^\circ$ or $90^\circ$, the position is unchanged).
- **Breeding:** $x_i'=x_{cb}+r_1\odot(x_i-L_b)+r_2\odot(x_i-U_b)$, where $r_1,r_2\sim U(0,1)^D$.
- **Foraging:** $x_i'=x_i+c_1\odot(x_i-L_f)+c_2\odot(x_i-U_f)$, where $c_1\sim\mathcal{N}(0,I_D)$ and $c_2\sim U(0,1)^D$; add $\mathrm{Levy}(t)$ only when the current fitness exceeds the population mean.
- **Stealing:** $x_i'=x_{gb}+S_0g\odot(|x_i-x_{cb}|+|x_i-x_{gb}|)+G(t)$, where $g\sim\mathcal{N}(0,I_D)$.

The stage-dependent Levy perturbation is

$$\mathrm{Levy}(t)=\alpha_s(ub-lb)(1-t/T)^\rho\frac{u}{(|v|+\varepsilon)^{1/\beta}},$$

where $u\sim\mathcal{N}(0,\sigma^2 I_D)$, $v\sim\mathcal{N}(0,I_D)$, and $\sigma$ is calculated using the Mantegna method. The stage-dependent Gaussian perturbation is

$$G(t)=\delta_s(1-t/T)^\lambda(ub-lb)\odot\epsilon_g,\qquad \epsilon_g\sim\mathcal{N}(0,I_D).$$

The default role proportions are 0.20, 0.13, 0.25, and 0.42 for rolling, breeding, foraging, and stealing individuals, respectively. With $N=30$, the corresponding role counts are 6, 4, 7, and 13. All proposals in an iteration are generated synchronously from the pre-update population. The algorithm evaluates $2\kappa N+NT$ candidates per run under the stated implementation.

## 3. Environment and Execution

The reported validation environment is Windows 11 (10.0.26200), Python 3.13.16, and NumPy 2.5.4. The code depends only on NumPy and the Python standard library; no GPU is required. Dependency versions are listed in `requirements.txt`. The actual Python, NumPy, and operating system versions, run parameters, and source-code SHA-256 hashes are written to `metadata.json` for each run.

Run the following commands from the repository root.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Example benchmark command (F1–F9, dimension 10, population size 30, maximum 300 iterations, and 30 independent runs per function):

```bash
python IDBO/run_benchmarks.py --dim 10 --population 30 --iterations 300 --runs 30 --noise-seed 2026 --base-seed 20261102 --out results_benchmark_10D
```

Pressure vessel case:

```bash
python IDBO_pressure_vessel/run_pressure_vessel.py --runs 30 --population 30 --iterations 300 --base-seed 20261103 --out results_pressure_vessel
```

`--dim` sets the dimension (at least 2). To run experiments at dimensions 30, 50, or 100, change this argument and specify a separate output directory for each setting. `--functions F1 F3 F7` selects a subset of benchmark functions. `--out` specifies the output directory relative to the current working directory. The directory must not already exist; the program exits with an error if it does. Use a new directory for each rerun. The two entry-point scripts can be executed independently.

## 4. Benchmark Functions

The functions use no shift, rotation, or function-value offset. The ranges below apply independently to every variable component.

| Function | Name | Search range per dimension | Exact optimum |
| --- | --- | --- | --- |
| F1 | Sphere | [-100, 100] | 0 |
| F2 | Schwefel 1.2 | [-100, 100] | 0 |
| F3 | Schwefel 2.22 | [-100, 100] | 0 |
| F4 | Schwefel 2.21 | [-100, 100] | 0 |
| F5 | Rosenbrock | [-30, 30] | 0 |
| F6 | Griewank | [-600, 600] | 0 |
| F7 | Quartic Function (with pseudo-noise) | [-1.28, 1.28] | 0 |
| F8 | Rastrigin | [-5.12, 5.12] | 0 |
| F9 | Ackley | [-32, 32] | 0 |

The mathematical definitions are implemented in `IDBO/benchmarks.py`. F5 has an all-ones global-optimum vector; the global-optimum vector for each of the other functions is the all-zeros vector. All optima lie within their respective search ranges.

F7 uses the perturbation

$$q(x)=\sin^2(1234.567\,x^{\mathrm{T}}v),\qquad F_7(x)=\sum_{j=1}^{D}jx_j^4+10^{-16}q(x).$$

The vector $v$ is generated by a random-number generator initialized with `noise_seed + 7*1009` and then kept fixed. Since $q(x)\geq0$, $q(0)=0$, and the quartic term is also nonnegative, the exact global minimum of F7 is 0 at the origin. The generated vector and formula are recorded in `benchmark_instances.json`.

## 5. Algorithm Parameters and Implementation Details

| Parameter | Default value |
| --- | --- |
| Population size $N$ / maximum iterations $T$ | 30 / 300 |
| Dimension $D$ (command-line argument `--dim`) | 10 (minimum 2) |
| Candidate expansion factor $\kappa$ | 3 |
| Half-width coefficient of the central narrow domain $\eta$ | 0.2 |
| Narrow-domain sampling proportion in the original candidate set | 0.7 |
| Role proportions (rolling, breeding, foraging, stealing) | (0.20, 0.13, 0.25, 0.42) |
| Levy $\beta$ / $\varepsilon$ | 1.5 / 1e-12 |
| Stage-dependent Levy scale $\alpha$ | (0.06, 0.03, 0.012) |
| Levy decay exponent $\rho$ | 1 |
| Stage-dependent Gaussian scale $\delta$ | (0.08, 0.04, 0.01) |
| Gaussian decay exponent $\lambda$ | 3 |
| Rolling-update probability / positive-sign probability | 0.9 / 0.9 |
| Rolling parameters $k$ / $b$ | 0.1 / 0.3 |
| Stealing parameter $S_0$ | 0.5 |

Parameters are defined in `Config`. The benchmark functions and pressure vessel case use the same IDBO parameter settings. The run scripts allow the population size and iteration count to be specified; other parameters use the defaults listed above.

Implementation conventions:

- Initialization generates $\kappa N$ candidates from the full and narrow domains and adds their opposition-based candidates. After evaluation, the best $N$ candidates are retained and randomly shuffled before role assignment. With the default $N=30$, the numbers of rolling, breeding, foraging, and stealing individuals are 6, 4, 7, and 13, respectively.
- Endpoints of local bounds are sorted component-wise and then clipped to the global bounds. Breeding candidates are restricted to the breeding local bounds; all candidates are evaluated within the global search range.
- The stage-dependent Levy perturbation is added only to foraging individuals whose fitness is worse than the population mean. Stealing individuals receive the adaptive Gaussian perturbation.
- All proposals in an iteration are generated synchronously from the pre-update population. A proposal is accepted only if it strictly improves the fitness of its corresponding individual, and the historical global best is updated accordingly.
- The number of objective-function evaluations per run is $2\kappa N+NT$, which is 9,180 under the default settings. The convergence trajectory includes the initialization point and therefore contains $T+1$ points.

## 6. Pressure Vessel Design Case

The design variables are ordered as $x=[T_s,T_h,R,L]$, representing shell thickness, head thickness, inner radius, and cylindrical-section length, respectively. The two thickness variables lie on a discrete grid with a step size of 0.0625, while $R$ and $L$ are continuous:

$$T_s=0.0625n_1,\quad T_h=0.0625n_2,\quad n_1,n_2\in\{0,1,\ldots,1600\}$$

$$0\leq T_s,T_h\leq100,\qquad 10\leq R\leq200,\qquad 10\leq L\leq240.$$

The objective function is

$$f(x)=0.6224T_sRL+1.7781T_hR^2+3.1661T_s^2L+19.84T_s^2R.$$

The constraints are

$$g_1=0.0193R-T_s\leq0,\qquad g_2=0.00954R-T_h\leq0,$$

$$g_3=1296000-\pi R^2L-\frac{4}{3}\pi R^3\leq0,\qquad g_4=L-240\leq0.$$

Before evaluation, the two thickness variables in the continuous encoding are mapped to the nearest 0.0625 grid points. The cost and constraints are calculated using the decoded design variables. Result files report the actual design variables; trajectory files save both the actual variables and the continuous encoding.

A feasibility-priority fitness score is used. Let $UPPER=[100,100,200,240]$, $C=f(UPPER)=57378560$, and $s=[100,100,1296000,240]$. For a decoded design $x$, the internal score is

$$S(x)=\begin{cases}
f(x)/C, & \text{if all } g_i(x)\leq0;\\
2+\sum_i\max\{g_i(x)/s_i,0\}, & \text{otherwise}.
\end{cases}$$

All fitness comparisons and Levy-trigger decisions use $S(x)$, while engineering results report the original cost $f(x)$. Feasibility is determined using the decoded design variables and requires satisfaction of the variable bounds, thickness discretization, and all constraints, with a constraint tolerance of 0. Cost statistics include feasible runs only; the feasibility rate is also reported. Under the default settings, each run performs 9,180 objective-function evaluations.

## 7. Random Seeds

| Purpose | Setting |
| --- | --- |
| Benchmark experiments | `base_seed=20261102`; the seed for function `Fi` and run `r` is `SeedSequence([base_seed,i,r]).generate_state(1)[0]` |
| Pressure vessel experiments | `base_seed=20261103`; the seed for run `r` is `SeedSequence([base_seed,r]).generate_state(1)[0]` |
| Fixed F7 vector | `noise_seed=2026`; a D-dimensional standard normal vector is generated using `default_rng(noise_seed+7*1009)` |
| Direct use of the IDBO class | Default `seed=2026`; run scripts explicitly pass the seed for each run |

`i` is the function index, and `r` starts at 0. Run numbers in CSV files start at 1. Per-run seeds are saved in `raw_results.csv`, and the F7 vector is saved in `benchmark_instances.json`. The F7 vector seed and optimizer run seeds serve different purposes. Optimization trajectories can be checked under the same problem, configuration, seeds, and software environment. Compared algorithms use the same function definitions and the same fixed F7 vector.

## 8. Output Files

| File | Description |
| --- | --- |
| `metadata.json` | Run parameters, configuration, seed rules, environment (Python, NumPy, and operating system), and source-code SHA-256 hashes |
| `benchmark_instances.json` | Function names, dimensions, bounds, transformations, exact optima and optimum locations; the fixed F7 vector, amplitude, and formula |
| `raw_results.csv` | Benchmark: function, run number, seed, best value, absolute error, exact optimum, evaluation count, and runtime. Pressure vessel: run number, seed, feasibility, cost, score, $T_s/T_h/R/L$, $g_1$–$g_4$, normalized constraint violation, evaluation count, and runtime |
| `summary.csv` | Benchmark: best value, mean, sample standard deviation (`ddof=1`), median, interquartile range, and mean runtime. Pressure vessel: number of runs, number of feasible runs, feasibility rate, statistical scope (feasible runs only), best value, mean, sample standard deviation, median, interquartile range, and mean runtime. For a benchmark experiment with only one run, sample standard deviation is left blank. |
| `F*_trajectories.npz` | Per-run convergence trajectories, final positions, and cumulative evaluation counts |
| `best_solution.json` | Best feasible pressure vessel design, including design variables, cost, and constraint residuals; `null` if no feasible solution is found |
| `trajectories.npz` | Best score per iteration, feasible cost converted from the score (NaN during infeasible stages), decoded design variables, continuous encoding, and cumulative evaluation counts for each run |

## 9. Copyright and Usage Notice

Copyright (c) 2026 The Authors. All rights reserved.

This repository is publicly accessible for peer review and academic inspection. Reviewers are permitted to download and execute the code solely for the purpose of evaluating and verifying the submitted manuscript.

No permission is granted for other uses, modification, redistribution, or commercial exploitation without prior written permission from the authors.
