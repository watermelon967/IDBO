# IDBO：基准函数与压力容器设计案例

本仓库提供 IDBO 优化算法、F1–F9 基准函数、压力容器设计案例及运行脚本，并说明运行参数、随机种子和结果文件。

压力容器案例见同仓库的 `IDBO_pressure_vessel` 目录。

## 1. 文件结构

| 路径 | 内容 |
| --- | --- |
| `requirements.txt` | Python 依赖 |
| `IDBO/idbo.py` | IDBO 算法与参数配置 |
| `IDBO/benchmarks.py` | F1–F9 基准函数 |
| `IDBO/run_benchmarks.py` | 基准实验运行入口 |
| `IDBO_pressure_vessel/idbo.py` | 与 `IDBO/idbo.py` 相同 |
| `IDBO_pressure_vessel/pressure_vessel.py` | 压力容器模型、厚度离散化与约束评分 |
| `IDBO_pressure_vessel/run_pressure_vessel.py` | 压力容器实验运行入口 |

两个目录可分别运行。基准函数与压力容器模型均由代码定义，无需下载外部数据。



## 2. IDBO Algorithm Pseudocode

The concise pseudocode below summarizes the implementation in `IDBO/idbo.py`. The pressure-vessel case uses the same optimizer; its objective wrapper decodes the two thickness variables and calculates the constraint-based fitness score.

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

- Rolling: $x_i'=x_i+s kx_{i,\mathrm{prev}}+b|x_i-x_w|$; otherwise, for $\theta\notin\{0^\circ,90^\circ\}$, $x_i'=x_i+\tan(\theta)|x_i-x_{i,\mathrm{prev}}|$ (with $\theta$ sampled in degrees; at $0^\circ$ or $90^\circ$, the position is unchanged).
- Breeding: $x_i'=x_{cb}+r_1\odot(x_i-L_b)+r_2\odot(x_i-U_b)$, where $r_1,r_2\sim U(0,1)^D$.
- Foraging: $x_i'=x_i+c_1\odot(x_i-L_f)+c_2\odot(x_i-U_f)$, where $c_1\sim\mathcal{N}(0,I_D)$ and $c_2\sim U(0,1)^D$; add $\mathrm{Levy}(t)$ only when the current fitness exceeds the population mean.
- Stealing: $x_i'=x_{gb}+S_0g\odot(|x_i-x_{cb}|+|x_i-x_{gb}|)+G(t)$, where $g\sim\mathcal{N}(0,I_D)$.

The stage-dependent Levy perturbation is

$$\mathrm{Levy}(t)=\alpha_s(ub-lb)(1-t/T)^\rho\frac{u}{(|v|+\varepsilon)^{1/\beta}},$$

where $u\sim\mathcal{N}(0,\sigma^2 I_D)$, $v\sim\mathcal{N}(0,I_D)$, and $\sigma$ is calculated using the Mantegna method. The stage-dependent Gaussian perturbation is

$$G(t)=\delta_s(1-t/T)^\lambda(ub-lb)\odot\epsilon_g,\qquad \epsilon_g\sim\mathcal{N}(0,I_D).$$

The default role proportions are 0.20, 0.13, 0.25, and 0.42 for rolling, breeding, foraging, and stealing individuals, respectively. With $N=30$, the corresponding role counts are 6, 4, 7, and 13. All proposals in an iteration are generated synchronously from the pre-update population. The algorithm evaluates $2\kappa N+NT$ candidates per run under the stated implementation.

## 3. 环境与运行

运行验证环境：Windows 11（10.0.26200）、Python 3.13.16、NumPy 2.5.4。代码仅依赖 NumPy 与 Python 标准库，无需 GPU。依赖版本见 `requirements.txt`；每次运行的实际 Python、NumPy、操作系统版本、运行参数与源码 SHA-256 写入 `metadata.json`。

以下命令均在仓库根目录执行。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

基准实验运行示例（F1–F9，维度 10、种群规模 30、最大迭代 300，每个函数独立运行 30 次）：

```bash
python IDBO/run_benchmarks.py --dim 10 --population 30 --iterations 300 --runs 30 --noise-seed 2026 --base-seed 20261102 --out results_benchmark_10D
```

压力容器案例：

```bash
python IDBO_pressure_vessel/run_pressure_vessel.py --runs 30 --population 30 --iterations 300 --base-seed 20261103 --out results_pressure_vessel
```

`--dim` 设置维度（至少为 2）；运行 30、50 或 100 维实验时，修改该参数并分别指定输出目录。`--functions F1 F3 F7` 选择函数子集。`--out` 为输出目录，相对命令执行时的工作目录；目录必须尚不存在，已存在时程序报错退出，重跑请指定新目录。两个入口脚本可分别直接运行。

## 4. 基准函数

函数不使用平移、旋转或函数值偏置；下表范围适用于每个变量分量。

| 编号 | 名称 | 每维搜索范围 | 函数表达式 | 精确最优值 |
| --- | --- | --- | --- | --- |
| F1 | Sphere | [-100,100] | $\sum_{j=1}^{D}x_j^2$ | 0 |
| F2 | Schwefel 1.2 | [-100,100] | $\sum_{j=1}^{D}(\sum_{k=1}^{j}x_k)^2$ | 0 |
| F3 | Schwefel 2.22 | [-100,100] | $\sum_{j=1}^{D}\lvert x_j\rvert+\prod_{j=1}^{D}\lvert x_j\rvert$ | 0 |
| F4 | Schwefel 2.21 | [-100,100] | $\max_j\lvert x_j\rvert$ | 0 |
| F5 | Rosenbrock | [-30,30] | $\sum_{j=1}^{D-1}[100(x_{j+1}-x_j^2)^2+(x_j-1)^2]$ | 0 |
| F6 | Griewank | [-600,600] | $\sum_jx_j^2/4000-\prod_j\cos(x_j/\sqrt j)+1$ | 0 |
| F7 | Quartic Function (with pseudo-noise) | [-1.28,1.28] | $\sum_{j=1}^{D}jx_j^4+10^{-16}q(x)$ | 0 |
| F8 | Rastrigin | [-5.12,5.12] | $\sum_j[x_j^2-10\cos(2\pi x_j)+10]$ | 0 |
| F9 | Ackley | [-32,32] | $-20e^{-0.2\sqrt{\sum_jx_j^2/D}}-e^{\sum_j\cos(2\pi x_j)/D}+20+e$ | 0 |

F5 的最优位置为全 1 向量，其余函数的最优位置为全 0 向量，均位于搜索范围内。

F7 的微扰为

$$q(x)=\sin^2(1234.567\,x^{\mathrm{T}}v),\qquad F_7(x)=\sum_{j=1}^{D}jx_j^4+10^{-16}q(x)$$

其中 $v$ 由 `noise_seed + 7*1009` 初始化的随机数生成器产生的标准正态向量确定，生成后保持不变；$q(x)\ge0$ 且 $q(0)=0$，四次项同样非负，故 $F_7$ 的精确全局最优值为 0，取得于原点。完整定义见 `IDBO/benchmarks.py`，向量与公式记录于 `benchmark_instances.json`。

## 5. 算法参数与实现约定

| 参数 | 默认值 |
| --- | --- |
| 种群规模 N / 最大迭代次数 T | 30 / 300 |
| 维度 D（命令行 `--dim`） | 10（不小于 2） |
| 候选群扩增倍数 κ | 3 |
| 中心窄域半宽系数 η | 0.2 |
| 原候选群的窄域采样比例 | 0.7 |
| 滚球、繁殖、觅食、偷窃角色比例 | (0.20,0.13,0.25,0.42) |
| Lévy β / ε | 1.5 / 1e-12 |
| Lévy 分阶段尺度 α | (0.06,0.03,0.012) |
| Lévy 衰减指数 ρ | 1 |
| 高斯分阶段尺度 δ | (0.08,0.04,0.01) |
| 高斯衰减指数 λ | 3 |
| 滚球更新概率 / 滚球正号概率 | 0.9 / 0.9 |
| 滚球 k / b | 0.1 / 0.3 |
| 偷窃 $S_0$ | 0.5 |

参数在 `Config` 中统一定义，各基准函数与压力容器案例使用同一组算法参数。运行脚本可指定种群规模和迭代次数，其余参数取上表值。

运行约定：

- 初始化生成 $\kappa N$ 个全域与窄域候选，并加入反向候选，评价后保留最优 $N$ 个；保留个体随机打乱后分配固定角色。默认 $N=30$ 时，滚球、繁殖、觅食、偷窃个体数量依次为 6、4、7、13。
- 局部边界的候选端点先逐维排序，再裁剪至全局范围；繁殖候选限制在繁殖局部区间内，所有候选均在全局搜索范围内评价。
- 仅对适应度劣于种群均值的觅食个体叠加分阶段 Lévy 扰动，偷窃个体叠加自适应高斯扰动。
- 每轮基于更新前的种群同步生成候选，仅接受评分严格改善的候选，并记录历史最优解。
- 每次运行的评价次数为 $2\kappa N+NT$，默认设置下为 9180 次；收敛轨迹包含初始化点，共 $T+1$ 个点。

## 6. 压力容器案例

设计变量按 $x=[T_s,T_h,R,L]$ 排列，依次为壳体厚度、封头厚度、内半径与圆柱段长度。两个厚度取 0.0625 的离散网格，$R$ 与 $L$ 为连续变量：

$$T_s=0.0625\,n_1,\quad T_h=0.0625\,n_2,\quad n_1,n_2\in\{0,1,\dots,1600\}$$

$$0\le T_s,T_h\le100,\qquad 10\le R\le200,\qquad 10\le L\le240$$

目标函数：

$$f(x)=0.6224T_sRL+1.7781T_hR^2+3.1661T_s^2L+19.84T_s^2R$$

约束条件：

$$g_1=0.0193R-T_s\le0,\qquad g_2=0.00954R-T_h\le0$$

$$g_3=1296000-\pi R^2L-\frac{4}{3}\pi R^3\le0,\qquad g_4=L-240\le0$$

评价前，将连续编码中的两个厚度变量映射到最近的 0.0625 网格点，并使用解码后的实际变量计算成本与约束。结果文件报告实际设计变量，轨迹文件同时保存实际变量与连续编码。

采用可行性优先评分。令 $UPPER=[100,100,200,240]$、$C=f(UPPER)=57378560$、$s=[100,100,1296000,240]$，以下 $x$ 均指解码后的设计变量，内部评分为

$$S(x)=\begin{cases}
f(x)/C, & \text{全部 } g_i(x)\le 0;\\
2+\sum_i\max\{g_i(x)/s_i,\,0\}, & \text{否则}.
\end{cases}$$

所有适应度比较及 Lévy 扰动的触发判断均使用 $S(x)$，工程结果报告原始成本 $f(x)$。可行性按解码后的实际变量判定，要求满足变量范围、厚度离散要求及全部约束，约束容差为 0。成本统计仅包含可行运行，同时报告可行率。默认设置下，每次运行评价 9180 次。

## 7. 随机种子

| 用途 | 设置 |
| --- | --- |
| 基准实验 | `base_seed=20261102`；Fi 的第 r 次运行种子为 `SeedSequence([base_seed,i,r]).generate_state(1)[0]` |
| 压力容器实验 | `base_seed=20261103`；第 r 次运行种子为 `SeedSequence([base_seed,r]).generate_state(1)[0]` |
| F7 固定向量 | `noise_seed=2026`；`default_rng(noise_seed+7*1009)` 生成 D 维标准正态向量 |
| 直接调用 IDBO 类 | 默认 `seed=2026`；运行脚本显式传入各次运行种子 |

`i` 为函数编号，`r` 从 0 开始；CSV 中运行编号从 1 开始。逐次种子保存在 `raw_results.csv`，F7 向量保存在 `benchmark_instances.json`。F7 的向量种子与优化运行种子用途不同；相同问题、配置、种子与软件环境下可复核优化轨迹。比较算法时使用相同的函数定义与同一 F7 向量。

## 8. 结果文件

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 运行参数、配置、种子规则、环境（Python、NumPy、操作系统）与源码 SHA-256 |
| `benchmark_instances.json` | 各函数的名称、维度、范围、变换方式、精确最优值与最优位置；F7 的固定向量、幅度与公式 |
| `raw_results.csv` | 基准：函数、运行编号、种子、最优值、绝对误差、精确最优值、评价次数、耗时；压力容器：运行编号、种子、可行性、成本、评分、Ts/Th/R/L、g1–g4、归一化违反量、评价次数、耗时 |
| `summary.csv` | 基准：最优值、均值、样本标准差（ddof=1）、中位数、四分位距、平均耗时；压力容器：运行次数、可行次数、可行率、统计口径（仅可行运行）、最优值、均值、样本标准差、中位数、四分位距、平均耗时；基准实验仅运行一次时，样本标准差留空 |
| `F*_trajectories.npz` | 逐次收敛轨迹、逐次最终位置与逐次累计评价次数 |
| `best_solution.json` | 压力容器最佳可行方案，含设计变量、成本与约束残差；无可行结果为 `null` |
| `trajectories.npz` | 逐代最优评分、由评分换算的可行成本（不可行阶段为 NaN）、解码后的实际变量、连续编码与逐次累计评价次数 |

## Copyright and Usage Notice

Copyright (c) 2026 The Authors. All rights reserved.

This repository is made publicly available for
peer-review and academic inspection purposes only.
No license is granted to use, modify, reproduce,
redistribute, or commercially exploit the source code
or its implementations without prior written permission
from the authors.
