# Path-Space Model Risk via Signature-Induced Optimal Transport

This package reproduces the numerical tables and Appendix E figures in
*Path-Space Model Risk via Signature-Induced Optimal Transport*. It contains the
path simulators, signature and optimal-transport calculations, surrogate models,
robust-risk evaluations, and reporting code used in the paper.

## Installation

Python 3.9--3.12 and a C/C++ compiler are required. Path signatures are computed
with `iisignature==0.24`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4
python -m pip install --no-build-isolation iisignature==0.24
python -m pip install -r requirements.txt
```

## Usage

List the available experiment targets:

```bash
python reproduce.py --list
```

Run all experiments with the smoke profile:

```bash
python reproduce.py --profile smoke all
```

Run the reported experiment design or an individual target:

```bash
python reproduce.py --profile paper all
python reproduce.py --profile paper options
python reproduce.py --profile paper e2
```

The `smoke` profile uses small Monte Carlo samples. The `paper` profile uses the
reported grids, sample sizes, time discretization, and five replicates defined in
`paper/settings.py`.

Results are written to `outputs/` by default. Use `--run-dir` to choose another
directory:

```bash
python reproduce.py --profile smoke --run-dir outputs/smoke all
```

## Outputs

Experiment summaries are written under `main_results/`, per-replicate results
under `diagnostics/`, and the six numerical paper tables under `paper_tables/`.
Appendix figures are written under `appendix_e1/figures/` and
`appendix_e2/figures/`.

The reported tables, Appendix E CSVs, and figures are included under
`results/reference/`.

## Experiments

| Paper result | Target | Implementation |
|---|---|---|
| Table 1: joint and decoupled budgets | `budget` | `paper/section5.py` |
| Tables 2--3: affine-signature expectation and probability | `fixed` | `sig_dro/workflows/affine_scores.py` |
| Table 4: Asian and lookback options | `options` | `sig_dro/workflows/options.py` |
| Table 5: reserve-functional definitions | No simulation | `sig_dro/functionals.py` |
| Table 6: reserve-functional results | `reserve` | `sig_dro/workflows/reserve.py` |
| Table 7: BM19/J1 comparison | `bm19` | `paper/section5.py` |
| Appendix E.1: parametric geometry | `e1` | `paper/appendix.py` |
| Appendix E.2: model-library geometry | `e2` | `paper/appendix.py` |
| Appendix E.3: penalty and depth ablations | `e3` | `paper/appendix.py` |

## Package layout

```text
reproduce.py          command-line entry point
paper/                experiment definitions and reported configurations
sig_dro/              numerical methods, path models, and workflows
results/reference/    reported tables, Appendix E CSVs, and figures
tests/                unit and package checks
```

## Citation and license

Citation metadata is provided in `CITATION.cff`. The code is released under the
BSD 3-Clause License.
