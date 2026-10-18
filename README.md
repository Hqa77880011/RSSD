# RSSD

Implementation of constrained rate-equivariant spiking dynamics for event streams.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The generator creates synthetic event streams with independent recording groups. It records event times in seconds, sensor geometry, classes and seed. Prepared interventions share the same locked grouping.

```bash
python -m rssd.synthetic --output data/generated --groups 24 --seed 17
rssd prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

For external data: Acquire [DVS-Gesture](https://research.ibm.com/interactive/dvsgesture/), [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101) or [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671). Export each recording as NPZ `events` [N,4] with sorted time in seconds, x, y, polarity; scalar `duration`; and `sensor_size` [height,width]. A JSON manifest contains `{id,path,group,split,label}`. Preserve official recording/subject splits before deriving interventions. The preparation command validates grouped separation and the event layout.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
rssd train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
```

## Evaluation

```bash
rssd evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

## Comparisons and ablations

Fit each supported `--variant` into a separate directory, evaluate its own checkpoint, then pass the evaluated run directories to `compare --inputs`.

```bash
rssd compare --inputs runs/reproduce --output outputs/comparison
rssd plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `history.csv`, `model.pt`, `samples.csv`, `rate_curve.csv`, `metrics.json`, `source.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
