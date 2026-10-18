# RSSD

PyTorch implementation of constrained rate-equivariant spiking dynamics for event streams.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Event streams

Acquire [DVS-Gesture](https://research.ibm.com/interactive/dvsgesture/), [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101) or [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671). Export each recording as NPZ `events` [N,4] with sorted time in seconds, x, y, polarity; scalar `duration`; and `sensor_size` [height,width]. A JSON manifest contains `{id,path,group,split,label}`. Preserve official recording/subject splits before deriving interventions. The preparation command validates grouped separation and the event layout.

```bash
rssd prepare --input data/manifest.json --data data/events
rssd train --data data/events --variant rssd --output runs/full
rssd evaluate --data data/events --checkpoint runs/full/model.pt --output runs/full
rssd compare --inputs runs/full --output outputs/comparison
rssd plot --output outputs/comparison
```

## Dynamics and controls

Fixed physical-time bins feed a compact learned spatial projection and recurrent LIF stage. The local causal EMA uses actual events per second. A learned sigmoid beta sets `tau=tau0/rho`, `s=rho**beta`, `threshold=threshold0*rho**(1-beta)`; exact exponential decay and zero/subtractive reset use the same normalization. A straight-through sigmoid supplies spike gradients. The head reads the mean rectified normalized membrane representation; spike states are retained for diagnostics. This compact projection backbone makes the constrained dynamics explicit and differs from the paper's larger recognition architectures.

Train `lif`, `plif`, `tet`, `explicit_norm`, `decay_only`, `threshold_only`, `drive_only`, `rate_permuted` or `oracle` as separate runs. Explicit normalization uses the instantaneous bin rate, whereas drive-only uses the shared causal EMA. PLIF learns the baseline time constant; TET uses time-step CE; the oracle receives the applied timestamp factor. `rate_permuted` reflects the estimated rate within its clipping interval, deliberately breaking the correct rate relation. All controls share the same stage/head dimensions and input bins. External SEW/Spikformer architectures are not represented by renamed compact models.

## Interventions and metrics

`--intervention` selects timestamp scaling, random thinning, local superposition, uniform noise, burst noise or spatial structured deletion. Timestamp scaling divides timestamps and duration by the rate factor; thinning retains each event with that factor capped at one; superposition adds local jittered copies above one; noise/burst adds `abs(factor-1)*N` events. Structured deletion removes one side of the sensor and is a failure control. Use separate output directories for each intervention.

`configs/default.json` controls time bins, spatial grid, reference event rate, causal EMA, clipping, tau/threshold/reset, rate grid, training and device. The transformation law is a continuous local-time law; event binning and a causal estimated rate yield approximate discrete correspondence. Evaluation aligns normalized membrane, firing and pre-head representations on semantic time before computing MSE. Metrics include clean accuracy at factor one, normalized area under accuracy versus **log rate factor**, spike count, count relative to the nominal sequence and the three alignment errors. Noise/deletion results use physical-time alignment and are not asserted invariant.

Training writes the final checkpoint and loss history. Evaluation writes `samples.csv`, `rate_curve.csv` and `metrics.json`; comparison emits CSV/JSON and PNG/SVG plots. Rates use [0,1]. Keep the rate grid, splits, augmentation seed and local binning paired across methods.

## Registered compact reproduction setting

Moving-edge trajectories produce timestamped spatial event returns and polarities. The classification target is the physical trajectory family. Whole recording groups are split before the existing timestamp/thinning/noise interventions; labels are not appended as feature channels.

Generate the declared input process, preprocess it, then use the existing method entry points:

```bash
python -m rssd.synthetic --output data/generated --groups 24 --seed 17
rssd prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
rssd train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
rssd evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

`source.json` records the data-generating assumptions and seed. The configuration records `input_source`. All inputs for this selected setting are generated locally using installed package dependencies.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
rssd-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.

A sequence with zero nominal spikes has an undefined spike ratio, recorded as null.
