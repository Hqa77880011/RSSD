"""Generate event returns from three moving-edge trajectory families."""

import argparse
import json
from pathlib import Path

import numpy as np


def recording(rng, motion, count=10000, size=32):
    times = np.sort(rng.uniform(0, 1, count))
    phase = 2 * np.pi * (times + 0.08 * np.sin(2 * np.pi * times))
    offset = rng.uniform(-0.5, 0.5)
    edge = rng.uniform(-0.65, 0.65, count)
    if motion == 0:
        x, y = 0.55 * np.sin(phase), edge
        derivative = np.cos(phase)
    elif motion == 1:
        x, y = edge, 0.55 * np.sin(phase)
        derivative = np.cos(phase)
    elif motion == 2:
        x, y = 0.5 * np.cos(phase) + 0.2 * edge, 0.5 * np.sin(phase) + 0.2 * edge
        derivative = np.sin(phase)
    else:
        raise ValueError("Unknown physical motion family")
    coordinates = np.column_stack([x + 0.08 * offset, y - 0.08 * offset])
    coordinates += rng.normal(0, 0.025, coordinates.shape)
    coordinates = np.clip(np.rint((coordinates + 1) * (size - 1) / 2), 0, size - 1)
    polarity = (derivative + rng.normal(0, 0.15, count) >= 0).astype(int)
    return dict(events=np.column_stack([times, coordinates, polarity]).astype(np.float64),
                duration=np.float32(1), sensor_size=np.array([size, size]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.groups < 6:
        parser.error("Use at least six independent recording groups")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for group in range(args.groups):
        split = "train" if group < 0.6 * args.groups else "validation" if group < 0.8 * args.groups else "test"
        for motion in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, group, motion]))
            name = f"events_{group:03d}_{motion}.npz"
            np.savez_compressed(output / name, **recording(rng, motion))
            records.append(dict(id=f"{group}:{motion}", path=name, group=f"recording-{group}",
                                split=split, label=motion))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="finite moving-edge event-return process", seed=args.seed, groups=args.groups,
        truth="trajectory family; timestamps and coordinates are actual sampled returns",
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
