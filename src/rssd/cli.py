import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Dynamics
from .common import load, save, table, plot


def prepare(a, c):
    records = load(a.input)
    groups = {}
    result = []
    for record in records:
        if record["group"] in groups and groups[record["group"]] != record["split"]:
            raise ValueError("Recording group crosses splits")
        groups[record["group"]] = record["split"]
        path = Path(record["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        raw = np.load(path)
        events = raw["events"]
        if (
            events.ndim != 2
            or events.shape[1] != 4
            or np.any(np.diff(events[:, 0]) < 0)
        ):
            raise ValueError("events must be sorted [time_seconds,x,y,polarity]")
        result.append(dict(record, path=str(path.resolve())))
    save(Path(a.data) / "manifest.json", result)


def bin_events(path, c, rng, factor=1.0, kind="timestamp"):
    raw = np.load(path)
    events = raw["events"].copy().astype(float)
    duration = float(raw["duration"])
    height, width = raw["sensor_size"].astype(int)
    if kind == "timestamp":
        events[:, 0] /= factor
        duration /= factor
    elif kind == "thinning":
        events = events[rng.random(len(events)) < min(factor, 1.0)]
    elif kind == "superposition":
        count = max(0, int((factor - 1) * len(events)))
        if count:
            extra = events[rng.integers(len(events), size=count)].copy()
            extra[:, 0] = np.clip(
                extra[:, 0] + rng.normal(0, c["dt"] / 10, count), 0, duration
            )
            events = np.r_[events, extra]
    elif kind in ("noise", "burst"):
        count = int(abs(factor - 1) * len(events))
        extra = np.column_stack(
            [
                rng.uniform(0, duration if kind == "noise" else duration * 0.1, count),
                rng.integers(width, size=count),
                rng.integers(height, size=count),
                rng.integers(2, size=count),
            ]
        )
        events = np.r_[events, extra]
    elif kind == "deletion":
        events = events[events[:, 1] >= width * 0.3]
    else:
        raise ValueError(kind)
    bins = max(1, int(np.ceil(duration / c["dt"])))
    frames = np.zeros((bins, 2, c["grid"], c["grid"]), np.float32)
    if len(events):
        t = np.minimum((events[:, 0] / c["dt"]).astype(int), bins - 1)
        xx = np.minimum((events[:, 1] / width * c["grid"]).astype(int), c["grid"] - 1)
        yy = np.minimum((events[:, 2] / height * c["grid"]).astype(int), c["grid"] - 1)
        polarity = (events[:, 3] > 0).astype(int)
        if min(t.min(), xx.min(), yy.min()) < 0:
            raise ValueError("Negative event coordinate or timestamp")
        np.add.at(frames, (t, polarity, yy, xx), 1)
    counts = frames.sum((1, 2, 3))
    return torch.from_numpy(frames).to(c["device"]), torch.from_numpy(counts).to(
        c["device"]
    )


def train(a, c):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "train"]
    classes = max(r["label"] for r in records) + 1
    model = Dynamics(classes, c, a.variant).to(c["device"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    rows = []
    for epoch in range(c["epochs"]):
        total = 0.0
        rng.shuffle(records)
        for record in records:
            factor = float(rng.choice(c["rate_factors"]))
            frames, counts = bin_events(record["path"], c, rng, factor)
            result = model(frames, counts, factor if a.variant == "oracle" else None)
            target = torch.tensor([record["label"]], device=c["device"])
            if a.variant == "tet":
                loss = nn.functional.cross_entropy(
                    result["time_logits"], target.expand(len(frames))
                )
            else:
                loss = nn.functional.cross_entropy(result["logits"][None], target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        rows.append(dict(epoch=epoch + 1, loss=total / len(records)))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    table(output / "history.csv", rows)
    torch.save(
        dict(
            model=model.cpu().state_dict(), classes=classes, config=c, variant=a.variant
        ),
        output / "model.pt",
    )


def semantic_mse(reference, changed, factor, dt):
    ref = reference.detach().cpu().numpy()
    other = changed.detach().cpu().numpy()
    time = np.arange(len(other)) * dt * factor
    source = np.arange(len(ref)) * dt
    valid = time <= source[-1]
    if not valid.any():
        return None
    aligned = np.column_stack(
        [np.interp(time[valid], source, ref[:, k]) for k in range(ref.shape[1])]
    )
    return float(np.mean((aligned - other[valid]) ** 2))


def evaluate(a, c):
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = state["config"]
    model = Dynamics(state["classes"], c, state["variant"]).to(c["device"])
    model.load_state_dict(state["model"])
    model.eval()
    rng = np.random.default_rng(c["seed"])
    rows = []
    with torch.no_grad():
        for record in load(Path(a.data) / "manifest.json"):
            if record["split"] != "test":
                continue
            frames, counts = bin_events(record["path"], c, rng, 1.0)
            base = model(frames, counts, 1.0 if state["variant"] == "oracle" else None)
            for factor in c["rate_factors"]:
                altered, changed_counts = bin_events(
                    record["path"], c, rng, factor, a.intervention
                )
                result = model(
                    altered,
                    changed_counts,
                    factor if state["variant"] == "oracle" else None,
                )
                semantic_factor = factor if a.intervention == "timestamp" else 1.0
                rows.append(
                    dict(
                        id=record["id"],
                        method=state["variant"],
                        factor=factor,
                        intervention=a.intervention,
                        correct=bool(result["logits"].argmax() == record["label"]),
                        spike_count=float(result["firing"].sum()),
                        spike_ratio=float(result["firing"].sum() / base["firing"].sum())
                        if float(base["firing"].sum()) > 0
                        else None,
                        state_mse=semantic_mse(
                            base["states"], result["states"], semantic_factor, c["dt"]
                        ),
                        firing_mse=semantic_mse(
                            base["firing"], result["firing"], semantic_factor, c["dt"]
                        ),
                        representation_mse=semantic_mse(
                            base["features"],
                            result["features"],
                            semantic_factor,
                            c["dt"],
                        ),
                    )
                )
    curve = [
        dict(
            factor=f, accuracy=np.mean([r["correct"] for r in rows if r["factor"] == f])
        )
        for f in c["rate_factors"]
    ]
    rates = np.array([r["factor"] for r in curve])
    accuracy = np.array([r["accuracy"] for r in curve])
    order = np.argsort(rates)
    area = (
        float(
            np.sum(
                (accuracy[order][1:] + accuracy[order][:-1])
                * 0.5
                * np.diff(np.log(rates[order]))
            )
            / (np.log(rates.max()) - np.log(rates.min()))
        )
        if len(rates) > 1
        else float(accuracy[0])
    )
    summary = dict(
        method=state["variant"],
        intervention=a.intervention,
        clean_accuracy=np.mean([r["correct"] for r in rows if r["factor"] == 1]),
        rate_auc=area,
        spike_count=np.mean([r["spike_count"] for r in rows]),
        spike_ratio=np.mean(
            [r["spike_ratio"] for r in rows if r["spike_ratio"] is not None]
        )
        if any(r["spike_ratio"] is not None for r in rows)
        else None,
        state_mse=np.mean([r["state_mse"] for r in rows]),
        firing_mse=np.mean([r["firing_mse"] for r in rows]),
        representation_mse=np.mean([r["representation_mse"] for r in rows]),
    )
    table(Path(a.output) / "samples.csv", rows)
    table(Path(a.output) / "rate_curve.csv", curve)
    save(Path(a.output) / "metrics.json", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/events")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--checkpoint")
    p.add_argument("--inputs", nargs="*")
    p.add_argument(
        "--variant",
        choices=[
            "rssd",
            "lif",
            "plif",
            "tet",
            "explicit_norm",
            "decay_only",
            "threshold_only",
            "drive_only",
            "rate_permuted",
            "oracle",
        ],
        default="rssd",
    )
    p.add_argument(
        "--intervention",
        choices=[
            "timestamp",
            "thinning",
            "superposition",
            "noise",
            "burst",
            "deletion",
        ],
        default="timestamp",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
