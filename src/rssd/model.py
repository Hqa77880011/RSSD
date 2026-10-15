import torch
from torch import nn


class Dynamics(nn.Module):
    def __init__(self, classes, c, variant):
        super().__init__()
        self.c = c
        self.variant = variant
        self.projection = nn.Linear(2 * c["grid"] ** 2, c["width"])
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(
            torch.tensor(float(c["tau0"])).log(), requires_grad=variant == "plif"
        )
        self.head = nn.Linear(c["width"], classes)

    def forward(self, frames, counts, oracle=None):
        c = self.c
        v = frames.new_zeros(c["width"])
        rate = frames.new_tensor(c["reference_rate"])
        beta = self.beta_raw.sigmoid()
        u_sequence = []
        spikes = []
        features = []
        ratios = []
        for t, (frame, count) in enumerate(zip(frames, counts)):
            rate = (1 - c["rate_ema"]) * rate + c["rate_ema"] * count / c["dt"]
            rho = (rate / c["reference_rate"]).clamp(*c["rate_clip"])
            if oracle is not None:
                rho = rho.new_tensor(oracle)
            if self.variant == "rate_permuted":
                rho = frames.new_tensor(c["rate_clip"][0] + c["rate_clip"][1]) - rho
            coupled = self.variant in ("rssd", "oracle", "rate_permuted")
            decay_rho = (
                rho if coupled or self.variant == "decay_only" else rho.new_tensor(1.0)
            )
            alpha = torch.exp(-decay_rho * c["dt"] / self.log_tau.exp())
            scale = (
                rho**beta
                if coupled
                else rho
                if self.variant == "drive_only"
                else rho.new_tensor(1.0)
            )
            if self.variant == "explicit_norm":
                scale = (count / c["dt"] / c["reference_rate"]).clamp(*c["rate_clip"])
            threshold = c["threshold0"] * (
                rho ** (1 - beta)
                if coupled
                else rho
                if self.variant == "threshold_only"
                else rho.new_tensor(1.0)
            )
            current = self.projection(frame.flatten() / c["dt"] / c["reference_rate"])
            v = alpha * v + (1 - alpha) * current / scale
            normalized = v / threshold
            surrogate = torch.sigmoid(5 * (normalized - 1))
            spike = (
                (normalized >= 1).to(normalized.dtype) + surrogate - surrogate.detach()
            )
            u_sequence.append(normalized)
            spikes.append(spike)
            features.append(normalized.relu())
            ratios.append(rho)
            if c["reset"] == "zero":
                v = v * (1 - spike)
            elif c["reset"] == "subtractive":
                v = v - threshold * spike
            else:
                raise ValueError("Unknown reset")
        states = torch.stack(u_sequence)
        firing = torch.stack(spikes)
        representation = torch.stack(features)
        logits = self.head(representation.mean(0))
        return dict(
            logits=logits,
            states=states,
            firing=firing,
            features=representation,
            ratio=torch.stack(ratios),
            time_logits=self.head(representation),
        )
