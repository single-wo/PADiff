import math
import torch
import torch.nn as nn


ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Feed-forward network with a shared hidden width."""

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        num_layer=2,
        norm=True,
        act_fn="relu",
        act_last=False,
    ):
        super().__init__()
        if num_layer < 1:
            raise ValueError("num_layer must be at least 1")
        if act_fn not in ACTIVATIONS:
            raise ValueError("unknown activation: %s" % act_fn)

        dimensions = [in_dim] + [hidden_dim] * (num_layer - 1) + [out_dim]
        layers = []
        for layer_index, (left, right) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(left, right))
            is_last = layer_index == num_layer - 1
            if not is_last or act_last:
                if norm:
                    layers.append(nn.LayerNorm(right))
                layers.append(ACTIVATIONS[act_fn]())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class GaussianSmearing(nn.Module):
    """Expand scalar distances over fixed Gaussian radial basis functions."""

    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, type_="exp"):
        super().__init__()
        if num_gaussians < 2:
            raise ValueError("num_gaussians must be at least 2")
        self.start = float(start)
        self.stop = float(stop)
        if type_ == "exp":
            offset = torch.exp(
                torch.linspace(
                    start=math.log(start + 1),
                    end=math.log(stop + 1),
                    steps=num_gaussians,
                )
            ) - 1
        elif type_ == "linear":
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise ValueError("type_ must be 'exp' or 'linear'")
        differences = torch.diff(offset)
        differences = torch.cat([differences[:1], differences])
        self.register_buffer("coeff", -0.5 / differences.square())
        self.register_buffer("offset", offset)

    def forward(self, distances):
        distances = distances.clamp(min=self.start, max=self.stop)
        differences = distances.reshape(-1, 1) - self.offset.reshape(1, -1)
        return torch.exp(self.coeff * differences.square())
