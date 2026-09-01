from __future__ import annotations
import torch

RAW_PHORE_TYPE_NAMES = {
    1: "donor",
    2: "aromatic",
    3: "positive_ionizable",
    4: "acceptor",
    5: "hydrophobic",
    6: "negative_ionizable",
}
RAW_PHORE_TYPES = tuple(RAW_PHORE_TYPE_NAMES)
NUM_PHORE_TYPES = len(RAW_PHORE_TYPES)
RAW_TO_MODEL = {raw: index for index, raw in enumerate(RAW_PHORE_TYPES)}
MODEL_TO_RAW = {index: raw for raw, index in RAW_TO_MODEL.items()}
DIRECTIONAL_RAW_PHORE_TYPES = frozenset({1, 2, 4})

NUM_ANCHOR_TYPES = 7
PHORE_DIRECTION_TYPE_DIM = 16
PHORE_DIRECTION_ANCHOR_TOPK = 12
PHORE_DIRECTION_DISTANCE_SCALE = 5.0
PHORE_ATOM_RBF_DIM = 16
PHORE_ATOM_DISTANCE_SCALE = 6.0


def raw_to_model_phore_types(labels: torch.Tensor) -> torch.Tensor:

    labels = labels.long()
    if labels.numel() == 0:
        return labels
    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    if minimum < RAW_PHORE_TYPES[0] or maximum > RAW_PHORE_TYPES[-1]:
        raise ValueError(
            "Expected processed pharmacophore labels in [1, 6], got [%d, %d]"
            % (minimum, maximum)
        )
    return labels - 1


def model_to_raw_phore_types(labels: torch.Tensor) -> torch.Tensor:

    labels = labels.long()
    if labels.numel() == 0:
        return labels
    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    if minimum < 0 or maximum >= NUM_PHORE_TYPES:
        raise ValueError(
            "Expected model pharmacophore labels in [0, %d], got [%d, %d]"
            % (NUM_PHORE_TYPES - 1, minimum, maximum)
        )
    return labels + 1
