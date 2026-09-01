from __future__ import annotations
import torch
try:
    from torch_geometric.nn.pool import knn_graph as _pyg_knn_graph
except (ImportError, OSError):  
    _pyg_knn_graph = None


def _torch_knn_graph(
    positions: torch.Tensor,
    k: int,
    batch: torch.Tensor | None,
    loop: bool,
    flow: str,
) -> torch.Tensor:

    if flow not in {"source_to_target", "target_to_source"}:
        raise ValueError(f"Unsupported message flow: {flow}")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if positions.ndim != 2:
        raise ValueError("positions must have shape [num_nodes, num_features]")
    if batch is None:
        batch = torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)
    if batch.ndim != 1 or batch.numel() != positions.size(0):
        raise ValueError("batch must contain one graph index per node")

    targets, sources = [], []
    for graph_id in torch.unique(batch, sorted=True):
        node_ids = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        node_count = int(node_ids.numel())
        neighbor_count = min(k, node_count if loop else node_count - 1)
        if neighbor_count <= 0:
            continue

        distances = torch.cdist(positions[node_ids], positions[node_ids])
        if not loop:
            distances.fill_diagonal_(float("inf"))
        neighbor_local = distances.topk(
            neighbor_count, dim=1, largest=False, sorted=True
        ).indices
        targets.append(node_ids.repeat_interleave(neighbor_count))
        sources.append(node_ids[neighbor_local.reshape(-1)])

    if not targets:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)
    target = torch.cat(targets)
    source = torch.cat(sources)
    if flow == "target_to_source":
        return torch.stack((target, source), dim=0)
    return torch.stack((source, target), dim=0)


def knn_graph(
    positions: torch.Tensor,
    k: int,
    batch: torch.Tensor | None = None,
    loop: bool = False,
    flow: str = "source_to_target",
) -> torch.Tensor:
    """Call PyG's k-NN implementation or a deterministic PyTorch fallback."""
    if _pyg_knn_graph is not None:
        try:
            return _pyg_knn_graph(
                positions, k=k, batch=batch, loop=loop, flow=flow
            )
        except (ImportError, OSError):
            pass
    return _torch_knn_graph(positions, k, batch, loop, flow)
