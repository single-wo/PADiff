import torch
import torch.nn as nn
from torch.nn import Linear, Module, ModuleList

from models.layers import GaussianSmearing, MLP
from utils.scatter import scatter_sum


class NodeBlock(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)
        if self.use_gate:
            self.gate = MLP(edge_dim + node_dim + 1, hidden_dim, hidden_dim)
        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)

    def forward(self, node_features, edge_index, edge_features, node_time):
        num_nodes = node_features.size(0)
        target, source = edge_index
        node_hidden = self.node_net(node_features)
        edge_hidden = self.edge_net(edge_features)
        messages = self.msg_net(edge_hidden * node_hidden[source])
        if self.use_gate:
            gate = self.gate(torch.cat([
                edge_features,
                node_features[source],
                node_time[source],
            ], dim=-1))
            messages = messages * torch.sigmoid(gate)
        aggregated = scatter_sum(
            messages, target, dim=0, dim_size=num_nodes
        )
        output = self.centroid_lin(node_features) + aggregated
        output = self.layer_norm(output)
        return self.out_transform(self.act(output))


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim + node_dim + 1, out_dim, 32)

    def forward(self, bond_features, node_features, time_features):
        interaction = (
            self.bond_linear(bond_features)
            * self.node_linear(node_features)
        )
        interaction = self.inter_module(interaction)
        if self.use_gate:
            gate = self.gate(torch.cat([
                bond_features, node_features, time_features
            ], dim=-1))
            interaction = interaction * torch.sigmoid(gate)
        return interaction


class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True):
        super().__init__()
        intermediate_dim = edge_dim * 2 if hidden_dim is None else hidden_dim
        self.bond_ffn_left = BondFFN(
            edge_dim, node_dim, intermediate_dim, use_gate
        )
        self.bond_ffn_right = BondFFN(
            edge_dim, node_dim, intermediate_dim, use_gate
        )
        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)
        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

    def forward(self, edge_features, edge_index, node_features, edge_time):
        num_nodes = node_features.size(0)
        left_node, right_node = edge_index
        left_messages = self.bond_ffn_left(
            edge_features, node_features[left_node], edge_time
        )
        left_messages = scatter_sum(
            left_messages, right_node, dim=0, dim_size=num_nodes
        )[left_node]
        right_messages = self.bond_ffn_right(
            edge_features, node_features[right_node], edge_time
        )
        right_messages = scatter_sum(
            right_messages, left_node, dim=0, dim_size=num_nodes
        )[right_node]
        edge_features = (
            left_messages
            + right_messages
            + self.node_ffn_left(node_features[left_node])
            + self.node_ffn_right(node_features[right_node])
            + self.self_ffn(edge_features)
        )
        edge_features = self.layer_norm(edge_features)
        return self.out_transform(self.act(edge_features))


class CoordinateUpdate(Module):
    """Equivariant coordinate update driven by scalar edge messages."""

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.edge_lin = BondFFN(
            edge_dim, edge_dim, node_dim, use_gate, out_dim=1
        )

    def forward(
        self,
        node_features,
        edge_features,
        edge_index,
        relative_vectors,
        distances,
        edge_time,
    ):
        left_node, right_node = edge_index
        left_features = self.left_lin_edge(node_features[left_node])
        right_features = self.right_lin_edge(node_features[right_node])
        edge_weights = self.edge_lin(
            edge_features, left_features * right_features, edge_time
        )
        safe_distance = distances.clamp_min(1e-8).unsqueeze(-1)
        edge_forces = (
            edge_weights * relative_vectors
            / safe_distance
            / (safe_distance + 1.0)
        )
        return scatter_sum(
            edge_forces,
            left_node,
            dim=0,
            dim_size=node_features.shape[0],
        )


class EGNNDenoiser(Module):
    """E(3)-equivariant joint node, edge, and coordinate denoiser."""

    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs
        num_gaussians = kwargs.get("num_gaussians", 16)
        start = kwargs.get("start", 0)
        self.distance_expansion = GaussianSmearing(
            start=start, stop=cutoff, num_gaussians=num_gaussians
        )
        self.update_edge = kwargs.get("update_edge", True)
        self.update_pos = kwargs.get("update_pos", True)
        input_edge_dim = (
            edge_dim + num_gaussians if self.update_edge else num_gaussians
        )

        self.node_blocks_with_edge = ModuleList()
        self.edge_embs = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=node_dim,
                use_gate=use_gate,
            ))
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim,
                    node_dim=node_dim,
                    use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(CoordinateUpdate(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    hidden_dim=edge_dim,
                    use_gate=use_gate,
                ))

    def forward(
        self,
        node_features,
        node_positions,
        edge_features,
        edge_index,
        node_time,
        edge_time,
        movable_node_mask,
    ):
        for block_index in range(self.num_blocks):
            if self.update_pos or block_index == 0:
                distance_features, relative_vectors, distances = (
                    self._distance_features(node_positions, edge_index)
                )
            edge_input = (
                torch.cat([edge_features, distance_features], dim=-1)
                if self.update_edge else distance_features
            )
            edge_features = self.edge_embs[block_index](edge_input)
            node_update = self.node_blocks_with_edge[block_index](
                node_features, edge_index, edge_features, node_time
            )
            if self.update_edge:
                edge_features = edge_features + self.edge_blocks[block_index](
                    edge_features, edge_index, node_features, edge_time
                )
            node_features = node_features + node_update
            if self.update_pos:
                position_update = self.pos_blocks[block_index](
                    node_features,
                    edge_features,
                    edge_index,
                    relative_vectors,
                    distances,
                    edge_time,
                )
                node_positions = (
                    node_positions
                    + position_update * movable_node_mask[:, None]
                )
        return node_features, node_positions, edge_features

    def _distance_features(self, positions, edge_index):
        relative_vectors = positions[edge_index[0]] - positions[edge_index[1]]
        distances = torch.linalg.vector_norm(relative_vectors, dim=-1)
        return (
            self.distance_expansion(distances),
            relative_vectors,
            distances,
        )
