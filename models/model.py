from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from models.transition import ContinuousTransition, GeneralCategoricalTransition
from models.egnn import EGNNDenoiser
from models.layers import GaussianSmearing, MLP
from models.diffusion import get_beta_schedule, log_sample_categorical
from utils.geometry import geometry_regularization_losses
from utils.graph import knn_graph
from utils.scatter import scatter_mean, scatter_softmax, scatter_sum
from utils.phore_types import (
    NUM_ANCHOR_TYPES,
    NUM_PHORE_TYPES,
    PHORE_ATOM_DISTANCE_SCALE,
    PHORE_ATOM_RBF_DIM,
    PHORE_DIRECTION_ANCHOR_TOPK,
    PHORE_DIRECTION_DISTANCE_SCALE,
    PHORE_DIRECTION_TYPE_DIM,
    model_to_raw_phore_types,
    raw_to_model_phore_types,
)


PROTEIN_ROLE = 0
ANCHOR_ROLE = 1
PHORE_ROLE = 2
LIGAND_ROLE = 3
NUM_NODE_ROLES = 4
ATOM_MASK_TRANSITION_MODES = {"tomask"}
LOSS_WEIGHTS = {
    "ligand_pos": 1.0,
    "ligand_node": 100.0,
    "ligand_edge": 100.0,
    "ligand_bond_length": 2.0,
    "ligand_internal_clash": 1.0,
    "ligand_distance_valence": 0.5,
    "pocket_ligand_clash": 0.5,
    "phore_pos": 1.0,
    "phore_type": 100.0,
    "phore_vec": 1.0,
    "phore_atom_assignment": 1.0,
    "phore_atom_center": 1.0,
    "phore_atom_coverage": 0.5,
    "phore_type_compatibility": 1.0,
}


def _get(config, key, default=None):
    return getattr(config, key, default)


def atom_transition_uses_mask(init_prob):
    return init_prob in ATOM_MASK_TRANSITION_MODES


def ligand_atom_class_layout(real_atom_type_count, model_config):
    """Return the exact-count ligand atom layout: real classes plus MASK."""
    real_count = int(real_atom_type_count)
    if real_count <= 0:
        raise ValueError("real_atom_type_count must be positive")
    diff_config = _get(model_config, "diff", None)
    atom_diff_config = _get(diff_config, "diff_atom", None)
    has_mask = atom_transition_uses_mask(
        _get(atom_diff_config, "init_prob", None)
    )
    return {
        "real_atom_type_count": real_count,
        "mask_atom_class": real_count if has_mask else None,
        "ligand_node_types": real_count + int(has_mask),
    }


class PADiff(Module):

    def __init__(
        self,
        config,
        protein_node_types,
        ligand_node_types,
        num_edge_types,
        num_phore_types=None,
        num_anchor_types=None,
        real_atom_type_count=None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.protein_node_types = int(protein_node_types)
        self.ligand_node_types = int(ligand_node_types)
        atom_init_prob = _get(config.diff.diff_atom, "init_prob", None)
        if real_atom_type_count is None:
            real_atom_type_count = self.ligand_node_types - int(
                atom_transition_uses_mask(atom_init_prob)
            )
        self.real_atom_type_count = int(real_atom_type_count)
        expected_layout = ligand_atom_class_layout(
            self.real_atom_type_count, config
        )
        if expected_layout["ligand_node_types"] != self.ligand_node_types:
            raise ValueError(
                "ligand_node_types does not match the configured real+MASK layout"
            )
        self.mask_atom_class = expected_layout["mask_atom_class"]
        self.num_edge_types = int(num_edge_types)
        self.use_pharmacophore = bool(_get(config, "use_pharmacophore", True))
        configured_phore_types = (
            num_phore_types if num_phore_types is not None
            else _get(config, "num_phore_types", None)
        )

        self.legacy_phore_schema = configured_phore_types == 11
        self.num_phore_types = int(
            configured_phore_types
            if configured_phore_types is not None else NUM_PHORE_TYPES
        )
        if self.num_phore_types not in {NUM_PHORE_TYPES, 11}:
            raise ValueError(
                "PADiff supports the compact six-class schema or legacy 11-class checkpoints"
            )
        if self.num_phore_types == 11:
            self.legacy_phore_schema = True
        self.num_anchor_types = int(
            num_anchor_types if num_anchor_types is not None
            else _get(config, "num_anchor_types", NUM_ANCHOR_TYPES)
        )
        self.k = int(_get(config, "knn", 32))
        self.cutoff_mode = _get(config, "cutoff_mode", "knn")
        self.center_pos_mode = _get(config, "center_pos_mode", "protein")
        self.condition_on_pocket_only = bool(
            _get(config, "condition_on_pocket_only", False)
        )
        self.define_betas_alphas(config.diff)

        node_dim = int(config.node_dim)
        edge_dim = int(config.edge_dim)
        time_dim = int(config.diff.time_dim)
        role_dim = NUM_NODE_ROLES
        content_dim = node_dim - role_dim
        node_base_dim = content_dim - time_dim
        edge_base_dim = edge_dim - time_dim
        if node_base_dim <= 0 or edge_base_dim <= 0:
            raise ValueError("node_dim/edge_dim must be larger than role/time dimensions")

        self.protein_node_embedder = nn.Linear(self.protein_node_types, content_dim, bias=False)
        self.anchor_type_embedder = nn.Embedding(self.num_anchor_types, content_dim)
        self.anchor_scalar_embedder = nn.Sequential(
            nn.Linear(2, content_dim), nn.SiLU(), nn.Linear(content_dim, content_dim)
        )

        self.ligand_node_embedder = nn.Linear(self.ligand_node_types, node_base_dim, bias=False)
        self.phore_node_embedder = nn.Linear(self.num_phore_types, node_base_dim, bias=False)
        self.time_emb = GaussianSmearing(
            stop=self.num_timesteps, num_gaussians=time_dim, type_="linear"
        )

        self.spatial_edge_embedder = nn.Linear(NUM_NODE_ROLES ** 2, edge_base_dim, bias=False)
        self.ligand_edge_embedder = nn.Linear(self.num_edge_types, edge_base_dim, bias=False)

        if config.denoiser.backbone != "EGNN":
            raise NotImplementedError(config.denoiser.backbone)
        self.denoiser = EGNNDenoiser(node_dim, edge_dim, **config.denoiser)

        self.ligand_node_decoder = MLP(node_dim, self.ligand_node_types, node_dim)
        self.ligand_edge_decoder = MLP(edge_dim, self.num_edge_types, edge_dim)
        self.phore_node_decoder = MLP(node_dim, self.num_phore_types, node_dim)

        realization_rbf_dim = PHORE_ATOM_RBF_DIM
        realization_hidden = node_dim
        realization_cutoff = PHORE_ATOM_DISTANCE_SCALE
        self.phore_atom_rbf = GaussianSmearing(
            stop=realization_cutoff,
            num_gaussians=realization_rbf_dim,
        )
        pair_input_dim = (
            node_dim * 2 + self.num_phore_types + self.ligand_node_types
            + realization_rbf_dim
        )
        self.phore_atom_pair_head = MLP(
            pair_input_dim, 1, realization_hidden
        )

        self.atom_phore_capability_head = MLP(
            node_dim + self.ligand_node_types,
            self.num_phore_types,
            realization_hidden,
        )

        direction_type_dim = PHORE_DIRECTION_TYPE_DIM
        self.phore_direction_type_embedder = nn.Linear(
            self.num_phore_types, direction_type_dim, bias=False
        )
        direction_pair_dim = node_dim * 2 + direction_type_dim + 3
        self.phore_direction_head = nn.Sequential(
            nn.Linear(direction_pair_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Linear(node_dim, max(node_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(node_dim // 2, 16), 3),
        )
        self.direction_topk = PHORE_DIRECTION_ANCHOR_TOPK
        self.direction_distance_scale = PHORE_DIRECTION_DISTANCE_SCALE

        self.loss_weights = dict(LOSS_WEIGHTS)
        if not self.use_pharmacophore:
            for module in (
                self.phore_node_embedder,
                self.phore_node_decoder,
                self.phore_atom_rbf,
                self.phore_atom_pair_head,
                self.atom_phore_capability_head,
                self.phore_direction_type_embedder,
                self.phore_direction_head,
            ):
                module.requires_grad_(False)

    def define_betas_alphas(self, config):
        self.num_timesteps = int(config.num_timesteps)
        self.categorical_space = _get(config, "categorical_space", "discrete")
        if self.categorical_space != "discrete":
            raise NotImplementedError(
                "PADiff uses discrete categorical diffusion"
            )

        pos_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **config.diff_pos)
        self.pos_transition = ContinuousTransition(pos_betas)
        self.phore_pos_transition = ContinuousTransition(pos_betas)

        atom_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **config.diff_atom)
        self.node_transition = GeneralCategoricalTransition(
            atom_betas,
            self.ligand_node_types,
            init_prob=_get(config.diff_atom, "init_prob", None),
        )

        bond_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **config.diff_bond)
        self.edge_transition = GeneralCategoricalTransition(
            bond_betas,
            self.num_edge_types,
            init_prob=_get(config.diff_bond, "init_prob", None),
        )

        phore_diff = _get(config, "diff_phore", config.diff_atom)
        phore_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **phore_diff)
        phore_init = _get(phore_diff, "init_prob", None)
        if phore_init == "uniform":
            phore_init = None
        self.phore_transition = GeneralCategoricalTransition(
            phore_betas, self.num_phore_types, init_prob=phore_init
        )

    def phore_types_to_model(self, labels):
        """Convert persisted raw labels to the checkpoint's model schema."""
        labels = labels.long()
        if self.legacy_phore_schema:
            return labels
        return raw_to_model_phore_types(labels)

    def phore_types_to_raw(self, labels):
        """Convert model labels to the stable 1..6 reporting schema."""
        labels = labels.long()
        if self.legacy_phore_schema:
            return labels
        return model_to_raw_phore_types(labels)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device
        )
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0
        )[:num_graphs]
        return time_step, torch.ones_like(time_step).float() / self.num_timesteps

    @staticmethod
    def _role_onehot(role, dtype):
        return F.one_hot(role, num_classes=NUM_NODE_ROLES).to(dtype=dtype)

    @staticmethod
    def _empty_anchor_context(protein_pos, protein_batch):
        return (
            protein_batch.new_empty((0,), dtype=torch.long),
            protein_pos.new_empty((0, 3)),
            protein_batch.new_empty((0,), dtype=torch.long),
            protein_pos.new_empty((0,)),
            protein_pos.new_empty((0, 3)),
        )

    def _maybe_disable_anchors(
        self, protein_pos, protein_batch, anchor_type, anchor_pos,
        anchor_batch, anchor_confidence, anchor_vec,
    ):
        if self.condition_on_pocket_only:
            return self._empty_anchor_context(protein_pos, protein_batch)
        return anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec

    def _center_context(
        self,
        protein_pos,
        protein_batch,
        ligand_pos,
        ligand_batch,
        phore_pos,
        phore_batch,
        anchor_pos,
        anchor_batch,
        num_graphs,
    ):
        if self.center_pos_mode == "protein":
            offset = scatter_mean(
                protein_pos, protein_batch, dim=0, dim_size=num_graphs
            )
        elif self.center_pos_mode == "none":
            offset = protein_pos.new_zeros((num_graphs, 3))
        else:
            raise NotImplementedError(self.center_pos_mode)
        return (
            protein_pos - offset[protein_batch],
            ligand_pos - offset[ligand_batch],
            phore_pos - offset[phore_batch],
            anchor_pos - offset[anchor_batch],
            offset,
        )

    def _compose_nodes(
        self,
        ligand_h,
        ligand_pos,
        ligand_batch,
        phore_h,
        phore_pos,
        phore_batch,
        anchor_h,
        anchor_pos,
        anchor_batch,
        protein_h,
        protein_pos,
        protein_batch,
    ):

        raw_h = torch.cat([ligand_h, phore_h, anchor_h, protein_h], dim=0)
        raw_pos = torch.cat([ligand_pos, phore_pos, anchor_pos, protein_pos], dim=0)
        raw_batch = torch.cat([ligand_batch, phore_batch, anchor_batch, protein_batch], dim=0)
        raw_role = torch.cat([
            ligand_batch.new_full((ligand_h.size(0),), LIGAND_ROLE),
            phore_batch.new_full((phore_h.size(0),), PHORE_ROLE),
            anchor_batch.new_full((anchor_h.size(0),), ANCHOR_ROLE),
            protein_batch.new_full((protein_h.size(0),), PROTEIN_ROLE),
        ])
        sort_idx = torch.sort(raw_batch, stable=True).indices
        inverse = torch.empty_like(sort_idx)
        inverse[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

        n_ligand = ligand_h.size(0)
        n_phore = phore_h.size(0)
        n_anchor = anchor_h.size(0)
        ligand_map = inverse[:n_ligand]
        phore_map = inverse[n_ligand:n_ligand + n_phore]
        anchor_map = inverse[n_ligand + n_phore:n_ligand + n_phore + n_anchor]
        protein_map = inverse[n_ligand + n_phore + n_anchor:]
        role = raw_role[sort_idx]
        return {
            "h": raw_h[sort_idx],
            "pos": raw_pos[sort_idx],
            "batch": raw_batch[sort_idx],
            "role": role,
            "ligand_mask": role == LIGAND_ROLE,
            "phore_mask": role == PHORE_ROLE,
            "anchor_mask": role == ANCHOR_ROLE,
            "protein_mask": role == PROTEIN_ROLE,
            "generated_mask": (role == LIGAND_ROLE) | (role == PHORE_ROLE),
            "ligand_map": ligand_map,
            "phore_map": phore_map,
            "anchor_map": anchor_map,
            "protein_map": protein_map,
        }

    def _get_spatial_edges(self, pos, batch, role):
        if self.cutoff_mode not in ("knn", "hybrid"):
            raise ValueError(f"Unsupported cutoff mode: {self.cutoff_mode}")
        k = max(1, min(self.k, max(pos.size(0) - 1, 1)))
        edge_index = knn_graph(pos, k=k, batch=batch, flow="target_to_source")
        target_role = role[edge_index[0]]
        source_role = role[edge_index[1]]

        target_is_generated = (target_role == LIGAND_ROLE) | (target_role == PHORE_ROLE)
        source_is_generated = (source_role == LIGAND_ROLE) | (source_role == PHORE_ROLE)
        allowed = target_is_generated
        allowed &= ~(target_is_generated & source_is_generated &
                     (target_role == LIGAND_ROLE) & (source_role == LIGAND_ROLE))
        edge_index = edge_index[:, allowed]
        target_role = target_role[allowed]
        source_role = source_role[allowed]
        pair_type = target_role * NUM_NODE_ROLES + source_role
        pair_h = F.one_hot(pair_type, NUM_NODE_ROLES ** 2).float()
        return edge_index, pair_h

    def _predict_phore_vectors(
        self,
        phore_h,
        phore_pos,
        phore_state,
        phore_batch,
        context_h,
        context_pos,
        context_vec,
        context_confidence,
        context_batch,
    ):
        if phore_pos.numel() == 0:
            return phore_pos.new_zeros((0, 3))
        if context_pos.numel() == 0:
            return F.normalize(phore_pos.new_ones(phore_pos.shape), dim=-1)

        p_indices, c_indices = [], []
        for graph_id in torch.unique(phore_batch).tolist():
            p_graph = torch.nonzero(phore_batch == graph_id, as_tuple=False).flatten()
            c_graph = torch.nonzero(context_batch == graph_id, as_tuple=False).flatten()
            if p_graph.numel() == 0 or c_graph.numel() == 0:
                continue
            distances = torch.cdist(phore_pos[p_graph], context_pos[c_graph])
            topk = min(max(self.direction_topk, 1), c_graph.numel())
            nearest = distances.topk(topk, dim=1, largest=False).indices
            p_indices.append(p_graph[:, None].expand(-1, topk).reshape(-1))
            c_indices.append(c_graph[nearest.reshape(-1)])
        if not p_indices:
            return F.normalize(phore_pos.new_ones(phore_pos.shape), dim=-1)

        p_idx = torch.cat(p_indices)
        c_idx = torch.cat(c_indices)
        delta = context_pos[c_idx] - phore_pos[p_idx]
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        unit_delta = delta / distance
        unit_context_vec = F.normalize(context_vec[c_idx].float(), dim=-1, eps=1e-8)
        confidence = context_confidence[c_idx].float().view(-1, 1)
        alignment = (unit_context_vec * (-unit_delta)).sum(dim=-1, keepdim=True)
        type_h = self.phore_direction_type_embedder(phore_state[p_idx].float())
        geometry = torch.cat([
            distance / self.direction_distance_scale,
            confidence,
            alignment,
        ], dim=-1)
        coeff = self.phore_direction_head(torch.cat([
            phore_h[p_idx], context_h[c_idx], type_h, geometry
        ], dim=-1))
        attention = scatter_softmax(coeff[:, 0], p_idx, dim=0)
        pair_vector = (
            torch.tanh(coeff[:, 1:2]) * unit_delta
            + torch.tanh(coeff[:, 2:3]) * unit_context_vec
        )
        vector = scatter_sum(
            attention[:, None] * pair_vector,
            p_idx,
            dim=0,
            dim_size=phore_pos.size(0),
        )
        fallback = scatter_sum(
            attention[:, None] * unit_delta,
            p_idx,
            dim=0,
            dim_size=phore_pos.size(0),
        )
        use_fallback = vector.norm(dim=-1, keepdim=True) < 1e-6
        vector = torch.where(use_fallback, fallback, vector)
        return F.normalize(vector, dim=-1, eps=1e-8)


    @staticmethod
    def _build_phore_atom_pairs(phore_batch, ligand_batch):
        """Return all within-graph pharmacophore/ligand pairs.

        Pharmacophore counts are small, so using all pairs avoids silently
        dropping a true multi-atom feature from a distance-truncated KNN set.
        """
        phore_indices, atom_indices = [], []
        for graph_id in torch.unique(phore_batch).tolist():
            p_idx = torch.nonzero(phore_batch == graph_id, as_tuple=False).flatten()
            a_idx = torch.nonzero(ligand_batch == graph_id, as_tuple=False).flatten()
            if p_idx.numel() == 0 or a_idx.numel() == 0:
                continue
            phore_indices.append(p_idx[:, None].expand(-1, a_idx.numel()).reshape(-1))
            atom_indices.append(a_idx[None, :].expand(p_idx.numel(), -1).reshape(-1))
        if not phore_indices:
            empty = phore_batch.new_empty((0,), dtype=torch.long)
            return torch.stack([empty, empty], dim=0)
        return torch.stack([torch.cat(phore_indices), torch.cat(atom_indices)], dim=0)

    def _predict_phore_atom_assignments(
        self, phore_h, ligand_h, pred_phore_node, pred_ligand_node,
        pred_phore_pos, pred_ligand_pos, phore_batch, ligand_batch,
    ):
        """Predict type-aware phore-to-atom realization links in one pass."""
        atom_prob = F.softmax(pred_ligand_node, dim=-1)
        phore_prob = F.softmax(pred_phore_node, dim=-1)
        atom_capability_logits = self.atom_phore_capability_head(
            torch.cat([ligand_h, atom_prob], dim=-1)
        )
        pair_index = self._build_phore_atom_pairs(phore_batch, ligand_batch)
        if pair_index.numel() == 0:
            return pair_index, pred_ligand_pos.new_empty((0,)), atom_capability_logits

        ph_idx, atom_idx = pair_index
        distance = (pred_phore_pos[ph_idx] - pred_ligand_pos[atom_idx]).norm(
            dim=-1, keepdim=True
        )
        distance_h = self.phore_atom_rbf(distance)
        relation_logits = self.phore_atom_pair_head(torch.cat([
            phore_h[ph_idx],
            ligand_h[atom_idx],
            phore_prob[ph_idx],
            atom_prob[atom_idx],
            distance_h,
        ], dim=-1)).squeeze(-1)

        compatibility_logits = (
            atom_capability_logits[atom_idx] * phore_prob[ph_idx]
        ).sum(dim=-1)
        return pair_index, relation_logits + compatibility_logits, atom_capability_logits

    def forward(
        self,
        protein_node,
        protein_pos,
        protein_batch,
        anchor_type,
        anchor_pos,
        anchor_batch,
        anchor_confidence,
        anchor_vec,
        phore_node_pert,
        phore_pos_pert,
        phore_batch,
        ligand_node_pert,
        ligand_pos_pert,
        ligand_batch,
        ligand_edge_pert,
        ligand_edge_index,
        ligand_edge_batch,
        t,
    ):

        anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec = self._maybe_disable_anchors(
            protein_pos, protein_batch, anchor_type, anchor_pos, anchor_batch,
            anchor_confidence, anchor_vec,
        )
        time_ligand = self.time_emb(t[ligand_batch])
        time_phore = self.time_emb(t[phore_batch])
        ligand_h = torch.cat([
            self.ligand_node_embedder(ligand_node_pert.float()), time_ligand
        ], dim=-1)
        phore_h = torch.cat([
            self.phore_node_embedder(phore_node_pert.float()), time_phore
        ], dim=-1)
        protein_h = self.protein_node_embedder(protein_node.float())
        dtype = ligand_h.dtype
        protein_h = protein_h.to(dtype)
        anchor_confidence = anchor_confidence.float().view(-1)
        anchor_vec_norm = anchor_vec.float().norm(dim=-1)
        anchor_h = self.anchor_type_embedder(anchor_type.long()) + self.anchor_scalar_embedder(
            torch.stack([anchor_confidence, anchor_vec_norm], dim=-1)
        )
        anchor_h = anchor_h.to(dtype)

        ligand_h = torch.cat([
            ligand_h, self._role_onehot(ligand_batch.new_full((ligand_h.size(0),), LIGAND_ROLE), dtype)
        ], dim=-1)
        phore_h = torch.cat([
            phore_h, self._role_onehot(phore_batch.new_full((phore_h.size(0),), PHORE_ROLE), dtype)
        ], dim=-1)
        anchor_h = torch.cat([
            anchor_h, self._role_onehot(anchor_batch.new_full((anchor_h.size(0),), ANCHOR_ROLE), dtype)
        ], dim=-1)
        protein_h = torch.cat([
            protein_h, self._role_onehot(protein_batch.new_full((protein_h.size(0),), PROTEIN_ROLE), dtype)
        ], dim=-1)

        composed = self._compose_nodes(
            ligand_h, ligand_pos_pert, ligand_batch,
            phore_h, phore_pos_pert, phore_batch,
            anchor_h, anchor_pos, anchor_batch,
            protein_h, protein_pos, protein_batch,
        )
        spatial_edge_index, spatial_pair_h = self._get_spatial_edges(
            composed["pos"], composed["batch"], composed["role"]
        )
        spatial_edge_batch = composed["batch"][spatial_edge_index[0]]
        spatial_edge_h = torch.cat([
            self.spatial_edge_embedder(spatial_pair_h.to(dtype)),
            self.time_emb(t[spatial_edge_batch]),
        ], dim=-1)

        mapped_ligand_edges = composed["ligand_map"][ligand_edge_index]
        explicit_edge_h = torch.cat([
            self.ligand_edge_embedder(ligand_edge_pert.float()),
            self.time_emb(t[ligand_edge_batch]),
        ], dim=-1)
        all_edge_h = torch.cat([explicit_edge_h, spatial_edge_h], dim=0)
        all_edge_index = torch.cat([mapped_ligand_edges, spatial_edge_index], dim=1)
        all_edge_batch = torch.cat([ligand_edge_batch, spatial_edge_batch], dim=0)

        node_h, node_pos, edge_h = self.denoiser(
            node_features=composed["h"],
            node_positions=composed["pos"],
            edge_features=all_edge_h,
            edge_index=all_edge_index,
            node_time=t[composed["batch"]].unsqueeze(-1).float() / self.num_timesteps,
            edge_time=t[all_edge_batch].unsqueeze(-1).float() / self.num_timesteps,
            movable_node_mask=composed["generated_mask"],
        )

        ligand_node_h = node_h[composed["ligand_mask"]]
        phore_node_h = node_h[composed["phore_mask"]]
        pred_ligand_node = self.ligand_node_decoder(ligand_node_h)
        pred_phore_node = (
            self.phore_node_decoder(phore_node_h)
            if self.use_pharmacophore
            else ligand_node_h.new_empty((0, self.num_phore_types))
        )
        pred_ligand_pos = node_pos[composed["ligand_mask"]]
        pred_phore_pos = node_pos[composed["phore_mask"]]

        n_directed_ligand_edges = ligand_edge_index.size(1)
        ligand_edge_h = edge_h[:n_directed_ligand_edges]
        n_halfedges = n_directed_ligand_edges // 2
        pred_ligand_halfedge = self.ligand_edge_decoder(
            ligand_edge_h[:n_halfedges] + ligand_edge_h[n_halfedges:]
        )

        if self.use_pharmacophore:
            phore_atom_index, pred_phore_atom_logits, pred_atom_phore_logits = (
                self._predict_phore_atom_assignments(
                    phore_node_h, ligand_node_h, pred_phore_node, pred_ligand_node,
                    pred_phore_pos, pred_ligand_pos, phore_batch, ligand_batch,
                )
            )
        else:
            empty_index = ligand_batch.new_empty((0,), dtype=torch.long)
            phore_atom_index = torch.stack([empty_index, empty_index], dim=0)
            pred_phore_atom_logits = pred_ligand_pos.new_empty((0,))
            pred_atom_phore_logits = pred_ligand_pos.new_empty(
                (pred_ligand_pos.size(0), 0)
            )

        if anchor_pos.numel() > 0:
            direction_context_h = node_h[composed["anchor_mask"]]
            direction_context_pos = node_pos[composed["anchor_mask"]]
            direction_context_vec = anchor_vec
            direction_context_conf = anchor_confidence
            direction_context_batch = anchor_batch
        else:
            direction_context_h = node_h[composed["protein_mask"]]
            direction_context_pos = node_pos[composed["protein_mask"]]
            direction_context_vec = torch.zeros_like(direction_context_pos)
            direction_context_conf = torch.ones(
                direction_context_pos.size(0), device=direction_context_pos.device
            )
            direction_context_batch = protein_batch
        pred_phore_vec = (
            self._predict_phore_vectors(
                phore_node_h,
                pred_phore_pos,
                phore_node_pert,
                phore_batch,
                direction_context_h,
                direction_context_pos,
                direction_context_vec,
                direction_context_conf,
                direction_context_batch,
            )
            if self.use_pharmacophore else pred_ligand_pos.new_empty((0, 3))
        )
        return {
            "pred_ligand_node": pred_ligand_node,
            "pred_ligand_pos": pred_ligand_pos,
            "pred_ligand_halfedge": pred_ligand_halfedge,
            "pred_phore_node": pred_phore_node,
            "pred_phore_pos": pred_phore_pos,
            "pred_phore_vec": pred_phore_vec,
            "phore_atom_index": phore_atom_index,
            "pred_phore_atom_logits": pred_phore_atom_logits,
            "pred_atom_phore_logits": pred_atom_phore_logits,
        }

    @staticmethod
    def _categorical_loss(transition, pred_logits, log_vt, log_v0, time_step, batch):
        log_recon = F.log_softmax(pred_logits, dim=-1)
        log_post_true = transition.q_v_posterior(
            log_v0, log_vt, time_step, batch, v0_prob=True
        )
        log_post_pred = transition.q_v_posterior(
            log_recon, log_vt, time_step, batch, v0_prob=True
        )
        return transition.compute_v_Lt(
            log_post_true, log_post_pred, log_v0, t=time_step, batch=batch
        ).mean().clamp_min(0.0)

    def _chemical_phore_targets(self, ligand_atom_feature, reference):
        """Map RDKit BaseFeatures columns to the active checkpoint schema."""
        target = reference.new_zeros(reference.shape)
        if ligand_atom_feature is None or ligand_atom_feature.numel() == 0:
            return target
        feature = ligand_atom_feature.to(
            device=reference.device, dtype=reference.dtype
        )
        if feature.dim() != 2 or feature.size(0) != reference.size(0):
            raise ValueError(
                "ligand_atom_feature must have shape [num_ligand_atoms, num_features]"
            )

        index = (lambda raw: raw if self.legacy_phore_schema else raw - 1)
        if feature.size(1) > 0:
            target[:, index(4)] = feature[:, 0]
        if feature.size(1) > 1:
            target[:, index(1)] = feature[:, 1]
        if feature.size(1) > 2:
            target[:, index(2)] = feature[:, 2]
        if feature.size(1) > 3:
            target[:, index(5)] = feature[:, 3]
        if feature.size(1) > 4:
            hydrophobic = index(5)
            target[:, hydrophobic] = torch.maximum(
                target[:, hydrophobic], feature[:, 4]
            )
        if feature.size(1) > 5:
            target[:, index(6)] = feature[:, 5]
        if feature.size(1) > 6:
            target[:, index(3)] = feature[:, 6]
        return target.clamp(0.0, 1.0)

    def get_loss(
        self,
        protein_node,
        protein_pos,
        protein_batch,
        anchor_type,
        anchor_pos,
        anchor_batch,
        anchor_confidence,
        anchor_vec,
        phore_type,
        phore_pos,
        phore_batch,
        phore_vec,
        ligand_node,
        ligand_pos,
        ligand_batch,
        halfedge_type,
        halfedge_index,
        halfedge_batch,
        num_mol,
        ph2atom_edge_index=None,
        ligand_atom_feature=None,
        ligand_element=None,
        time_step=None,
    ):
        num_graphs = int(num_mol)
        device = ligand_pos.device
        if self.use_pharmacophore:
            phore_type = self.phore_types_to_model(phore_type)
        else:
            phore_type = phore_type.new_empty((0,), dtype=torch.long)
            phore_pos = ligand_pos.new_empty((0, 3))
            phore_batch = ligand_batch.new_empty((0,), dtype=torch.long)
            phore_vec = ligand_pos.new_empty((0, 3))
            ph2atom_edge_index = None
        anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec = self._maybe_disable_anchors(
            protein_pos, protein_batch, anchor_type, anchor_pos, anchor_batch,
            anchor_confidence, anchor_vec,
        )
        protein_pos, ligand_pos, phore_pos, anchor_pos, _ = self._center_context(
            protein_pos, protein_batch, ligand_pos, ligand_batch,
            phore_pos, phore_batch, anchor_pos, anchor_batch, num_graphs,
        )
        if time_step is None:
            time_step, _ = self.sample_time(num_graphs, device)
        else:
            time_step = torch.as_tensor(
                time_step, dtype=torch.long, device=device
            ).reshape(-1)
            if time_step.numel() == 1 and num_graphs > 1:
                time_step = time_step.repeat(num_graphs)
            if time_step.numel() != num_graphs:
                raise ValueError(
                    "time_step must contain one value per graph: expected %d, got %d"
                    % (num_graphs, time_step.numel())
                )
            if (
                time_step.numel() > 0
                and (time_step.min().item() < 0
                     or time_step.max().item() >= self.num_timesteps)
            ):
                raise ValueError(
                    "time_step values must be in [0, %d]" %
                    (self.num_timesteps - 1)
                )

        ligand_pos_pert = self.pos_transition.add_noise(ligand_pos, time_step, ligand_batch)
        ligand_node_pert, log_node_t, log_node_0 = self.node_transition.add_noise(
            ligand_node, time_step, ligand_batch
        )

        has_phore = phore_type.numel() > 0
        if has_phore:
            phore_pos_pert = self.phore_pos_transition.add_noise(
                phore_pos, time_step, phore_batch
            )
            phore_node_pert, log_phore_t, log_phore_0 = self.phore_transition.add_noise(
                phore_type, time_step, phore_batch
            )
        else:
            phore_pos_pert = phore_pos
            phore_node_pert = phore_pos.new_zeros((0, self.num_phore_types))
            log_phore_t = phore_pos.new_zeros((0, self.num_phore_types))
            log_phore_0 = phore_pos.new_zeros((0, self.num_phore_types))
        halfedge_pert, log_halfedge_t, log_halfedge_0 = self.edge_transition.add_noise(
            halfedge_type, time_step, halfedge_batch
        )
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        ligand_edge_pert = torch.cat([halfedge_pert, halfedge_pert], dim=0)

        preds = self(
            protein_node, protein_pos, protein_batch,
            anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec,
            phore_node_pert, phore_pos_pert, phore_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, time_step,
        )

        ligand_pos_loss = F.mse_loss(preds["pred_ligand_pos"], ligand_pos)

        geometry_config = _get(self.config, "geometry_regularization", None)
        geometry_losses = geometry_regularization_losses(
            pred_ligand_pos=preds["pred_ligand_pos"],
            target_ligand_pos=ligand_pos,
            halfedge_index=halfedge_index,
            halfedge_type=halfedge_type,
            ligand_batch=ligand_batch,
            protein_pos=protein_pos,
            protein_batch=protein_batch,
            ligand_element=ligand_element,
            bond_smooth_l1_beta=float(_get(
                geometry_config, "bond_smooth_l1_beta", 0.10
            )),
            internal_clash_distance=float(_get(
                geometry_config, "internal_clash_distance", 1.20
            )),
            distance_valence_margin=float(_get(
                geometry_config, "distance_valence_margin", 0.05
            )),
            pocket_clash_distance=float(_get(
                geometry_config, "pocket_clash_distance", 1.50
            )),
            compute_bond_length=self.loss_weights["ligand_bond_length"] > 0.0,
            compute_internal_clash=self.loss_weights["ligand_internal_clash"] > 0.0,
            compute_distance_valence=(
                self.loss_weights["ligand_distance_valence"] > 0.0
            ),
            compute_pocket_clash=self.loss_weights["pocket_ligand_clash"] > 0.0,
        )

        raw_losses = {
            "ligand_pos": ligand_pos_loss,
            "ligand_bond_length": geometry_losses["ligand_bond_length"],
            "ligand_internal_clash": geometry_losses["ligand_internal_clash"],
            "ligand_distance_valence": geometry_losses[
                "ligand_distance_valence"
            ],
            "pocket_ligand_clash": geometry_losses["pocket_ligand_clash"],
            "ligand_node": self._categorical_loss(
                self.node_transition, preds["pred_ligand_node"],
                log_node_t, log_node_0, time_step, ligand_batch,
            ),
            "ligand_edge": self._categorical_loss(
                self.edge_transition, preds["pred_ligand_halfedge"],
                log_halfedge_t, log_halfedge_0, time_step, halfedge_batch,
            ),
            "phore_pos": (
                F.mse_loss(preds["pred_phore_pos"], phore_pos)
                if has_phore else ligand_pos.new_zeros(())
            ),
            "phore_type": (
                self._categorical_loss(
                    self.phore_transition, preds["pred_phore_node"],
                    log_phore_t, log_phore_0, time_step, phore_batch,
                ) if has_phore else ligand_pos.new_zeros(())
            ),

            "phore_vec": (
                (
                    1.0 - F.cosine_similarity(
                        preds["pred_phore_vec"], phore_vec.float(), dim=-1, eps=1e-8
                    ).abs()
                ).mean() if has_phore else ligand_pos.new_zeros(())
            ),
        }

        zero = ligand_pos.new_zeros(())
        pair_index = preds["phore_atom_index"]
        pair_logits = preds["pred_phore_atom_logits"]
        atom_capability_logits = preds["pred_atom_phore_logits"]
        has_assignment = (
            has_phore and ph2atom_edge_index is not None
            and ph2atom_edge_index.numel() > 0 and pair_index.numel() > 0
        )
        if has_assignment:
            true_edges = ph2atom_edge_index.long()
            true_pair_ids = (
                true_edges[0] * ligand_pos.size(0) + true_edges[1]
            )
            candidate_pair_ids = (
                pair_index[0] * ligand_pos.size(0) + pair_index[1]
            )
            assignment_target = torch.isin(
                candidate_pair_ids, true_pair_ids
            ).to(dtype=ligand_pos.dtype)
            num_positive = assignment_target.sum().clamp_min(1.0)
            num_negative = (assignment_target.numel() - assignment_target.sum()).clamp_min(1.0)
            assignment_pos_weight = (num_negative / num_positive).clamp(1.0, 20.0)
            raw_losses["phore_atom_assignment"] = F.binary_cross_entropy_with_logits(
                pair_logits, assignment_target, pos_weight=assignment_pos_weight
            )

            true_ph_idx, true_atom_idx = true_edges
            atom_type_target = self._chemical_phore_targets(
                ligand_atom_feature, atom_capability_logits
            )

            atom_type_target[
                true_atom_idx, phore_type[true_ph_idx].long()
            ] = 1.0
            capability_positive = atom_type_target.sum().clamp_min(1.0)
            capability_negative = (
                atom_type_target.numel() - atom_type_target.sum()
            ).clamp_min(1.0)
            capability_pos_weight = (
                capability_negative / capability_positive
            ).clamp(1.0, 20.0)
            raw_losses["phore_type_compatibility"] = (
                F.binary_cross_entropy_with_logits(
                    atom_capability_logits, atom_type_target,
                    pos_weight=capability_pos_weight,
                )
            )

            ph_idx, atom_idx = pair_index
            assignment_prob = torch.sigmoid(pair_logits)
            assignment_mass = scatter_sum(
                assignment_prob, ph_idx, dim=0, dim_size=phore_pos.size(0)
            ).clamp_min(1e-8)
            soft_center = scatter_sum(
                assignment_prob[:, None] * preds["pred_ligand_pos"][atom_idx],
                ph_idx, dim=0, dim_size=phore_pos.size(0),
            ) / assignment_mass[:, None]
            supervised_phores = torch.zeros(
                phore_pos.size(0), dtype=torch.bool, device=device
            )
            supervised_phores[true_ph_idx] = True
            true_center = scatter_mean(
                preds["pred_ligand_pos"][true_atom_idx],
                true_ph_idx, dim=0, dim_size=phore_pos.size(0),
            )
            hard_center_loss = F.mse_loss(
                preds["pred_phore_pos"][supervised_phores],
                true_center[supervised_phores],
            )
            soft_center_loss = F.mse_loss(
                preds["pred_phore_pos"][supervised_phores],
                soft_center[supervised_phores],
            )

            raw_losses["phore_atom_center"] = (
                hard_center_loss + 0.1 * soft_center_loss
            )

            coverage_terms = []
            for phore_id in torch.nonzero(
                    supervised_phores, as_tuple=False).flatten().tolist():
                phore_pair_logits = pair_logits[ph_idx == phore_id]
                if phore_pair_logits.numel() > 0:
                    coverage_terms.append(F.softplus(-phore_pair_logits.max()))
            raw_losses["phore_atom_coverage"] = (
                torch.stack(coverage_terms).mean() if coverage_terms else zero
            )
        else:
            raw_losses["phore_atom_assignment"] = zero
            raw_losses["phore_atom_center"] = zero
            raw_losses["phore_atom_coverage"] = zero
            raw_losses["phore_type_compatibility"] = zero

        if not self.use_pharmacophore:
            raw_losses = {
                key: value for key, value in raw_losses.items()
                if not key.startswith("phore_")
            }
        weighted = {
            key: value * self.loss_weights[key] for key, value in raw_losses.items()
        }
        total = sum(weighted.values())
        loss_dict = {"loss": total}
        loss_dict.update({f"loss_{key}": value for key, value in weighted.items()})
        return loss_dict, preds

    @torch.no_grad()
    def sample(
        self,
        n_graphs,
        protein_node,
        protein_pos,
        protein_batch,
        anchor_type,
        anchor_pos,
        anchor_batch,
        anchor_confidence,
        anchor_vec,
        ligand_batch,
        halfedge_index,
        halfedge_batch,
        phore_batch,
    ):

        anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec = self._maybe_disable_anchors(
            protein_pos, protein_batch, anchor_type, anchor_pos, anchor_batch,
            anchor_confidence, anchor_vec,
        )
        device = ligand_batch.device
        if not self.use_pharmacophore:
            phore_batch = ligand_batch.new_empty((0,), dtype=torch.long)
        n_ligand = ligand_batch.numel()
        n_phore = phore_batch.numel()
        n_halfedge = halfedge_batch.numel()
        _, ligand_state, log_node_type = self.node_transition.sample_init(n_ligand)
        if n_phore:
            _, phore_state, log_phore_type = self.phore_transition.sample_init(n_phore)
        else:
            phore_state = protein_pos.new_empty((0, self.num_phore_types))
            log_phore_type = protein_pos.new_empty((0, self.num_phore_types))
        _, halfedge_state, log_halfedge_type = self.edge_transition.sample_init(n_halfedge)

        pocket_center = scatter_mean(
            protein_pos, protein_batch, dim=0, dim_size=n_graphs
        )
        ligand_pos = pocket_center[ligand_batch] + torch.randn((n_ligand, 3), device=device)
        phore_pos = pocket_center[phore_batch] + torch.randn((n_phore, 3), device=device)
        protein_pos, ligand_pos, phore_pos, anchor_pos, offset = self._center_context(
            protein_pos, protein_batch, ligand_pos, ligand_batch,
            phore_pos, phore_batch, anchor_pos, anchor_batch, n_graphs,
        )

        ligand_node_traj = ligand_state.new_zeros(
            (self.num_timesteps + 1, n_ligand, ligand_state.size(-1))
        )
        ligand_pos_traj = ligand_pos.new_zeros((self.num_timesteps + 1, n_ligand, 3))
        halfedge_traj = halfedge_state.new_zeros(
            (self.num_timesteps + 1, n_halfedge, halfedge_state.size(-1))
        )
        phore_node_traj = phore_state.new_zeros(
            (self.num_timesteps + 1, n_phore, phore_state.size(-1))
        )
        phore_pos_traj = phore_pos.new_zeros((self.num_timesteps + 1, n_phore, 3))
        phore_vec_traj = phore_pos.new_zeros((self.num_timesteps + 1, n_phore, 3))
        ligand_node_traj[0] = ligand_state
        ligand_pos_traj[0] = ligand_pos + offset[ligand_batch]
        halfedge_traj[0] = halfedge_state
        phore_node_traj[0] = phore_state
        phore_pos_traj[0] = phore_pos + offset[phore_batch]

        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        preds = None
        for i, step in tqdm(
            enumerate(range(self.num_timesteps - 1, -1, -1)),
            total=self.num_timesteps,
            desc=("Joint diffusion" if self.use_pharmacophore else "Ligand diffusion"),
        ):
            time_step = torch.full((n_graphs,), step, dtype=torch.long, device=device)
            ligand_edge_state = torch.cat([halfedge_state, halfedge_state], dim=0)
            preds = self(
                protein_node, protein_pos, protein_batch,
                anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec,
                phore_state, phore_pos, phore_batch,
                ligand_state, ligand_pos, ligand_batch,
                ligand_edge_state, ligand_edge_index, ligand_edge_batch, time_step,
            )

            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                ligand_pos, preds["pred_ligand_pos"], time_step, ligand_batch
            )
            phore_pos_prev = (
                self.phore_pos_transition.get_prev_from_recon(
                    phore_pos, preds["pred_phore_pos"], time_step, phore_batch
                )
                if n_phore else phore_pos
            )
            log_node_type = self.node_transition.q_v_posterior(
                F.log_softmax(preds["pred_ligand_node"], dim=-1),
                log_node_type, time_step, ligand_batch, v0_prob=True,
            )
            ligand_type_prev = log_sample_categorical(log_node_type)
            ligand_state_prev = self.node_transition.onehot_encode(ligand_type_prev)
            if n_phore:
                log_phore_type = self.phore_transition.q_v_posterior(
                    F.log_softmax(preds["pred_phore_node"], dim=-1),
                    log_phore_type, time_step, phore_batch, v0_prob=True,
                )
                phore_type_prev = log_sample_categorical(log_phore_type)
                phore_state_prev = self.phore_transition.onehot_encode(phore_type_prev)
            else:
                phore_state_prev = phore_state
            log_halfedge_type = self.edge_transition.q_v_posterior(
                F.log_softmax(preds["pred_ligand_halfedge"], dim=-1),
                log_halfedge_type, time_step, halfedge_batch, v0_prob=True,
            )
            halfedge_type_prev = log_sample_categorical(log_halfedge_type)
            halfedge_state_prev = self.edge_transition.onehot_encode(halfedge_type_prev)

            ligand_node_traj[i + 1] = ligand_state_prev
            ligand_pos_traj[i + 1] = ligand_pos_prev + offset[ligand_batch]
            halfedge_traj[i + 1] = halfedge_state_prev
            phore_node_traj[i + 1] = phore_state_prev
            phore_pos_traj[i + 1] = phore_pos_prev + offset[phore_batch]
            phore_vec_traj[i + 1] = preds["pred_phore_vec"]
            ligand_state, ligand_pos, halfedge_state = (
                ligand_state_prev, ligand_pos_prev, halfedge_state_prev
            )
            phore_state, phore_pos = phore_state_prev, phore_pos_prev

        return {
            "pred": [
                preds["pred_ligand_node"],
                ligand_pos + offset[ligand_batch],
                preds["pred_ligand_halfedge"],
            ],
            "terminal_ligand_node_state": ligand_state,
            "terminal_ligand_halfedge_state": halfedge_state,
            "phore_pred": [
                preds["pred_phore_node"],
                preds["pred_phore_pos"] + offset[phore_batch],
                preds["pred_phore_vec"],
            ],
            "terminal_phore_node_state": phore_state,
            "phore_atom_assignment": {
                "index": preds["phore_atom_index"],
                "logits": preds["pred_phore_atom_logits"],
                "atom_type_logits": preds["pred_atom_phore_logits"],
            },
            "traj": [ligand_node_traj, ligand_pos_traj, halfedge_traj],
            "phore_traj": [phore_node_traj, phore_pos_traj, phore_vec_traj],
        }
