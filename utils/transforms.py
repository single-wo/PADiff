import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from utils.dataset import ProteinLigandData
import utils.data as utils_data
from utils.bond_decode import decode_bond_probabilities


aromatic_idx = utils_data.atom_families_id["Aromatic"]

# H, B, C, N, O, F, P, S, Cl, Br, I
map_atom_type_only_to_index = {
    1: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    15: 6,
    16: 7,
    17: 8,
    35: 9,
    53: 10,
}

map_atom_type_aromatic_to_index = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12,
    (35, False): 13,
    (35, True): 14,
    (53, False): 15,
    (53, True): 16,
}

map_atom_type_full_to_index = {
    (1, 'S', False): 0,
    (5, 'SP2', False): 1,
    (6, 'SP', False): 2,
    (6, 'SP2', False): 3,
    (6, 'SP2', True): 4,
    (6, 'SP3', False): 5,
    (7, 'SP', False): 6,
    (7, 'SP2', False): 7,
    (7, 'SP2', True): 8,
    (7, 'SP3', False): 9,
    (8, 'SP2', False): 10,
    (8, 'SP2', True): 11,
    (8, 'SP3', False): 12,
    (9, 'SP3', False): 13,
    (15, 'SP2', False): 14,
    (15, 'SP2', True): 15,
    (15, 'SP3', False): 16,
    (15, 'SP3D', False): 17,
    (16, 'SP2', False): 18,
    (16, 'SP2', True): 19,
    (16, 'SP3', False): 20,
    (16, 'SP3D', False): 21,
    (16, 'SP3D2', False): 22,
    (17, 'SP3', False): 23,
    (35, 'SP3', False): 24,
    (53, 'SP3', False): 25,
}

map_index_to_atom_type_only = {v: k for k, v in map_atom_type_only_to_index.items()}
map_index_to_atom_type_aromatic = {v: k for k, v in map_atom_type_aromatic_to_index.items()}
map_index_to_atom_type_full = {v: k for k, v in map_atom_type_full_to_index.items()}

def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == "basic":
        index = map_atom_type_only_to_index[int(atom_num)]
    elif mode == "aromatic":
        if (int(atom_num), bool(is_aromatic)) in map_atom_type_aromatic_to_index:
            index = map_atom_type_aromatic_to_index[(int(atom_num), bool(is_aromatic))]
        else:
            print(int(atom_num), bool(is_aromatic))
            index = map_atom_type_aromatic_to_index[1, False]
    elif mode == "full":
        index = map_atom_type_full_to_index[(int(atom_num), str(hybridization), bool(is_aromatic))]
    else:
        raise ValueError
    return index















class FeatureComplex(object):
    def __init__(self, mode="basic", sample=False):
        super().__init__()
        # H, C, N, O, S, Se
        self.protein_atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])
        self.max_num_aa = 21
        assert mode in ["basic", "aromatic", "full"], "Mode has to be one of basic, aromatic or full!"
        self.mode = mode
        self.sample = sample
        self.ele_to_nodetype = {k[0]: v for k, v in map_atom_type_aromatic_to_index.items()}
        self.nodetype_to_ele = {v: k[0] for k, v in map_atom_type_aromatic_to_index.items()}
        self.follow_batch = ["protein_element", "ligand_element", "ligand_bond_type",
                             "ligand_halfedge_type", "phore_type", "protein_anchor_type"]
        self.exclude_keys = ["ligand_nbh_list", "num_bonds", "num_atoms"]


    @property
    def protein_feat_dim(self):
        protein_feat_dim = self.protein_atomic_numbers.size(0) + self.max_num_aa + 1
        return protein_feat_dim

    @property
    def atom_feat_dim(self):
        if self.mode == "basic":
            ligand_atom_feat_dim = len(map_atom_type_only_to_index)
        elif self.mode == "aromatic":
            ligand_atom_feat_dim = len(map_atom_type_aromatic_to_index)
        elif self.mode == "full":
            ligand_atom_feat_dim = len(map_atom_type_full_to_index)
        return ligand_atom_feat_dim

    @property
    def bond_feat_dim(self):
        bond_feat_dim = len(utils_data.bond_types)
        return bond_feat_dim

    def __call__(self, data: ProteinLigandData):
        data.protein_num_atoms = len(data.protein_element)
        data.ligand_num_atoms = len(data.ligand_element)
        data.ligand_num_bonds = data.ligand_bond_index.size(1) // 2
        element = data.protein_element.view(-1, 1) == self.protein_atomic_numbers.view(1, -1)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feat = x

        element_list = data.ligand_element
        hybrid_list = data.ligand_hybridization
        aromatic_list = [v[aromatic_idx] for v in data.ligand_atom_feature]

        y = torch.tensor(
            [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybrid_list, aromatic_list)]
        )
        data.ligand_atom_feat_full = y
        data.ligand_bond_feat = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.bond_types))

        # build half edge
        if not self.sample:
            edge_type_mat = torch.zeros([data.ligand_num_atoms, data.ligand_num_atoms], dtype=torch.long)
            for i in range(data.ligand_num_bonds * 2):
                edge_type_mat[data.ligand_bond_index[0, i], data.ligand_bond_index[1, i]] = data.ligand_bond_type[i]
            halfedge_index = torch.triu_indices(data.ligand_num_atoms, data.ligand_num_atoms, offset=1)
            halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
            data.ligand_halfedge_index = halfedge_index
            data.ligand_halfedge_type = halfedge_type
            assert (data.ligand_halfedge_type > 0).sum() == data.ligand_num_bonds
            data.num_nodes = data.ligand_num_atoms

        return data

    def decode_output(
        self, pred_node, pred_pos, pred_halfedge, halfedge_index,
    ):
        """Decode atoms and apply the repository's valence-aware bond policy."""
        pred_atom = softmax(pred_node, axis=-1)
        atom_type_all = np.argmax(pred_atom, axis=-1)
        atom_prob_all = np.max(pred_atom, axis=-1)
        isnot_masked_atom = atom_type_all < self.atom_feat_dim
        edge_index_changer = None
        if not isnot_masked_atom.all():
            edge_index_changer = -np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())

        atom_type = atom_type_all[isnot_masked_atom]
        atom_prob = atom_prob_all[isnot_masked_atom]
        if self.mode == "basic":
            atom_descriptors = [
                (map_index_to_atom_type_only[int(i)], False) for i in atom_type
            ]
        elif self.mode == "aromatic":
            atom_descriptors = [
                map_index_to_atom_type_aromatic[int(i)] for i in atom_type
            ]
        else:
            atom_descriptors = [
                (map_index_to_atom_type_full[int(i)][0],
                 map_index_to_atom_type_full[int(i)][2])
                for i in atom_type
            ]
        element = np.asarray([item[0] for item in atom_descriptors], dtype=np.int64)
        atom_is_aromatic = np.asarray(
            [item[1] for item in atom_descriptors], dtype=bool
        )
        atom_pos = pred_pos[isnot_masked_atom]

        result = {
            "element": element,
            "atom_type": atom_type.astype(np.int64),
            "atom_is_aromatic": atom_is_aromatic,
            "atom_pos": atom_pos,
            "atom_prob": atom_prob,
        }
        if self.bond_feat_dim == 1:
            return result

        edge_probability = softmax(pred_halfedge, axis=-1)
        decoded_halfedge_index = np.asarray(halfedge_index, dtype=np.int64)
        if edge_index_changer is not None:
            decoded_halfedge_index = edge_index_changer[decoded_halfedge_index]
            keep_edge = ~(decoded_halfedge_index < 0).any(axis=0)
            decoded_halfedge_index = decoded_halfedge_index[:, keep_edge]
            edge_probability = edge_probability[keep_edge]

        bond_index, bond_type, bond_prob, stats = decode_bond_probabilities(
            elements=element,
            halfedge_index=decoded_halfedge_index,
            edge_probabilities=edge_probability,
            atom_is_aromatic=atom_is_aromatic,
            aromatic_policy="cycle",
            connectivity_first=True,
            sanitize=True,
        )
        result.update({
            "bond_type": bond_type,
            "bond_index": bond_index,
            "bond_prob": bond_prob,
            "bond_decode_stats": stats,
        })
        return result




class RandomRotation(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(M)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q = torch.from_numpy(Q.astype(np.float32))
        for name in ("ligand_pos", "protein_pos", "phore_pos", "protein_anchor_pos"):
            value = getattr(data, name, None)
            if torch.is_tensor(value) and value.numel() > 0:
                setattr(data, name, value @ Q)
        for name in ("phore_vec", "protein_anchor_vec"):
            value = getattr(data, name, None)
            if torch.is_tensor(value) and value.numel() > 0:
                setattr(data, name, F.normalize(value @ Q, dim=-1, eps=1e-8))
        return data

def make_data_placeholder(n_nodes_list=None, device=None):
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }
