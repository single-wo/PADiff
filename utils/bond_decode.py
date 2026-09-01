from __future__ import annotations
import numpy as np
from rdkit import Chem


BOND_ORDER_X2 = {1: 2, 2: 4, 3: 6, 4: 3}
MAX_VALENCE_X2 = {
    1: 2, 5: 6, 6: 8, 7: 6, 8: 4, 9: 2,
    15: 10, 16: 12, 17: 2, 35: 2, 53: 2,
}


def max_valence_x2(atomic_number):
    return int(MAX_VALENCE_X2.get(int(atomic_number), 8))


def _unique_edges(bond_index, bond_type, bond_prob=None):
    bond_index = np.asarray(bond_index, dtype=np.int64)
    bond_type = np.asarray(bond_type, dtype=np.int64)
    probabilities = (
        np.ones(len(bond_type), dtype=np.float64)
        if bond_prob is None else np.asarray(bond_prob, dtype=np.float64)
    )
    unique = {}
    if bond_index.size == 0:
        return []
    for i, j, bond, probability in zip(
        bond_index[0], bond_index[1], bond_type, probabilities
    ):
        i, j, bond = int(i), int(j), int(bond)
        if i == j or bond not in BOND_ORDER_X2:
            continue
        if i > j:
            i, j = j, i
        old = unique.get((i, j))
        if old is None or float(probability) > old[1]:
            unique[(i, j)] = (bond, float(probability))
    return [
        (i, j, bond, probability)
        for (i, j), (bond, probability) in unique.items()
    ]


def _duplicate_edges(edges):
    if not edges:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    starts = [edge[0] for edge in edges]
    ends = [edge[1] for edge in edges]
    bond_type = [edge[2] for edge in edges]
    bond_prob = [edge[3] for edge in edges]
    return (
        np.asarray([starts + ends, ends + starts], dtype=np.int64),
        np.asarray(bond_type + bond_type, dtype=np.int64),
        np.asarray(bond_prob + bond_prob, dtype=np.float64),
    )


def _aromatic_cycle_edges(edges):
    all_edges = {
        (min(i, j), max(i, j))
        for i, j, bond, _ in edges if int(bond) in BOND_ORDER_X2
    }
    aromatic_edges = {
        (min(i, j), max(i, j))
        for i, j, bond, _ in edges if int(bond) == 4
    }
    adjacency = {}
    for i, j in all_edges:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    in_cycle = set()
    for edge_i, edge_j in aromatic_edges:
        stack, visited = [edge_i], {edge_i}
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, ()):
                edge = (min(node, neighbor), max(node, neighbor))
                if edge == (edge_i, edge_j):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if edge_j in visited:
            in_cycle.add((edge_i, edge_j))
    return in_cycle


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(int(size)))

    def find(self, item):
        item = int(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        self.parent[left] = right
        return True


def valence_statistics(elements, bond_index, bond_type):
    elements = np.asarray(elements, dtype=np.int64)
    usage = np.zeros(len(elements), dtype=np.int64)
    for i, j, bond, _ in _unique_edges(bond_index, bond_type):
        order = BOND_ORDER_X2[bond]
        if 0 <= i < len(usage) and 0 <= j < len(usage):
            usage[i] += order
            usage[j] += order
    caps = np.asarray([max_valence_x2(z) for z in elements], dtype=np.int64)
    over = usage > caps
    return {
        "valence_usage_x2": usage,
        "valence_cap_x2": caps,
        "num_overvalent_atoms": int(over.sum()),
        "overvalent_atom_fraction": float(over.mean()) if len(over) else 0.0,
        "valence_valid": bool(not over.any()),
    }


def _apply_aromatic_policy(edges, atom_is_aromatic, policy):
    policy = str(policy or "cycle").lower()
    if policy not in {"keep", "cycle", "single"}:
        raise ValueError("aromatic_policy must be keep, cycle, or single")
    aromatic_atoms = (
        None if atom_is_aromatic is None
        else np.asarray(atom_is_aromatic, dtype=bool)
    )
    cycle_edges = _aromatic_cycle_edges(edges) if policy == "cycle" else set()
    decoded, demoted = [], 0
    for i, j, bond, probability in edges:
        if bond != 4:
            decoded.append((i, j, bond, probability))
            continue
        compatible = (
            aromatic_atoms is None
            or (aromatic_atoms[i] and aromatic_atoms[j])
        )
        keep = compatible and (
            policy == "keep" or (
                policy == "cycle" and (min(i, j), max(i, j)) in cycle_edges
            )
        )
        if keep:
            decoded.append((i, j, bond, probability))
        else:
            decoded.append((i, j, 1, probability))
            demoted += 1
    return decoded, demoted


def _select_edges_with_valence(elements, edges, connectivity_first=True):
    elements = np.asarray(elements, dtype=np.int64)
    caps = np.asarray([max_valence_x2(z) for z in elements], dtype=np.int64)
    usage = np.zeros(len(elements), dtype=np.int64)
    selected, selected_keys = [], set()
    union_find = _UnionFind(len(elements))
    ordered = sorted(edges, key=lambda edge: edge[3], reverse=True)

    def add(edge, require_component_merge):
        i, j, bond, probability = edge
        key = (min(i, j), max(i, j))
        if key in selected_keys:
            return False
        if require_component_merge and union_find.find(i) == union_find.find(j):
            return False
        available = min(caps[i] - usage[i], caps[j] - usage[j])
        chosen = int(bond)
        if BOND_ORDER_X2[chosen] > available:
            chosen = 1
        order = BOND_ORDER_X2[chosen]
        if order > available:
            return False
        selected.append((i, j, chosen, probability))
        selected_keys.add(key)
        usage[i] += order
        usage[j] += order
        union_find.union(i, j)
        return True

    if connectivity_first:
        for edge in ordered:
            add(edge, require_component_merge=True)
    for edge in ordered:
        add(edge, require_component_merge=False)
    return selected, union_find



def _rdkit_graph(elements, edges, atom_is_aromatic=None):
    aromatic_atoms = (
        np.zeros(len(elements), dtype=bool)
        if atom_is_aromatic is None
        else np.asarray(atom_is_aromatic, dtype=bool)
    )
    aromatic_incident = set()
    for node_i, node_j, bond, _ in edges:
        if int(bond) == 4:
            aromatic_incident.update((int(node_i), int(node_j)))
    rw_mol = Chem.RWMol()
    for atom_index, atomic_number in enumerate(elements):
        atom = Chem.Atom(int(atomic_number))
        atom.SetIsAromatic(bool(
            aromatic_atoms[atom_index] and atom_index in aromatic_incident
        ))
        rw_mol.AddAtom(atom)
    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for node_i, node_j, bond, _ in edges:
        rw_mol.AddBond(int(node_i), int(node_j), bond_types[int(bond)])
        if int(bond) == 4:
            rw_mol.GetBondBetweenAtoms(int(node_i), int(node_j)).SetIsAromatic(True)
    return rw_mol.GetMol()


def _incremental_rdkit_filter(elements, edges, atom_is_aromatic):
    accepted = []
    rejected = 0
    for edge in sorted(edges, key=lambda item: item[3], reverse=True):
        trial = accepted + [edge]
        try:
            mol = _rdkit_graph(elements, trial, atom_is_aromatic)
            mol.UpdatePropertyCache(strict=True)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            rejected += 1
            continue
        accepted.append(edge)
    return accepted, rejected


def _whole_graph_aromatic_repair(elements, edges, atom_is_aromatic):
    stats = {
        "whole_graph_sanitize_attempted": True,
        "initial_sanitize_success": False,
        "sanitize_success": False,
        "kekulize_repair_attempted": False,
        "sanitize_demoted_aromatic_bonds": 0,
        "sanitize_error": None,
    }
    try:
        mol = _rdkit_graph(elements, edges, atom_is_aromatic)
        Chem.SanitizeMol(mol)
        stats["initial_sanitize_success"] = True
        stats["sanitize_success"] = True
        return edges, stats
    except Exception as error:
        stats["sanitize_error"] = "%s: %s" % (type(error).__name__, error)
        is_kekulize = "kekul" in stats["sanitize_error"].lower()
        if not is_kekulize or not any(int(edge[2]) == 4 for edge in edges):
            return edges, stats

    stats["kekulize_repair_attempted"] = True
    repaired = [
        (node_i, node_j, 1 if int(bond) == 4 else int(bond), probability)
        for node_i, node_j, bond, probability in edges
    ]
    stats["sanitize_demoted_aromatic_bonds"] = int(
        sum(int(edge[2]) == 4 for edge in edges)
    )
    try:
        mol = _rdkit_graph(elements, repaired, np.zeros(len(elements), dtype=bool))
        Chem.SanitizeMol(mol)
        stats["sanitize_success"] = True
        stats["sanitize_error"] = None
        return repaired, stats
    except Exception as error:
        stats["sanitize_error"] = "%s: %s" % (type(error).__name__, error)
        return repaired, stats

def decode_bond_probabilities(
    elements,
    halfedge_index,
    edge_probabilities,
    atom_is_aromatic=None,
    aromatic_policy="cycle",
    connectivity_first=True,
    sanitize=True,
):

    elements = np.asarray(elements, dtype=np.int64)
    halfedge_index = np.asarray(halfedge_index, dtype=np.int64)
    probabilities = np.asarray(edge_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 5:
        raise ValueError("edge_probabilities must have shape [num_halfedges, >=5]")

    positive = []
    for edge_id, (i, j) in enumerate(zip(halfedge_index[0], halfedge_index[1])):
        i, j = int(i), int(j)
        if i < 0 or j < 0 or i >= len(elements) or j >= len(elements) or i == j:
            continue
        bond_scores = probabilities[edge_id, 1:5]
        bond = int(np.argmax(bond_scores)) + 1
        bond_probability = float(bond_scores[bond - 1])
        if bond_probability > float(probabilities[edge_id, 0]):
            positive.append((min(i, j), max(i, j), bond, bond_probability))

    positive, aromatic_demoted = _apply_aromatic_policy(
        positive, atom_is_aromatic, aromatic_policy
    )
    selected, _ = _select_edges_with_valence(
        elements, positive, connectivity_first=connectivity_first
    )
    selected, final_aromatic_demoted = _apply_aromatic_policy(
        selected, atom_is_aromatic, aromatic_policy
    )
    aromatic_demoted += final_aromatic_demoted

    incremental_rejected = 0
    sanitize_stats = {
        "whole_graph_sanitize_attempted": False,
        "initial_sanitize_success": None,
        "sanitize_success": None,
        "kekulize_repair_attempted": False,
        "sanitize_demoted_aromatic_bonds": 0,
        "sanitize_error": None,
    }
    if bool(sanitize):
        selected, incremental_rejected = _incremental_rdkit_filter(
            elements, selected, atom_is_aromatic
        )
        selected, sanitize_stats = _whole_graph_aromatic_repair(
            elements, selected, atom_is_aromatic
        )
        aromatic_demoted += int(
            sanitize_stats.get("sanitize_demoted_aromatic_bonds", 0)
        )

    union_find = _UnionFind(len(elements))
    for i, j, _, _ in selected:
        union_find.union(i, j)
    components = (
        len({union_find.find(i) for i in range(len(elements))})
        if len(elements) else 0
    )
    bond_index, bond_type, bond_prob = _duplicate_edges(selected)
    stats = valence_statistics(elements, bond_index, bond_type)
    stats.update({
        "mode": "valence_aware",
        "aromatic_policy": str(aromatic_policy),
        "positive_candidate_bonds": len(positive),
        "selected_bonds": len(selected),
        "removed_or_rejected_bonds": len(positive) - len(selected),
        "incremental_sanitize_rejected_bonds": int(incremental_rejected),
        "demoted_aromatic_bonds": int(aromatic_demoted),
        "final_demoted_aromatic_bonds": int(final_aromatic_demoted),
        "num_components": int(components),
    })
    stats.update(sanitize_stats)
    return bond_index, bond_type, bond_prob, stats
