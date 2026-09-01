from __future__ import annotations
import json
import os
from functools import lru_cache
import numpy as np
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from scipy.optimize import linear_sum_assignment

from utils.phore_types import (
    DIRECTIONAL_RAW_PHORE_TYPES,
    RAW_PHORE_TYPE_NAMES,
)


PHORE_TYPE_NAMES = RAW_PHORE_TYPE_NAMES

RDKIT_FAMILY_TO_PHORE_TYPE = {
    "Donor": 1,
    "Aromatic": 2,
    "PosIonizable": 3,
    "Acceptor": 4,
    "Hydrophobe": 5,
    "LumpedHydrophobe": 5,
    "NegIonizable": 6,
}

DIRECTIONAL_PHORE_TYPES = DIRECTIONAL_RAW_PHORE_TYPES


@lru_cache(maxsize=1)
def _feature_factory():
    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef_path)


def _merge_overlapping_features(features):
    """Merge same-type features sharing atoms (e.g. fused aromatic rings)."""
    merged = []
    for phore_type, atom_ids in features:
        atom_set = set(int(index) for index in atom_ids)
        overlapping = [
            index for index, item in enumerate(merged)
            if item[0] == phore_type and item[1].intersection(atom_set)
        ]
        if not overlapping:
            merged.append([phore_type, atom_set])
            continue
        base_index = overlapping[0]
        merged[base_index][1].update(atom_set)
        for index in reversed(overlapping[1:]):
            merged[base_index][1].update(merged[index][1])
            del merged[index]
    return [(item[0], tuple(sorted(item[1]))) for item in merged]


def _normal_vector(points):
    if points.shape[0] < 3:
        return None
    centered = points - points.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        return None
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    vector = vh[-1]
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _outward_vector(mol, atom_ids, positions):
    atom_set = set(atom_ids)
    external_neighbors = set()
    for atom_id in atom_ids:
        atom = mol.GetAtomWithIdx(int(atom_id))
        for neighbor in atom.GetNeighbors():
            neighbor_id = neighbor.GetIdx()
            if neighbor_id not in atom_set and neighbor.GetAtomicNum() > 1:
                external_neighbors.add(neighbor_id)
    if not external_neighbors:
        for atom_id in atom_ids:
            atom = mol.GetAtomWithIdx(int(atom_id))
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() > 1:
                    external_neighbors.add(neighbor.GetIdx())
    if not external_neighbors:
        return None
    center = positions[list(atom_ids)].mean(axis=0)
    neighbor_center = positions[sorted(external_neighbors)].mean(axis=0)
    vector = center - neighbor_center
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def extract_molecule_pharmacophores(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return []
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    raw_features = []
    for feature in _feature_factory().GetFeaturesForMol(mol):
        phore_type = RDKIT_FAMILY_TO_PHORE_TYPE.get(feature.GetFamily())
        if phore_type is None:
            continue
        atom_ids = tuple(int(index) for index in feature.GetAtomIds())
        if atom_ids:
            raw_features.append((phore_type, atom_ids))

    extracted = []
    for phore_type, atom_ids in _merge_overlapping_features(raw_features):
        atom_positions = positions[list(atom_ids)]
        center = atom_positions.mean(axis=0)
        if phore_type == 2:
            vector = _normal_vector(atom_positions)
        elif phore_type in DIRECTIONAL_PHORE_TYPES:
            vector = _outward_vector(mol, atom_ids, positions)
        else:
            vector = None
        extracted.append({
            "type": int(phore_type),
            "type_name": PHORE_TYPE_NAMES[int(phore_type)],
            "pos": center,
            "vec": vector,
            "atom_ids": atom_ids,
        })
    return extracted


def _as_numpy(value, dtype=None):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def match_generated_pharmacophores(
    generated_types,
    generated_positions,
    generated_vectors,
    extracted,
    distance_threshold=1.5,
    direction_cosine_threshold=0.7,
):
    generated_types = _as_numpy(generated_types, np.int64)
    generated_positions = _as_numpy(generated_positions, np.float64)
    generated_vectors = _as_numpy(generated_vectors, np.float64)
    generated_count = int(len(generated_types)) if generated_types is not None else 0
    extracted_count = len(extracted)
    large_cost = 1.0e6
    cost = np.full((generated_count, extracted_count), large_cost, dtype=np.float64)
    pair_info = {}

    for generated_id in range(generated_count):
        generated_type = int(generated_types[generated_id])
        for extracted_id, feature in enumerate(extracted):
            if generated_type != int(feature["type"]):
                continue
            distance = float(np.linalg.norm(
                generated_positions[generated_id] - feature["pos"]
            ))
            if not np.isfinite(distance) or distance > distance_threshold:
                continue
            direction_similarity = None
            extracted_vector = feature.get("vec")
            if (generated_type in DIRECTIONAL_PHORE_TYPES
                    and generated_vectors is not None
                    and extracted_vector is not None):
                generated_vector = generated_vectors[generated_id]
                generated_norm = np.linalg.norm(generated_vector)
                extracted_norm = np.linalg.norm(extracted_vector)
                if generated_norm > 1e-8 and extracted_norm > 1e-8:
                    direction_similarity = float(abs(np.dot(
                        generated_vector / generated_norm,
                        extracted_vector / extracted_norm,
                    )))
                    if direction_similarity < direction_cosine_threshold:
                        continue
            angular_penalty = (
                0.05 * (1.0 - direction_similarity)
                if direction_similarity is not None else 0.0
            )
            cost[generated_id, extracted_id] = distance + angular_penalty
            pair_info[(generated_id, extracted_id)] = (
                distance, direction_similarity
            )

    matches = []
    if generated_count and extracted_count:
        rows, cols = linear_sum_assignment(cost)
        for generated_id, extracted_id in zip(rows.tolist(), cols.tolist()):
            if cost[generated_id, extracted_id] >= large_cost:
                continue
            distance, direction_similarity = pair_info[(generated_id, extracted_id)]
            matches.append({
                "generated_index": generated_id,
                "extracted_index": extracted_id,
                "type": int(generated_types[generated_id]),
                "type_name": PHORE_TYPE_NAMES.get(
                    int(generated_types[generated_id]),
                    "type_%d" % int(generated_types[generated_id]),
                ),
                "distance": distance,
                "direction_similarity": direction_similarity,
                "atom_ids": extracted[extracted_id]["atom_ids"],
            })

    matched_count = len(matches)
    matched_by_type = {}
    generated_by_type = {}
    for phore_type in generated_types.tolist() if generated_types is not None else []:
        generated_by_type[int(phore_type)] = generated_by_type.get(int(phore_type), 0) + 1
    for match in matches:
        phore_type = match["type"]
        matched_by_type[phore_type] = matched_by_type.get(phore_type, 0) + 1
    per_type_ratio = {
        PHORE_TYPE_NAMES.get(phore_type, "type_%d" % phore_type): (
            float(matched_by_type.get(phore_type, 0)) / count
        )
        for phore_type, count in sorted(generated_by_type.items())
    }
    distances = [match["distance"] for match in matches]
    direction_similarities = [
        match["direction_similarity"] for match in matches
        if match["direction_similarity"] is not None
    ]
    return {
        "generated_phore_count": generated_count,
        "molecule_extracted_phore_count": extracted_count,
        "matched_phore_count": matched_count,
        "phore_realization_ratio": (
            float(matched_count) / generated_count if generated_count else None
        ),
        "all_phores_realized": (
            int(matched_count == generated_count) if generated_count else None
        ),
        "any_phore_realized": (
            int(matched_count > 0) if generated_count else None
        ),
        "mean_phore_match_distance": (
            float(np.mean(distances)) if distances else None
        ),
        "mean_phore_direction_similarity": (
            float(np.mean(direction_similarities))
            if direction_similarities else None
        ),
        "phore_realization_by_type": json.dumps(
            per_type_ratio, ensure_ascii=False, sort_keys=True
        ),
        "phore_matches": matches,
    }


def evaluate_pharmacophore_realization(
    mol,
    generated_types,
    generated_positions,
    generated_vectors,
    distance_threshold=1.5,
    direction_cosine_threshold=0.7,
):
    extracted = extract_molecule_pharmacophores(mol)
    metrics = match_generated_pharmacophores(
        generated_types,
        generated_positions,
        generated_vectors,
        extracted,
        distance_threshold=distance_threshold,
        direction_cosine_threshold=direction_cosine_threshold,
    )
    metrics["extracted_pharmacophores"] = extracted
    return metrics
