import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from utils.evaluation import analyze
from utils.evaluation.sascorer import compute_sa_score
from utils.phore_realization import evaluate_pharmacophore_realization


DEFAULT_MIN_QED = 0.50
DEFAULT_MIN_SA = 0.60
DEFAULT_MAX_LOGP = 5.0
PHORE_MATCH_DISTANCE_THRESHOLD = 1.5
PHORE_DIRECTION_COSINE_THRESHOLD = 0.7
SCREEN_WEIGHTS = {
    "qed": 0.25,
    "sa": 0.20,
    "distance_mol_stable": 0.20,
    "distance_atom_stable": 0.15,
    "phore_realization": 0.15,
    "model_confidence": 0.05,
}


def _get(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def model_confidence(record):
    ligand = record.get("ligand", {})
    atom_probability = np.asarray(ligand.get("atom_prob", []), dtype=np.float64)
    bond_probability = np.asarray(ligand.get("bond_prob", []), dtype=np.float64)
    atom_confidence = float(atom_probability.mean()) if atom_probability.size else 0.0
    bond_confidence = float(bond_probability.mean()) if bond_probability.size else 0.0
    return float(np.clip((atom_confidence + 0.5 * bond_confidence) / 1.5, 0.0, 1.0))


def distance_stability(record):
    ligand = record.get("ligand", {})
    positions = np.asarray(ligand.get("atom_pos", []), dtype=np.float64)
    elements = np.asarray(ligand.get("element", []), dtype=np.int64)
    if positions.shape != (len(elements), 3) or len(elements) == 0:
        return {
            "distance_mol_stable": 0.0,
            "distance_atom_stable_fraction": 0.0,
        }
    try:
        molecule_stable, stable_atoms, num_atoms = analyze.check_stability(
            positions, elements
        )
    except (AssertionError, KeyError, TypeError, ValueError):
        return {
            "distance_mol_stable": 0.0,
            "distance_atom_stable_fraction": 0.0,
        }
    return {
        "distance_mol_stable": float(bool(molecule_stable)),
        "distance_atom_stable_fraction": (
            float(stable_atoms) / float(num_atoms) if num_atoms else 0.0
        ),
    }


def pharmacophore_realization(record):
    molecule = record.get("rdmol")
    generated_types = record.get("phore_type")
    generated_positions = record.get("phore_pos")
    generated_vectors = record.get("phore_vec")
    if molecule is None or generated_types is None or generated_positions is None:
        return {
            "phore_realization_ratio": 0.0,
            "all_phores_realized": 0.0,
            "generated_phore_count": 0,
        }
    generated_types = np.asarray(generated_types, dtype=np.int64)
    if generated_types.size == 0:
        return {
            "phore_realization_ratio": 0.5,
            "all_phores_realized": 0.0,
            "generated_phore_count": 0,
        }
    try:
        metrics = evaluate_pharmacophore_realization(
            molecule,
            generated_types,
            np.asarray(generated_positions, dtype=np.float64),
            None if generated_vectors is None else np.asarray(
                generated_vectors, dtype=np.float64
            ),
            distance_threshold=PHORE_MATCH_DISTANCE_THRESHOLD,
            direction_cosine_threshold=PHORE_DIRECTION_COSINE_THRESHOLD,
        )
    except (RuntimeError, TypeError, ValueError):
        return {
            "phore_realization_ratio": 0.0,
            "all_phores_realized": 0.0,
            "generated_phore_count": int(generated_types.size),
        }
    return {
        "phore_realization_ratio": _finite(
            metrics.get("phore_realization_ratio"), 0.0
        ),
        "all_phores_realized": _finite(
            metrics.get("all_phores_realized"), 0.0
        ),
        "generated_phore_count": int(
            metrics.get("generated_phore_count", generated_types.size)
        ),
    }


def annotate_candidate(record, config=None):
    molecule = record.get("rdmol")
    metrics = {
        "qed": 0.0,
        "sa": 0.0,
        "logp": float("inf"),
        "model_confidence": model_confidence(record),
    }
    if molecule is not None:
        try:
            Chem.SanitizeMol(molecule)
            metrics.update({
                "qed": _finite(QED.qed(molecule), 0.0),
                "sa": _finite(compute_sa_score(molecule), 0.0),
                "logp": _finite(Crippen.MolLogP(molecule), float("inf")),
            })
        except (RuntimeError, TypeError, ValueError):
            pass
    metrics.update(distance_stability(record))
    metrics.update(pharmacophore_realization(record))

    min_qed = float(_get(config, "min_qed", DEFAULT_MIN_QED))
    min_sa = float(_get(config, "min_sa", DEFAULT_MIN_SA))
    max_logp = float(_get(config, "max_logp", DEFAULT_MAX_LOGP))
    hard_filter = bool(_get(config, "hard_property_filter", True))
    property_pass = bool(
        molecule is not None
        and metrics["qed"] >= min_qed
        and metrics["sa"] >= min_sa
        and metrics["logp"] <= max_logp
    )
    metrics["property_filter_pass"] = float(property_pass)
    metrics["screen_filter_pass"] = float(property_pass or not hard_filter)

    score = (
        SCREEN_WEIGHTS["qed"] * metrics["qed"]
        + SCREEN_WEIGHTS["sa"] * metrics["sa"]
        + SCREEN_WEIGHTS["distance_mol_stable"]
        * metrics["distance_mol_stable"]
        + SCREEN_WEIGHTS["distance_atom_stable"]
        * metrics["distance_atom_stable_fraction"]
        + SCREEN_WEIGHTS["phore_realization"]
        * metrics["phore_realization_ratio"]
        + SCREEN_WEIGHTS["model_confidence"]
        * metrics["model_confidence"]
    )
    metrics["multiobjective_score"] = float(score)
    record["candidate_screen"] = metrics
    return metrics


def screen_summary(records):
    fields = (
        "qed", "sa", "logp", "distance_mol_stable",
        "distance_atom_stable_fraction", "phore_realization_ratio",
        "all_phores_realized", "property_filter_pass", "multiobjective_score",
    )
    summary = {}
    for field in fields:
        values = []
        for record in records:
            value = record.get("candidate_screen", {}).get(field)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        summary[field] = float(np.mean(values)) if values else None
    return summary
