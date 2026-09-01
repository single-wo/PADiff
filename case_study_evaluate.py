from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from utils.case_study import (
    chemical_quality_flags,
    evaluate_generated_phore_hotspot_recovery,
    evaluate_hotspot_recovery,
    evaluate_pose_clashes,
    prepare_case_study,
)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    values = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None


def _median(values):
    values = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.median(values)) if values else None


def _affinity(value):
    value = _finite(value)
    return value if value is not None and value < 0 else None


def _load_selected_records(sample_path, record_filename):
    paths = sorted(Path(sample_path).glob("**/%s" % record_filename))
    if not paths:
        raise FileNotFoundError("No %s found under %s" % (record_filename, sample_path))
    if len(paths) != 1:
        raise ValueError(
            "A merged single-pocket case study requires exactly one %s; found %d"
            % (record_filename, len(paths))
        )
    payload = torch.load(str(paths[0]), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "finished" in payload:
        return list(payload["finished"]), paths[0]
    if isinstance(payload, list):
        return payload, paths[0]
    raise ValueError("Unsupported sample format: %s" % paths[0])


def _record_molecule(record):
    if record.get("rdmol") is not None:
        return record["rdmol"]
    return (record.get("ligand") or {}).get("rdmol")


def _scaffold_smiles(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames=None):
    fieldnames = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_docking_results(result_path):
    metrics_path = Path(result_path) / "metrics.pt"
    if not metrics_path.is_file():
        return {}
    payload = torch.load(str(metrics_path), map_location="cpu", weights_only=False)
    indexed = {}
    for result in payload.get("all_results", []):
        sample_index = result.get("sample_index")
        if sample_index is not None:
            indexed[int(sample_index)] = result
    return indexed


def _docked_molecule(result):
    try:
        return result["vina"]["dock"][0].get("rdmol")
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _metric_details(row, prefix, metrics):
    row[prefix + "_hotspot_recovery_ratio"] = metrics["hotspot_recovery_ratio"]
    row[prefix + "_all_hotspots_recovered"] = metrics["all_hotspots_recovered"]
    row[prefix + "_any_hotspot_recovered"] = metrics["any_hotspot_recovered"]
    row[prefix + "_hotspot_details"] = json.dumps(
        metrics["hotspot_details"], sort_keys=True
    )
    for detail in metrics["hotspot_details"]:
        name = detail["name"]
        row["%s_hotspot_%s_recovered" % (prefix, name)] = detail["recovered"]
        row["%s_hotspot_%s_min_distance" % (prefix, name)] = detail[
            "minimum_distance"
        ]


def _meets_minimum(value, threshold):
    value = _finite(value)
    return value is not None and value >= float(threshold)


def _passes_joint_filter(row, thresholds):
    affinity = _affinity(row.get("affinity"))
    logp = _finite(row.get("logp"))
    phore_count = _finite(row.get("num_pharmacophores"))
    if phore_count is None:
        phore_count = _finite(row.get("generated_phore_count"))
    requirements = [
        affinity is not None and affinity <= thresholds["max_affinity"],
        _meets_minimum(row.get("qed"), thresholds["min_qed"]),
        _meets_minimum(row.get("sa"), thresholds["min_sa"]),
        logp is not None and logp <= thresholds["max_logp"],
        int(_finite(row.get("distance_inferred_mol_stable")) or 0) == 1,
        int(_finite(row.get("all_phores_realized")) or 0) == 1,
        phore_count is not None and phore_count >= thresholds["min_phores"],
        int(_finite(row.get("generated_phore_all_hotspots_recovered")) or 0) == 1,
        int(_finite(row.get("docked_pose_all_hotspots_recovered")) or 0) == 1,
        int(_finite(row.get("alert_free")) or 0) == 1,
        _finite(row.get("docked_pose_has_severe_clash")) == 0.0,
    ]
    return int(all(requirements))


def _seed_summary(seed, rows, reference_affinity, top_k, selected_count=None):
    affinities = [
        value for value in (_affinity(row.get("affinity")) for row in rows)
        if value is not None
    ]
    ranked = sorted(affinities)
    top_values = ranked[:max(1, int(top_k))]
    smiles = [row.get("smiles") for row in rows if row.get("smiles")]
    scaffolds = [
        row.get("scaffold_smiles") for row in rows if row.get("scaffold_smiles")
    ]
    return {
        "seed": seed,
        "selected_molecules": int(selected_count if selected_count is not None else len(rows)),
        "evaluated_molecules": len(rows),
        "evaluation_success_rate": (
            float(len(rows)) / selected_count if selected_count else None
        ),
        "mean_affinity": _mean(affinities),
        "std_affinity": _std(affinities),
        "median_affinity": _median(affinities),
        "best_affinity": min(affinities) if affinities else None,
        "top_k": int(top_k),
        "top_k_mean_affinity": _mean(top_values),
        "fraction_no_worse_than_reference": _mean(
            int(value <= reference_affinity) for value in affinities
        ) if reference_affinity is not None else None,
        "fraction_1kcal_better_than_reference": _mean(
            int(value <= reference_affinity - 1.0) for value in affinities
        ) if reference_affinity is not None else None,
        "fraction_2kcal_better_than_reference": _mean(
            int(value <= reference_affinity - 2.0) for value in affinities
        ) if reference_affinity is not None else None,
        "mean_qed": _mean(row.get("qed") for row in rows),
        "mean_sa": _mean(row.get("sa") for row in rows),
        "whole_molecule_stable_ratio": _mean(
            row.get("distance_inferred_mol_stable") for row in rows
        ),
        "phore_realization_ratio": _mean(
            row.get("phore_realization_ratio") for row in rows
        ),
        "all_phores_realized_ratio": _mean(
            row.get("all_phores_realized") for row in rows
        ),
        "generated_phore_all_hotspots_ratio": _mean(
            row.get("generated_phore_all_hotspots_recovered") for row in rows
        ),
        "generated_pose_all_hotspots_ratio": _mean(
            row.get("generated_pose_all_hotspots_recovered") for row in rows
        ),
        "docked_pose_available_ratio": _mean(
            row.get("docked_pose_available") for row in rows
        ),
        "docked_pose_all_hotspots_ratio": _mean(
            row.get("docked_pose_all_hotspots_recovered") for row in rows
        ),
        "hotspot_retention_ratio": _mean(
            row.get("docked_pose_all_hotspots_recovered")
            for row in rows
            if int(_finite(row.get("generated_pose_all_hotspots_recovered")) or 0) == 1
        ),
        "docked_pose_severe_clash_ratio": _mean(
            row.get("docked_pose_has_severe_clash") for row in rows
        ),
        "alert_free_ratio": _mean(row.get("alert_free") for row in rows),
        "joint_hit_count": int(sum(int(row.get("joint_hit") or 0) for row in rows)),
        "joint_hit_rate": _mean(row.get("joint_hit") for row in rows),
        "unique_smiles": len(set(smiles)),
        "uniqueness": float(len(set(smiles))) / len(smiles) if smiles else None,
        "unique_scaffolds": len(set(scaffolds)),
        "scaffold_bearing_molecules": len(scaffolds),
        "scaffold_uniqueness": (
            float(len(set(scaffolds))) / len(scaffolds) if scaffolds else None
        ),
    }


def _replicate_statistics(seed_rows):
    metrics = [
        "mean_affinity", "top_k_mean_affinity",
        "fraction_no_worse_than_reference",
        "fraction_1kcal_better_than_reference",
        "fraction_2kcal_better_than_reference", "mean_qed", "mean_sa",
        "whole_molecule_stable_ratio", "phore_realization_ratio",
        "all_phores_realized_ratio", "generated_phore_all_hotspots_ratio",
        "generated_pose_all_hotspots_ratio",
        "docked_pose_available_ratio", "docked_pose_all_hotspots_ratio",
        "hotspot_retention_ratio", "evaluation_success_rate",
        "docked_pose_severe_clash_ratio", "alert_free_ratio",
        "joint_hit_rate", "uniqueness", "scaffold_uniqueness",
    ]
    output = []
    for metric in metrics:
        values = [
            value for value in (_finite(row.get(metric)) for row in seed_rows)
            if value is not None
        ]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if values else None
        half_width = (
            float(stats.t.ppf(0.975, len(values) - 1)) * sem
            if len(values) > 1 else 0.0
        )
        output.append({
            "metric": metric,
            "num_seeds": len(values),
            "seed_mean": mean,
            "seed_std": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "seed_min": min(values),
            "seed_max": max(values),
        })
    return output


def _pairwise_scaffold_jaccard(rows_by_seed):
    sets = {}
    for seed, rows in rows_by_seed.items():
        sets[seed] = {
            row.get("scaffold_smiles") for row in rows
            if row.get("scaffold_smiles")
        }
    values = []
    pair_rows = []
    seeds = sorted(sets, key=str)
    for left_index, left in enumerate(seeds):
        for right in seeds[left_index + 1:]:
            union = sets[left] | sets[right]
            value = (
                float(len(sets[left] & sets[right])) / len(union)
                if union else None
            )
            pair_rows.append({
                "seed_left": left,
                "seed_right": right,
                "left_unique_scaffolds": len(sets[left]),
                "right_unique_scaffolds": len(sets[right]),
                "shared_scaffolds": len(sets[left] & sets[right]),
                "union_scaffolds": len(union),
                "scaffold_jaccard": value,
            })
            if value is not None:
                values.append(value)
    return _mean(values), _std(values), pair_rows


def _diverse_shortlist(rows, maximum):
    ranked = sorted(
        [row for row in rows if int(row.get("joint_hit") or 0) == 1],
        key=lambda row: float(row["affinity"]),
    )
    selected = []
    seen_scaffolds = set()
    deferred = []
    for row in ranked:
        scaffold = row.get("scaffold_smiles")
        if scaffold and scaffold not in seen_scaffolds:
            selected.append(row)
            seen_scaffolds.add(scaffold)
        else:
            deferred.append(row)
        if len(selected) >= int(maximum):
            return selected
    selected.extend(deferred[:max(0, int(maximum) - len(selected))])
    return selected


def _write_shortlist_sdf(path, shortlist, records, docking_results, docked=False):
    writer = Chem.SDWriter(str(path))
    for rank, row in enumerate(shortlist, 1):
        sample_index = int(row["sample_index"])
        mol = None
        if docked:
            mol = _docked_molecule(docking_results.get(sample_index, {}))
        if mol is None:
            mol = _record_molecule(records[sample_index])
        if mol is None:
            continue
        mol = Chem.Mol(mol)
        for key in (
            "sample_index", "case_study_seed", "case_study_seed_sample_index",
            "affinity", "qed", "sa", "logp", "ligand_efficiency",
            "phore_realization_ratio", "all_phores_realized",
            "docked_pose_hotspot_recovery_ratio",
            "docked_pose_all_hotspots_recovered", "alert_free", "joint_hit",
        ):
            value = row.get(key)
            if value is not None:
                mol.SetProp(key, str(value))
        mol.SetProp("shortlist_rank", str(rank))
        writer.write(mol)
    writer.close()


def _find_multiseed_manifest(sample_path):
    candidates = [
        Path(sample_path) / "multiseed_manifest.json",
        Path(sample_path).parent / "multiseed_manifest.json",
    ]
    for path in candidates:
        if path.is_file():
            with path.open() as handle:
                return json.load(handle), str(path.resolve())
    return None, None


def _postprocess(manifest, sample_path, result_path, record_filename, top_k,
                 thresholds, shortlist_size):
    records, record_path = _load_selected_records(sample_path, record_filename)
    core_rows = _read_csv(Path(result_path) / "molecules.csv")
    detail_path = Path(result_path) / "molecule_details.csv"
    detail_rows = _read_csv(detail_path) if detail_path.is_file() else []
    detail_by_index = {
        int(row["ligand_id"]) - 1: row
        for row in detail_rows if row.get("ligand_id")
    }
    core_by_index = {}
    for row in core_rows:
        if not row.get("ligand_id"):
            continue
        sample_index = int(row["ligand_id"]) - 1
        merged = dict(detail_by_index.get(sample_index, {}))
        merged.update(row)
        core_by_index[sample_index] = merged
    docking_results = _load_docking_results(result_path)

    reference = Chem.SDMolSupplier(
        manifest["reference_ligand_path"], removeHs=False
    )[0]
    if reference is None:
        raise ValueError("Reference ligand could not be read")
    reference_fp = AllChem.GetMorganFingerprintAsBitVect(reference, radius=2, nBits=2048)
    reference_scaffold = _scaffold_smiles(reference)
    reference_hotspots = evaluate_hotspot_recovery(
        reference, manifest["receptor_path"], manifest.get("hotspots", [])
    )

    rows = []
    for sample_index, record in enumerate(records):
        core = core_by_index.get(sample_index, {})
        mol = _record_molecule(record)
        if mol is None or core.get("status") != "evaluated":
            continue
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        affinity = _affinity(core.get("affinity"))
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        scaffold = _scaffold_smiles(mol)
        generated_phore_hotspots = evaluate_generated_phore_hotspot_recovery(
            record.get("phore_pos"), record.get("phore_type"),
            manifest["receptor_path"], manifest.get("hotspots", []),
        )
        generated_hotspots = evaluate_hotspot_recovery(
            mol, manifest["receptor_path"], manifest.get("hotspots", [])
        )
        generated_clashes = evaluate_pose_clashes(mol, manifest["receptor_path"])
        result = docking_results.get(sample_index, {})
        docked_mol = _docked_molecule(result)
        docked_hotspots = (
            evaluate_hotspot_recovery(
                docked_mol, manifest["receptor_path"], manifest.get("hotspots", [])
            ) if docked_mol is not None else None
        )
        docked_clashes = (
            evaluate_pose_clashes(docked_mol, manifest["receptor_path"])
            if docked_mol is not None else None
        )
        alerts = chemical_quality_flags(mol)
        row = dict(core)
        row.update({
            "sample_index": sample_index,
            "case_study_seed": record.get("case_study_seed", 0),
            "case_study_seed_index": record.get("case_study_seed_index", 0),
            "case_study_seed_sample_index": record.get(
                "case_study_seed_sample_index", sample_index
            ),
            "candidate_index": record.get("candidate_index"),
            "selection_strict_feasible": int(bool(
                (record.get("selection_metadata") or {}).get("strict_feasible", False)
            )),
            "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
            "scaffold_smiles": scaffold,
            "reference_ecfp4_similarity": float(
                DataStructs.TanimotoSimilarity(fingerprint, reference_fp)
            ),
            "reference_scaffold_match": int(
                reference_scaffold is not None and scaffold == reference_scaffold
            ),
            "heavy_atoms": heavy_atoms,
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "ligand_efficiency": (
                -affinity / heavy_atoms if affinity is not None and heavy_atoms else None
            ),
            "docked_pose_available": int(docked_mol is not None),
        })
        _metric_details(row, "generated_phore", generated_phore_hotspots)
        _metric_details(row, "generated_pose", generated_hotspots)
        row.update({
            "generated_pose_%s" % key: value
            for key, value in generated_clashes.items()
        })
        if docked_hotspots is not None:
            _metric_details(row, "docked_pose", docked_hotspots)
        else:
            row.update({
                "docked_pose_hotspot_recovery_ratio": None,
                "docked_pose_all_hotspots_recovered": None,
                "docked_pose_any_hotspot_recovered": None,
                "docked_pose_hotspot_details": None,
            })
        if docked_clashes is not None:
            row.update({
                "docked_pose_%s" % key: value
                for key, value in docked_clashes.items()
            })
        else:
            row.update({
                "docked_pose_minimum_protein_distance": None,
                "docked_pose_severe_clash_count": None,
                "docked_pose_vdw_clash_count": None,
                "docked_pose_has_severe_clash": None,
                "docked_pose_has_vdw_clash": None,
            })
        row.update(alerts)
        row["catalog_alerts"] = json.dumps(alerts["catalog_alerts"], sort_keys=True)
        phore_count = _finite(row.get("num_pharmacophores"))
        if phore_count is None:
            phore_count = _finite(row.get("generated_phore_count"))
        row["property_pass"] = int(
            _meets_minimum(row.get("qed"), thresholds["min_qed"])
            and _meets_minimum(row.get("sa"), thresholds["min_sa"])
            and _finite(row.get("logp")) is not None
            and float(row["logp"]) <= thresholds["max_logp"]
        )
        row["affinity_pass"] = int(
            affinity is not None and affinity <= thresholds["max_affinity"]
        )
        row["stable_pass"] = int(
            int(_finite(row.get("distance_inferred_mol_stable")) or 0) == 1
        )
        row["phore_consistency_pass"] = int(
            int(_finite(row.get("all_phores_realized")) or 0) == 1
            and phore_count is not None
            and phore_count >= thresholds["min_phores"]
        )
        row["generated_phore_hotspot_pass"] = int(
            int(_finite(row.get("generated_phore_all_hotspots_recovered")) or 0) == 1
        )
        row["docked_hotspot_pass"] = int(
            int(_finite(row.get("docked_pose_all_hotspots_recovered")) or 0) == 1
        )
        row["chemical_pass"] = int(
            int(_finite(row.get("alert_free")) or 0) == 1
        )
        row["clash_pass"] = int(
            _finite(row.get("docked_pose_has_severe_clash")) == 0
        )
        row["joint_hit"] = _passes_joint_filter(row, thresholds)
        rows.append(row)

    ranked = sorted(
        [row for row in rows if _affinity(row.get("affinity")) is not None],
        key=lambda row: float(row["affinity"]),
    )
    top_rows = ranked[:max(1, int(top_k))]
    reference_affinity = _affinity(rows[0].get("affinity_reference")) if rows else None
    rows_by_seed = defaultdict(list)
    selected_by_seed = defaultdict(int)
    for record in records:
        selected_by_seed[record.get("case_study_seed", 0)] += 1
    for row in rows:
        rows_by_seed[row.get("case_study_seed", 0)].append(row)
    seed_rows = [
        _seed_summary(
            seed, rows_by_seed.get(seed, []), reference_affinity, top_k,
            selected_count=selected_by_seed[seed],
        )
        for seed in sorted(selected_by_seed, key=str)
    ]
    replicate_rows = _replicate_statistics(seed_rows)
    scaffold_jaccard_mean, scaffold_jaccard_std, scaffold_pair_rows = (
        _pairwise_scaffold_jaccard(rows_by_seed)
    )
    shortlist = _diverse_shortlist(rows, shortlist_size)
    hotspot_names = [item["name"] for item in manifest.get("hotspots", [])]
    affinities = [
        value for value in (_affinity(row.get("affinity")) for row in rows)
        if value is not None
    ]
    multiseed_manifest, multiseed_manifest_path = _find_multiseed_manifest(sample_path)
    reference_redocking_path = Path(result_path) / "reference_redocking.json"
    reference_redocking = None
    if reference_redocking_path.is_file():
        with reference_redocking_path.open() as handle:
            reference_redocking = json.load(handle)
    sampling_budget = None
    if multiseed_manifest:
        sampling_budget = {
            key: multiseed_manifest.get(key) for key in (
                "num_seeds", "seeds", "molecules_per_seed", "pooled_molecules",
                "raw_candidates", "selected_candidates", "strict_selected",
                "fallback_selected", "candidate_multiplier", "sampling_seconds",
                "pooled_unique_smiles", "pooled_smiles_uniqueness",
                "pooled_unique_scaffolds", "pooled_scaffold_bearing_molecules",
                "per_seed_budget",
            )
        }

    summary = {
        "case_id": manifest["id"],
        "sample_record": str(record_path.resolve()),
        "generated_records": len(records),
        "fully_evaluated_records": len(rows),
        "num_seeds": len(seed_rows),
        "seeds": [row["seed"] for row in seed_rows],
        "reference_ligand": manifest["source_ligand"],
        "reference_affinity": reference_affinity,
        "mean_affinity": _mean(affinities),
        "std_affinity": _std(affinities),
        "median_affinity": _median(affinities),
        "best_affinity": min(affinities) if affinities else None,
        "top_k": int(top_k),
        "pooled_top_k_mean_affinity": _mean(
            _affinity(row.get("affinity")) for row in top_rows
        ),
        "mean_seed_top_k_affinity": _mean(
            row.get("top_k_mean_affinity") for row in seed_rows
        ),
        "std_seed_top_k_affinity": _std(
            row.get("top_k_mean_affinity") for row in seed_rows
        ),
        "fraction_no_worse_than_reference": _mean(
            int(value <= reference_affinity) for value in affinities
        ) if reference_affinity is not None else None,
        "fraction_1kcal_better_than_reference": _mean(
            int(value <= reference_affinity - 1.0) for value in affinities
        ) if reference_affinity is not None else None,
        "fraction_2kcal_better_than_reference": _mean(
            int(value <= reference_affinity - 2.0) for value in affinities
        ) if reference_affinity is not None else None,
        "mean_qed": _mean(row.get("qed") for row in rows),
        "mean_sa": _mean(row.get("sa") for row in rows),
        "mean_reference_ecfp4_similarity": _mean(
            row.get("reference_ecfp4_similarity") for row in rows
        ),
        "maximum_reference_ecfp4_similarity": max(
            (row["reference_ecfp4_similarity"] for row in rows), default=None
        ),
        "reference_scaffold_match_ratio": _mean(
            row.get("reference_scaffold_match") for row in rows
        ),
        "mean_ligand_efficiency": _mean(
            row.get("ligand_efficiency") for row in rows
        ),
        "generated_phore_hotspot_recovery_ratio": _mean(
            row.get("generated_phore_hotspot_recovery_ratio") for row in rows
        ),
        "generated_phore_all_hotspots_recovered_ratio": _mean(
            row.get("generated_phore_all_hotspots_recovered") for row in rows
        ),
        "generated_pose_hotspot_recovery_ratio": _mean(
            row.get("generated_pose_hotspot_recovery_ratio") for row in rows
        ),
        "generated_pose_all_hotspots_recovered_ratio": _mean(
            row.get("generated_pose_all_hotspots_recovered") for row in rows
        ),
        "docked_pose_available_ratio": _mean(
            row.get("docked_pose_available") for row in rows
        ),
        "docked_pose_hotspot_recovery_ratio": _mean(
            row.get("docked_pose_hotspot_recovery_ratio") for row in rows
        ),
        "docked_pose_all_hotspots_recovered_ratio": _mean(
            row.get("docked_pose_all_hotspots_recovered") for row in rows
        ),
        "generated_to_docked_all_hotspot_retention_ratio": _mean(
            row.get("docked_pose_all_hotspots_recovered")
            for row in rows
            if int(_finite(row.get("generated_pose_all_hotspots_recovered")) or 0) == 1
        ),
        "docked_pose_severe_clash_ratio": _mean(
            row.get("docked_pose_has_severe_clash") for row in rows
        ),
        "alert_free_ratio": _mean(row.get("alert_free") for row in rows),
        "joint_filter": thresholds,
        "joint_hit_count": int(sum(int(row.get("joint_hit") or 0) for row in rows)),
        "joint_hit_rate": _mean(row.get("joint_hit") for row in rows),
        "filter_counts": {
            key: int(sum(int(row.get(key) or 0) for row in rows))
            for key in (
                "affinity_pass", "property_pass", "stable_pass",
                "phore_consistency_pass", "generated_phore_hotspot_pass",
                "docked_hotspot_pass",
                "chemical_pass", "clash_pass", "joint_hit",
            )
        },
        "diverse_shortlist_count": len(shortlist),
        "pairwise_seed_scaffold_jaccard_mean": scaffold_jaccard_mean,
        "pairwise_seed_scaffold_jaccard_std": scaffold_jaccard_std,
        "pairwise_seed_scaffold_pairs": len(scaffold_pair_rows),
        "reference_generated_pose_hotspots": reference_hotspots,
        "reference_redocking": reference_redocking,
        "reference_redocking_path": (
            str(reference_redocking_path.resolve())
            if reference_redocking is not None else None
        ),
        "hotspot_definition": manifest.get("hotspots", []),
        "hotspot_coordinate_note": (
            "Generated-phore metrics use the jointly generated PADiff phore nodes; "
            "generated-pose metrics use pharmacophores extracted from the PADiff "
            "molecular coordinates; docked-pose metrics use pharmacophores extracted "
            "from the Vina pose mapped back to the original molecular graph."
        ),
        "multiseed_manifest": multiseed_manifest_path,
        "sampling_budget": sampling_budget,
    }
    for name in hotspot_names:
        summary["generated_phore_hotspot_%s_recovery_ratio" % name] = _mean(
            row.get("generated_phore_hotspot_%s_recovered" % name) for row in rows
        )
        summary["generated_pose_hotspot_%s_recovery_ratio" % name] = _mean(
            row.get("generated_pose_hotspot_%s_recovered" % name) for row in rows
        )
        summary["docked_pose_hotspot_%s_recovery_ratio" % name] = _mean(
            row.get("docked_pose_hotspot_%s_recovered" % name) for row in rows
        )

    result_path = Path(result_path)
    _write_csv(result_path / "case_study_molecules.csv", rows)
    _write_csv(result_path / ("case_study_top%d.csv" % int(top_k)), top_rows)
    _write_csv(result_path / "case_study_seed_summary.csv", seed_rows)
    _write_csv(result_path / "case_study_replicate_statistics.csv", replicate_rows)
    _write_csv(
        result_path / "case_study_seed_scaffold_jaccard.csv",
        scaffold_pair_rows,
        fieldnames=(
            "seed_left", "seed_right", "left_unique_scaffolds",
            "right_unique_scaffolds", "shared_scaffolds",
            "union_scaffolds", "scaffold_jaccard",
        ),
    )
    _write_csv(result_path / "case_study_shortlist.csv", shortlist)
    _write_shortlist_sdf(
        result_path / "case_study_shortlist_generated.sdf",
        shortlist, records, docking_results, docked=False,
    )
    _write_shortlist_sdf(
        result_path / "case_study_shortlist_docked.sdf",
        shortlist, records, docking_results, docked=True,
    )
    with open(result_path / "case_study_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--base-config", default="configs/eval.yml")
    parser.add_argument("--record-filename", default="joint_samples.pt")
    parser.add_argument(
        "--docking-mode", choices=("vina_score", "vina_dock", "none"),
        default="vina_dock",
    )
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument("--joint-min-qed", type=float, default=0.70)
    parser.add_argument("--joint-min-sa", type=float, default=0.65)
    parser.add_argument("--joint-max-logp", type=float, default=5.0)
    parser.add_argument("--joint-max-affinity", type=float, default=-8.5)
    parser.add_argument("--joint-min-phores", type=int, default=3)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-core-evaluation", action="store_true")
    parser.add_argument("--unfiltered-top-k", action="store_true")
    args = parser.parse_args(argv)

    if args.docking_mode != "vina_dock":
        print(
            "Warning: strict joint hits require docked poses; docking_mode=%s will "
            "produce zero strict joint hits." % args.docking_mode,
            file=sys.stderr,
        )
    thresholds = {
        "min_qed": float(args.joint_min_qed),
        "min_sa": float(args.joint_min_sa),
        "max_logp": float(args.joint_max_logp),
        "max_affinity": float(args.joint_max_affinity),
        "min_phores": int(args.joint_min_phores),
        "require_whole_molecule_stable": True,
        "require_all_generated_phores": True,
        "require_all_generated_phore_hotspots": True,
        "require_all_docked_hotspots": True,
        "require_alert_free": True,
        "require_no_severe_clash": True,
    }

    manifest = prepare_case_study(args.case, force=args.force_prepare)
    result_path = Path(args.result_path).resolve()
    result_path.mkdir(parents=True, exist_ok=True)
    with open(args.base_config) as handle:
        config = yaml.safe_load(handle) or {}
    config.update({
        "sample_path": str(Path(args.sample_path).resolve()),
        "record_filename": args.record_filename,
        "result_path": str(result_path),
        "eval_num_examples": 1,
        "docking_mode": args.docking_mode,
        "exhaustiveness": int(args.exhaustiveness),
        "evaluate_reference_docking": args.docking_mode != "none",
        "top_k": int(args.top_k),
    })
    config.setdefault("topk_filter", {})["enabled"] = not args.unfiltered_top_k
    config["case_study"] = {
        "enabled": True,
        "id": manifest["id"],
        "receptor_path": manifest["receptor_path"],
        "reference_ligand_path": manifest["reference_ligand_path"],
        "reference_label": manifest["source_ligand"]["resname"],
        "docking_center": manifest["docking_center"],
        "docking_box_size": manifest["docking_box_size"],
    }
    generated_config = result_path / "evaluate_case.yml"
    with open(generated_config, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    if not args.skip_core_evaluation:
        subprocess.run([
            sys.executable,
            str(Path(__file__).resolve().with_name("evaluate.py")),
            "--config", str(generated_config),
            "--result_path", str(result_path),
        ], check=True)
    _postprocess(
        manifest, args.sample_path, result_path, args.record_filename, args.top_k,
        thresholds, args.shortlist_size,
    )


if __name__ == "__main__":
    main()
