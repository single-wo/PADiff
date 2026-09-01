import argparse
import json
import os
import pickle
import re
import sys
import time
import numpy as np
import torch
import yaml
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm
sys.path.append(".")
from utils.candidate_screen import (
    DEFAULT_MAX_LOGP, DEFAULT_MIN_QED, DEFAULT_MIN_SA,
    annotate_candidate, screen_summary,
)
from utils.data import PDBProtein
from utils.dataset import ProteinLigandData, get_dataset, to_torch_dict
from utils.cardinality_prior import (
    compute_pocket_descriptors,
    sample_atom_counts,
    sample_pharmacophore_counts as sample_pharmacophore_counts_from_prior,
)
from utils.misc import get_logger, get_new_log_dir, load_config, seed_all
from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges
from utils.sample_utils import separate_outputs
from utils.transforms import FeatureComplex, make_data_placeholder


def parse_pocket_pdb(pocket_pdb_path):

    protein_dict = to_torch_dict(PDBProtein(pocket_pdb_path).to_dict_atom())
    ligand_dict = {
        "element": torch.empty(0, dtype=torch.long),
        "hybridization": torch.empty(0, dtype=torch.long),
        "pos": torch.empty((0, 3), dtype=torch.float),
        "bond_index": torch.empty((2, 0), dtype=torch.long),
        "bond_type": torch.empty(0, dtype=torch.long),
        "atom_feature": torch.empty((0, 8), dtype=torch.float),
    }
    data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=protein_dict, ligand_dict=ligand_dict
    )
    data.protein_anchor_pos = torch.empty((0, 3), dtype=torch.float)
    data.protein_anchor_vec = torch.empty((0, 3), dtype=torch.float)
    data.protein_anchor_type = torch.empty(0, dtype=torch.long)
    data.protein_anchor_confidence = torch.empty(0, dtype=torch.float)
    data.phore_pos = torch.empty((0, 3), dtype=torch.float)
    data.phore_vec = torch.empty((0, 3), dtype=torch.float)
    data.phore_type = torch.empty(0, dtype=torch.long)
    data.protein_filename = pocket_pdb_path
    return data


def pocket_context_only(data):

    return ProteinLigandData(
        protein_atom_feat=data.protein_atom_feat,
        protein_pos=data.protein_pos,
        protein_element=data.protein_element,
    )


def sample_ligand_atom_counts(config, data, n_graphs, cardinality_prior):

    descriptor = compute_pocket_descriptors(data)
    pocket_size = float(descriptor[0])
    stratified_bins = int(getattr(config.sample, "atom_count_num_bins", 1))
    return sample_atom_counts(
        cardinality_prior,
        [pocket_size] * n_graphs,
        pocket_descriptor_vectors=[descriptor] * n_graphs,
        stratified_bins=stratified_bins,
    )





def sample_pharmacophore_counts(
    config, data, atom_counts, n_graphs, cardinality_prior
):

    if cardinality_prior is None:
        raise ValueError(
            "Checkpoint does not contain a training-only pharmacophore-count prior. "
            "Train a pocket-only checkpoint before sampling."
        )
    descriptor = compute_pocket_descriptors(data)
    pocket_sizes = [float(descriptor[0])] * n_graphs
    return sample_pharmacophore_counts_from_prior(
        cardinality_prior, pocket_sizes, atom_counts,
        pocket_descriptor_vectors=[descriptor] * n_graphs,
    )


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default) if config is not None else default


def sampling_protocol_settings(sample_config, requested=None):

    protocol = str(_config_value(sample_config, "protocol", "")).lower()
    if protocol not in {"adaptive", "direct"}:
        raise ValueError("sample.protocol must be 'adaptive' or 'direct'")
    is_adaptive = protocol == "adaptive"
    settings = {
        "enabled": True,
        "protocol": protocol,
        "strategy": "remaining_quota" if is_adaptive else "direct",
        "max_batches": int(_config_value(sample_config, "max_batches", 10))
        if is_adaptive else 1,
        "deduplicate_smiles": is_adaptive,
    }
    if settings["max_batches"] < 1:
        raise ValueError("sample.max_batches must be >= 1")
    if requested is not None:
        requested = int(requested)
        if requested < 1:
            raise ValueError("requested molecule count must be >= 1")
        settings["initial_candidates"] = requested
        settings["required_feasible_candidates"] = requested
        settings["max_generated_candidates"] = (
            requested if settings["strategy"] == "direct"
            else requested * settings["max_batches"]
        )
    return settings



def apply_candidate_screen_overrides(
    sample_config,
    min_qed=None,
    min_sa=None,
    max_logp=None,
    hard_property_filter=None,
):

    protocol = sampling_protocol_settings(sample_config)["protocol"]
    candidate_screen = EasyDict({
        "enabled": protocol == "adaptive",
        "hard_property_filter": protocol == "adaptive",
        "retain_atom_count_strata": protocol == "adaptive",
        "min_qed": DEFAULT_MIN_QED,
        "min_sa": DEFAULT_MIN_SA,
        "max_logp": DEFAULT_MAX_LOGP,
    })
    overrides = {
        "min_qed": min_qed,
        "min_sa": min_sa,
        "max_logp": max_logp,
        "hard_property_filter": hard_property_filter,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(candidate_screen, name, value)

    min_qed_value = float(candidate_screen.min_qed)
    min_sa_value = float(candidate_screen.min_sa)
    max_logp_value = float(candidate_screen.max_logp)
    if not 0.0 <= min_qed_value <= 1.0:
        raise ValueError("candidate_screen.min_qed must be in [0, 1]")
    if not 0.0 <= min_sa_value <= 1.0:
        raise ValueError("candidate_screen.min_sa must be in [0, 1]")
    if not np.isfinite(max_logp_value):
        raise ValueError("candidate_screen.max_logp must be finite")
    return candidate_screen


def validate_clean_sampling_config(config):
    sampling_protocol_settings(config.sample)

def tensor_to_numpy_list(values):
    return [value.detach().float().cpu().numpy() for value in values]



def terminal_real_atom_logits(onehot_state, decoder_logits, real_atom_type_count):
    state = onehot_state.float()
    decoder_logits = decoder_logits.float()
    real_count = int(real_atom_type_count)
    if state.ndim != 2 or decoder_logits.ndim != 2:
        raise ValueError("terminal state and decoder logits must have shape [N, K]")
    if state.size(0) != decoder_logits.size(0) or real_count <= 0:
        raise ValueError("invalid terminal atom tensors")
    sampled_class = state.argmax(dim=-1)
    real_logits = decoder_logits[:, :real_count].clone()
    sampled_real = sampled_class < real_count
    if sampled_real.any():
        real_logits[sampled_real] = -80.0
        real_logits[sampled_real, sampled_class[sampled_real]] = 0.0
    return real_logits


def plain_config(value):
    if isinstance(value, dict):
        return {key: plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    return value


def completed_case_ids(log_dir, requested, total_cases):
    completed = set()
    if not os.path.isdir(log_dir):
        return completed
    for name in os.listdir(log_dir):
        match = re.match(r"^(\d+)_", name)
        if match is None:
            continue
        case_id = int(match.group(1))
        if case_id >= total_cases:
            continue
        result_path = os.path.join(log_dir, name, "joint_samples.pt")
        if not os.path.isfile(result_path) or os.path.getsize(result_path) == 0:
            continue
        try:
            records = torch.load(result_path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
            continue
        if isinstance(records, (list, tuple)) and len(records) >= requested:
            completed.add(case_id)
    return completed


def find_resumable_log_dir(
    outdir, prefix, config, config_filename, requested, total_cases,
):
    if not os.path.isdir(outdir):
        return None, set()
    candidates = sorted(
        (
            os.path.join(outdir, name)
            for name in os.listdir(outdir)
            if name.startswith(prefix + "_")
            and os.path.isdir(os.path.join(outdir, name))
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    for candidate in candidates:
        saved_config_path = os.path.join(candidate, config_filename)
        if not os.path.isfile(saved_config_path):
            continue
        try:
            if load_config(saved_config_path) != config:
                continue
        except (OSError, ValueError):
            continue
        completed = completed_case_ids(candidate, requested, total_cases)
        if len(completed) < total_cases:
            return candidate, completed
        return None, set()
    return None, set()



def summarize_candidate_records(records):
    total = len(records)
    reconstructed = connected = final_sanitize_success = 0
    decoder_sanitize_success = decoder_initial_sanitize_success = 0
    repaired = decoder_repaired = reconstruction_repaired = 0
    valence_valid = selected = 0
    components = []
    demoted_aromatic = []
    atom_counts = []
    count_bins = {}
    for record in records:
        ligand = record.get("ligand", {})
        decode = ligand.get("bond_decode_stats", {})
        reconstruction = ligand.get("reconstruction_stats", {})
        molecule = record.get("rdmol")
        is_reconstructed = molecule is not None
        is_connected = bool(
            is_reconstructed and len(Chem.GetMolFrags(molecule)) == 1
        )
        decoder_ok = bool(decode.get("sanitize_success", is_reconstructed))
        decoder_initial_ok = bool(decode.get(
            "initial_sanitize_success", decoder_ok
        ))
        reconstruction_ok = bool(reconstruction.get(
            "final_sanitize_success", is_reconstructed
        ))
        decoder_did_repair = bool(decode.get(
            "kekulize_repair_attempted", False
        ))
        reconstruction_did_repair = bool(reconstruction.get(
            "kekulize_repair_attempted", False
        ))

        reconstructed += int(is_reconstructed)
        connected += int(is_connected)
        final_sanitize_success += int(is_reconstructed and reconstruction_ok)
        decoder_sanitize_success += int(decoder_ok)
        decoder_initial_sanitize_success += int(decoder_initial_ok)
        decoder_repaired += int(decoder_did_repair)
        reconstruction_repaired += int(reconstruction_did_repair)
        repaired += int(decoder_did_repair or reconstruction_did_repair)
        valence_valid += int(bool(decode.get("valence_valid", is_reconstructed)))
        selected += int(bool(record.get("selection_metadata", {}).get(
            "selected", False
        )))
        components.append(int(decode.get(
            "num_components", 1 if is_connected else 0
        )))
        demoted_aromatic.append(int(decode.get(
            "sanitize_demoted_aromatic_bonds",
            decode.get("final_demoted_aromatic_bonds", 0),
        )))
        if record.get("atom_count") is not None:
            atom_counts.append(int(record["atom_count"]))
        bin_id = str(int(record.get("atom_count_candidate_bin", 0)))
        count_bins[bin_id] = count_bins.get(bin_id, 0) + 1

    denominator = max(total, 1)
    return {
        "total": int(total),
        "reconstructed": int(reconstructed),
        "connected": int(connected),
        "sanitize_success": int(final_sanitize_success),
        "decoder_sanitize_success": int(decoder_sanitize_success),
        "decoder_initial_sanitize_success": int(decoder_initial_sanitize_success),
        "kekulize_repair_attempted": int(repaired),
        "decoder_kekulize_repair_attempted": int(decoder_repaired),
        "reconstruction_kekulize_repair_attempted": int(reconstruction_repaired),
        "valence_valid": int(valence_valid),
        "selected": int(selected),
        "reconstruction_rate": float(reconstructed / denominator),
        "connectivity_rate_all": float(connected / denominator),
        "connectivity_rate_reconstructed": float(
            connected / max(reconstructed, 1)
        ),
        "sanitize_success_rate": float(final_sanitize_success / denominator),
        "decoder_sanitize_success_rate": float(
            decoder_sanitize_success / denominator
        ),
        "decoder_initial_sanitize_success_rate": float(
            decoder_initial_sanitize_success / denominator
        ),
        "kekulize_repair_rate": float(repaired / denominator),
        "decoder_kekulize_repair_rate": float(decoder_repaired / denominator),
        "reconstruction_kekulize_repair_rate": float(
            reconstruction_repaired / denominator
        ),
        "valence_valid_rate": float(valence_valid / denominator),
        "mean_components": float(np.mean(components)) if components else 0.0,
        "mean_sanitize_demoted_aromatic_bonds": float(
            np.mean(demoted_aromatic)
        ) if demoted_aromatic else 0.0,
        "atom_count": {
            "mean": float(np.mean(atom_counts)) if atom_counts else 0.0,
            "min": int(min(atom_counts)) if atom_counts else 0,
            "max": int(max(atom_counts)) if atom_counts else 0,
        },
        "atom_count_bins": count_bins,
    }


_CANDIDATE_COUNT_FIELDS = (
    "total", "reconstructed", "connected", "sanitize_success",
    "decoder_sanitize_success", "decoder_initial_sanitize_success",
    "kekulize_repair_attempted", "decoder_kekulize_repair_attempted",
    "reconstruction_kekulize_repair_attempted", "valence_valid", "selected",
)


def aggregate_candidate_summaries(summaries):
    summaries = [summary for summary in summaries if isinstance(summary, dict)]
    counts = {
        field: int(sum(int(summary.get(field, 0)) for summary in summaries))
        for field in _CANDIDATE_COUNT_FIELDS
    }
    total = counts["total"]
    reconstructed = counts["reconstructed"]

    def weighted_mean(field):
        numerator = 0.0
        denominator = 0
        for summary in summaries:
            case_total = int(summary.get("total", 0))
            value = summary.get(field)
            if case_total > 0 and value is not None:
                numerator += float(value) * case_total
                denominator += case_total
        return float(numerator / denominator) if denominator else 0.0

    atom_count_bins = {}
    atom_total = atom_weighted_sum = 0.0
    atom_mins, atom_maxs = [], []
    for summary in summaries:
        for bin_id, count in summary.get("atom_count_bins", {}).items():
            atom_count_bins[str(bin_id)] = (
                atom_count_bins.get(str(bin_id), 0) + int(count)
            )
        atom_count = summary.get("atom_count", {})
        case_total = int(summary.get("total", 0))
        if case_total > 0 and atom_count.get("mean") is not None:
            atom_weighted_sum += float(atom_count["mean"]) * case_total
            atom_total += case_total
            atom_mins.append(int(atom_count.get("min", 0)))
            atom_maxs.append(int(atom_count.get("max", 0)))

    denominator = max(total, 1)
    return {
        **counts,
        "reconstruction_rate": float(reconstructed / denominator),
        "connectivity_rate_all": float(counts["connected"] / denominator),
        "connectivity_rate_reconstructed": float(
            counts["connected"] / max(reconstructed, 1)
        ),
        "sanitize_success_rate": float(
            counts["sanitize_success"] / denominator
        ),
        "decoder_sanitize_success_rate": float(
            counts["decoder_sanitize_success"] / denominator
        ),
        "decoder_initial_sanitize_success_rate": float(
            counts["decoder_initial_sanitize_success"] / denominator
        ),
        "kekulize_repair_rate": float(
            counts["kekulize_repair_attempted"] / denominator
        ),
        "decoder_kekulize_repair_rate": float(
            counts["decoder_kekulize_repair_attempted"] / denominator
        ),
        "reconstruction_kekulize_repair_rate": float(
            counts["reconstruction_kekulize_repair_attempted"] / denominator
        ),
        "valence_valid_rate": float(counts["valence_valid"] / denominator),
        "mean_components": weighted_mean("mean_components"),
        "mean_sanitize_demoted_aromatic_bonds": weighted_mean(
            "mean_sanitize_demoted_aromatic_bonds"
        ),
        "atom_count": {
            "mean": float(atom_weighted_sum / atom_total) if atom_total else 0.0,
            "min": int(min(atom_mins)) if atom_mins else 0,
            "max": int(max(atom_maxs)) if atom_maxs else 0,
        },
        "atom_count_bins": atom_count_bins,
    }


def aggregate_screen_summaries(weighted_summaries):
    fields = (
        "qed", "sa", "logp", "distance_mol_stable",
        "distance_atom_stable_fraction", "phore_realization_ratio",
        "all_phores_realized", "property_filter_pass", "multiobjective_score",
    )
    result = {}
    for field in fields:
        numerator = 0.0
        denominator = 0
        for summary, weight in weighted_summaries:
            value = summary.get(field) if isinstance(summary, dict) else None
            if value is None or int(weight) <= 0:
                continue
            numerator += float(value) * int(weight)
            denominator += int(weight)
        result[field] = float(numerator / denominator) if denominator else None
    return result


def write_case_candidate_audit(
    case_dir, raw_records, selected_records, sampling_audit=None,
):
    raw = summarize_candidate_records(raw_records)
    selected = summarize_candidate_records(selected_records)
    raw_screen = screen_summary(raw_records)
    selected_screen = screen_summary(selected_records)
    audit = {
        "raw": raw,
        "selected": selected,
        "raw_screen": raw_screen,
        "selected_screen": selected_screen,
        "selection_lift": {
            "reconstruction_rate": float(
                selected["reconstruction_rate"] - raw["reconstruction_rate"]
            ),
            "connectivity_rate_all": float(
                selected["connectivity_rate_all"] - raw["connectivity_rate_all"]
            ),
            "sanitize_success_rate": float(
                selected["sanitize_success_rate"] - raw["sanitize_success_rate"]
            ),
            "kekulize_repair_rate": float(
                selected["kekulize_repair_rate"] - raw["kekulize_repair_rate"]
            ),
        },
        "selection_uses_reference_ligand": False,
        "selection_uses_docking": False,
    }
    if sampling_audit is not None:
        audit["candidate_sampling"] = sampling_audit
    with open(os.path.join(case_dir, "candidate_audit.json"), "w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    return audit


def write_run_candidate_audit(log_dir):
    selected_records = []
    raw_summaries = []
    raw_screen_summaries = []
    case_count = 0
    sampling_cases = []
    for name in sorted(os.listdir(log_dir)):
        case_dir = os.path.join(log_dir, name)
        if not os.path.isdir(case_dir):
            continue
        raw_path = os.path.join(case_dir, "raw_candidates.pt")
        selected_path = os.path.join(case_dir, "joint_samples.pt")
        if not os.path.isfile(selected_path):
            continue
        try:
            selected = torch.load(
                selected_path, map_location="cpu", weights_only=False
            )
            selected_records.extend(selected)

            case_audit = None
            case_audit_path = os.path.join(case_dir, "candidate_audit.json")
            if os.path.isfile(case_audit_path):
                try:
                    with open(case_audit_path) as handle:
                        case_audit = json.load(handle)
                except (OSError, TypeError, ValueError):
                    case_audit = None

            if isinstance(case_audit, dict) and isinstance(
                case_audit.get("raw"), dict
            ):
                raw_summary = case_audit["raw"]
                raw_summaries.append(raw_summary)
                raw_screen_summaries.append((
                    case_audit.get("raw_screen", {}),
                    int(raw_summary.get("total", 0)),
                ))
            elif os.path.isfile(raw_path):
                raw_records = torch.load(
                    raw_path, map_location="cpu", weights_only=False
                )
                raw_summary = summarize_candidate_records(raw_records)
                raw_summaries.append(raw_summary)
                raw_screen_summaries.append((
                    screen_summary(raw_records), len(raw_records)
                ))

            if isinstance(case_audit, dict):
                sampling = case_audit.get("candidate_sampling")
                if sampling is not None:
                    sampling_cases.append(sampling)
            case_count += 1
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
            continue

    audit = {
        "pockets": int(case_count),
        "raw": aggregate_candidate_summaries(raw_summaries),
        "selected": summarize_candidate_records(selected_records),
        "raw_screen": aggregate_screen_summaries(raw_screen_summaries),
        "selected_screen": screen_summary(selected_records),
        "selection_uses_reference_ligand": False,
        "selection_uses_docking": False,
    }
    if sampling_cases:
        used_batches = [
            int(case["used_batches"])
            for case in sampling_cases if case.get("used_batches") is not None
        ]
        generated_counts = [
            int(case["generated_candidates"])
            for case in sampling_cases if case.get("generated_candidates") is not None
        ]
        unique_feasible_counts = [
            int(case["available_unique_feasible_candidates"])
            for case in sampling_cases
            if case.get("available_unique_feasible_candidates") is not None
        ]
        fallback_counts = [
            int(case.get("fallback_selected", 0)) for case in sampling_cases
        ]
        elapsed_seconds = [
            float(case["case_elapsed_seconds"])
            for case in sampling_cases if case.get("case_elapsed_seconds") is not None
        ]
        stop_reasons = {}
        for case in sampling_cases:
            reason = str(case.get("stop_reason", "unknown"))
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        protocols = sorted({
            str(case.get("protocol", "unknown")) for case in sampling_cases
        })
        audit["candidate_sampling"] = {
            "protocols": protocols,
            "cases": len(sampling_cases),
            "mean_used_batches": (
                float(np.mean(used_batches)) if used_batches else None
            ),
            "min_used_batches": (
                int(min(used_batches)) if used_batches else None
            ),
            "max_used_batches": (
                int(max(used_batches)) if used_batches else None
            ),
            "mean_generated_candidates": (
                float(np.mean(generated_counts)) if generated_counts else None
            ),
            "mean_unique_feasible_candidates": (
                float(np.mean(unique_feasible_counts))
                if unique_feasible_counts else None
            ),
            "total_fallback_selected": int(sum(fallback_counts)),
            "total_case_elapsed_seconds": (
                float(sum(elapsed_seconds)) if elapsed_seconds else None
            ),
            "mean_case_elapsed_seconds": (
                float(np.mean(elapsed_seconds)) if elapsed_seconds else None
            ),
            "stop_reasons": stop_reasons,
        }
    with open(os.path.join(log_dir, "candidate_audit.json"), "w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    return audit


def _candidate_quality(record, screen_config=None):
    ligand = record.get("ligand", {})
    decode = ligand.get("bond_decode_stats", {})
    reconstruction = ligand.get("reconstruction_stats", {})
    molecule = record.get("rdmol")
    connected = bool(molecule is not None and len(Chem.GetMolFrags(molecule)) == 1)
    atom_probability = np.asarray(ligand.get("atom_prob", []), dtype=np.float64)
    bond_probability = np.asarray(ligand.get("bond_prob", []), dtype=np.float64)
    confidence = (
        float(atom_probability.mean()) if atom_probability.size else 0.0
    ) + 0.5 * (
        float(bond_probability.mean()) if bond_probability.size else 0.0
    )
    quality = (
        int(molecule is not None),
        int(connected),
        int(bool(decode.get("sanitize_success", molecule is not None))),
        int(not bool(
            decode.get("kekulize_repair_attempted", False)
            or reconstruction.get("kekulize_repair_attempted", False)
        )),
        -int(decode.get("num_components", 1)),
    )
    screen_enabled = bool(
        screen_config.get("enabled", False)
        if isinstance(screen_config, dict) else
        getattr(screen_config, "enabled", False)
        if screen_config is not None else False
    )
    if screen_enabled:
        screen = record.get("candidate_screen", {})
        quality += (
            int(bool(screen.get("screen_filter_pass", False))),
            float(screen.get("multiobjective_score", 0.0)),
            float(screen.get("distance_mol_stable", 0.0)),
            float(screen.get("phore_realization_ratio", 0.0)),
        )
    return quality + (confidence,)


def select_pocket_candidates(
    records, requested, num_count_bins, screen_config=None
):
    requested = int(requested)
    screen_enabled = bool(
        screen_config.get("enabled", False)
        if isinstance(screen_config, dict) else
        getattr(screen_config, "enabled", False)
        if screen_config is not None else False
    )
    if screen_enabled:
        for record in records:
            if "candidate_screen" not in record:
                annotate_candidate(record, config=screen_config)
    retain_strata = bool(
        screen_config.get("retain_atom_count_strata", True)
        if screen_enabled and isinstance(screen_config, dict) else
        getattr(screen_config, "retain_atom_count_strata", True)
        if screen_enabled else True
    )
    method = (
        "multiobjective_stratified_atom_count"
        if screen_enabled and retain_strata else
        "multiobjective_global" if screen_enabled else
        "chemistry_confidence_stratified_atom_count"
    )
    for record in records:
        record["selection_metadata"] = {
            "selected": False,
            "rank": None,
            "method": method,
            "uses_reference_ligand": False,
            "uses_docking": False,
        }
    quality_key = lambda record: _candidate_quality(record, screen_config)
    if not retain_strata:
        selected = sorted(records, key=quality_key, reverse=True)[:requested]
        for rank, record in enumerate(selected):
            record["selection_metadata"].update({
                "selected": True,
                "rank": int(rank),
            })
        return selected

    num_count_bins = max(int(num_count_bins), 1)
    groups = {index: [] for index in range(num_count_bins)}
    for record in records:
        group = int(record.get("atom_count_candidate_bin", 0)) % num_count_bins
        groups[group].append(record)
    for group in groups.values():
        group.sort(key=quality_key, reverse=True)

    selected = []
    while len(selected) < requested:
        advanced = False
        for group_id in range(num_count_bins):
            if groups[group_id] and len(selected) < requested:
                selected.append(groups[group_id].pop(0))
                advanced = True
        if not advanced:
            break
    if len(selected) < requested:
        leftovers = [record for group in groups.values() for record in group]
        leftovers.sort(key=quality_key, reverse=True)
        selected.extend(leftovers[:requested - len(selected)])
    for rank, record in enumerate(selected):
        record["selection_metadata"].update({
            "selected": True,
            "rank": int(rank),
        })
    return selected[:requested]


def _candidate_is_feasible(record):
    molecule = record.get("rdmol")
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return False
    ligand = record.get("ligand", {})
    decode = ligand.get("bond_decode_stats", {})
    reconstruction = ligand.get("reconstruction_stats", {})
    if not bool(decode.get("sanitize_success", True)):
        return False
    if not bool(reconstruction.get("final_sanitize_success", True)):
        return False
    if not bool(decode.get("valence_valid", True)):
        return False
    if bool(decode.get("kekulize_repair_attempted", False)):
        return False
    if bool(reconstruction.get("kekulize_repair_attempted", False)):
        return False
    screen = record.get("candidate_screen", {})
    return bool(screen.get("screen_filter_pass", False))


def _candidate_smiles_key(record):
    molecule = record.get("rdmol")
    if molecule is None:
        return None
    try:
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except (RuntimeError, ValueError):
        return record.get("smiles")


def unique_feasible_candidates(records, deduplicate_smiles=True):
    feasible = []
    seen = set()
    for record in records:
        if not _candidate_is_feasible(record):
            continue
        if deduplicate_smiles:
            key = _candidate_smiles_key(record)
            if key is None or key in seen:
                continue
            seen.add(key)
        feasible.append(record)
    return feasible


def quota_batch_status(
    records,
    requested,
    num_count_bins,
    screen_config,
    settings,
    batch_index,
):
    for record in records:
        if "candidate_screen" not in record:
            annotate_candidate(record, config=screen_config)
    feasible = unique_feasible_candidates(
        records, deduplicate_smiles=settings["deduplicate_smiles"]
    )
    preview = select_pocket_candidates(
        feasible,
        min(int(requested), len(feasible)),
        num_count_bins,
        screen_config=screen_config,
    )
    scores = [
        float(record.get("candidate_screen", {}).get("multiobjective_score", 0.0))
        for record in preview
    ]
    required = int(settings.get("required_feasible_candidates", requested))
    reached_quota = len(feasible) >= required
    at_maximum = int(batch_index) >= int(settings["max_batches"])
    if reached_quota:
        stop_reason = "feasible_quota"
    elif at_maximum:
        stop_reason = "max_batches"
    else:
        stop_reason = None
    all_feasible_count = sum(_candidate_is_feasible(record) for record in records)
    return {
        "batch_index": int(batch_index),
        "generated_candidates": int(len(records)),
        "feasible_candidates": int(all_feasible_count),
        "unique_feasible_candidates": int(len(feasible)),
        "duplicate_feasible_candidates": int(all_feasible_count - len(feasible)),
        "required_feasible_candidates": int(required),
        "remaining_feasible_candidates": int(max(required - len(feasible), 0)),
        "top_selected_mean_score": float(np.mean(scores)) if scores else 0.0,
        "reached_feasible_quota": bool(reached_quota),
        "stop": stop_reason is not None,
        "stop_reason": stop_reason,
    }


def select_direct_candidates(records, requested, screen_config=None):
    requested = int(requested)
    selected = list(records[:requested])
    for record in records:
        if "candidate_screen" not in record:
            annotate_candidate(record, config=screen_config)
        metadata = record.setdefault("selection_metadata", {})
        metadata.update({"selected": False, "rank": None, "method": "direct_one_shot"})
    strict_count = 0
    for rank, record in enumerate(selected):
        strict_feasible = bool(_candidate_is_feasible(record))
        strict_count += int(strict_feasible)
        record["selection_metadata"].update({
            "selected": True,
            "rank": int(rank),
            "strict_feasible": strict_feasible,
            "uses_reference_ligand": False,
            "uses_docking": False,
        })
    return selected, {
        "available_unique_feasible_candidates": int(strict_count),
        "strict_selected": int(strict_count),
        "fallback_selected": int(len(selected) - strict_count),
    }


def select_quota_candidates(
    records,
    requested,
    num_count_bins,
    screen_config,
    deduplicate_smiles=True,
):
    requested = int(requested)
    for record in records:
        if "candidate_screen" not in record:
            annotate_candidate(record, config=screen_config)
    feasible = unique_feasible_candidates(records, deduplicate_smiles)
    strict_count = min(requested, len(feasible))
    selected = select_pocket_candidates(
        feasible, strict_count, num_count_bins, screen_config=screen_config
    )
    selected_ids = {id(record) for record in selected}
    fallback_count = requested - len(selected)
    if fallback_count > 0:
        fallback_pool = [
            record for record in records if id(record) not in selected_ids
        ]
        fallback = select_pocket_candidates(
            fallback_pool,
            fallback_count,
            num_count_bins,
            screen_config=screen_config,
        )
        selected.extend(fallback)
    for record in records:
        metadata = record.setdefault("selection_metadata", {})
        metadata.update({"selected": False, "rank": None})
    for rank, record in enumerate(selected):
        record.setdefault("selection_metadata", {}).update({
            "selected": True,
            "rank": int(rank),
            "strict_feasible": bool(_candidate_is_feasible(record)),
        })
    return selected[:requested], {
        "available_unique_feasible_candidates": int(len(feasible)),
        "strict_selected": int(strict_count),
        "fallback_selected": int(max(0, requested - strict_count)),
    }


def main(args):
    from models.model import PADiff, ligand_atom_class_layout

    config = load_config(args.config)
    validate_clean_sampling_config(config)
    screen_config = apply_candidate_screen_overrides(
        config.sample,
        min_qed=args.min_qed,
        min_sa=args.min_sa,
        max_logp=args.max_logp,
        hard_property_filter=args.hard_property_filter,
    )
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.num_mols is not None:
        config.sample.num_mols = int(args.num_mols)
    if args.max_cases is not None:
        config.sample.max_cases = int(args.max_cases)
    seed_all(config.sample.seed)

    checkpoint_path = getattr(config.model, "checkpoint", None)
    if not checkpoint_path:
        raise ValueError("Set model.checkpoint to a checkpoint trained with configs/train.yml")
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    train_config = checkpoint["config"]
    if not bool(getattr(train_config.model, "condition_on_pocket_only", False)):
        raise ValueError("The checkpoint was not trained in pocket-only mode")
    generation_metadata = checkpoint.get("generation_metadata", {})
    if generation_metadata and (
        generation_metadata.get("conditioning") != "protein_pocket_only"
        or int(generation_metadata.get("stages", 0)) != 1
        or generation_metadata.get("uses_reference_pharmacophore_condition", True)
        or generation_metadata.get("atom_capacity_mode")
        != "pocket_prior_exact_atom_count"
    ):
        raise ValueError(
            "Checkpoint metadata is incompatible with exact-count, single-stage "
            "pocket-only generation. Retrain with configs/train.yml."
        )
    cardinality_prior = checkpoint.get("cardinality_prior")
    if cardinality_prior is None:
        raise ValueError("Checkpoint has no training-split cardinality prior")

    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    featurizer = FeatureComplex(ligand_atom_mode, sample=True)
    transform = Compose([featurizer])
    if config.sample.mode == "test":
        _, subsets = get_dataset(config=config.data, transform=transform)
        data_list = subsets["test"]
        max_cases = int(getattr(config.sample, "max_cases", len(data_list)))
        data_list = [data_list[i] for i in range(min(max_cases, len(data_list)))]
    elif config.sample.mode == "pocket":
        data_list = [transform(parse_pocket_pdb(config.model.target))]
    else:
        raise ValueError("sample.mode must be 'test' or 'pocket'")

    requested = int(config.sample.num_mols)
    config_filename = os.path.basename(args.config)
    log_prefix = os.path.splitext(config_filename)[0]
    log_dir, completed_cases = find_resumable_log_dir(
        args.outdir, log_prefix, config, config_filename,
        requested, len(data_list),
    )
    resumed = log_dir is not None
    if not resumed:
        log_dir = get_new_log_dir(args.outdir, prefix=log_prefix)
        with open(os.path.join(log_dir, config_filename), "w") as config_file:
            yaml.safe_dump(plain_config(config), config_file, sort_keys=False)
    logger = get_logger("sample", log_dir)
    logger.info(
        "Config=%s | checkpoint=%s | device=%s | run_dir=%s",
        args.config, checkpoint_path, args.device, log_dir,
    )
    logger.info(
        "[Candidate screen] hard=%s | min_qed=%.3f | min_sa=%.3f | max_logp=%.3f",
        bool(screen_config.hard_property_filter),
        float(screen_config.min_qed),
        float(screen_config.min_sa),
        float(screen_config.max_logp),
    )
    if resumed:
        logger.info(
            "[Resume] Reusing %s | completed cases=%d/%d",
            log_dir, len(completed_cases), len(data_list),
        )

    atom_layout = ligand_atom_class_layout(
        featurizer.atom_feat_dim, train_config.model
    )
    ligand_node_types = atom_layout["ligand_node_types"]
    model = PADiff(
        config=train_config.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=ligand_node_types,
        real_atom_type_count=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for case_id, data in enumerate(tqdm(data_list, desc="Pocket")):
        if case_id in completed_cases:
            continue
        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+", "-",
            str(getattr(data, "protein_filename", case_id)),
        )
        case_dir = os.path.join(log_dir, f"{case_id:04d}_{safe_name[-80:]}")
        os.makedirs(case_dir, exist_ok=True)
        generated = []
        case_started_at = time.perf_counter()
        quota_batch_started_at = case_started_at
        sampling_settings = sampling_protocol_settings(config.sample, requested)
        batch_history = []
        batch_index = 1
        candidate_target = sampling_settings["initial_candidates"]
        final_candidate_target = sampling_settings["max_generated_candidates"]
        batch_candidate_start = 0
        stop_reason = None
        count_bins = max(int(getattr(config.sample, "atom_count_num_bins", 1)), 1)
        produced = 0
        while produced < final_candidate_target:
            n_graphs = min(
                int(args.batch_size or config.sample.batch_size),
                candidate_target - produced,
            )
            batch = Batch.from_data_list(
                [pocket_context_only(data) for _ in range(n_graphs)],
                follow_batch=["protein_element"],
            ).to(args.device)
            atom_counts = sample_ligand_atom_counts(
                config, data, n_graphs, cardinality_prior
            )
            pharmacophore_counts = (
                sample_pharmacophore_counts(
                    config, data, atom_counts, n_graphs, cardinality_prior
                )
                if model.use_pharmacophore else [0] * n_graphs
            )
            holder = make_data_placeholder(atom_counts, device=args.device)
            ligand_batch = holder["batch_node"]
            halfedge_index = holder["halfedge_index"]
            halfedge_batch = holder["batch_halfedge"]
            phore_batch = torch.repeat_interleave(
                torch.arange(n_graphs, device=args.device),
                torch.tensor(pharmacophore_counts, device=args.device),
            )
            logger.info(
                "Batch %d | graphs=%d | atoms=%d..%d%s",
                batch_index, n_graphs, min(atom_counts), max(atom_counts),
                (
                    " | pharmacophores=%d..%d" % (
                        min(pharmacophore_counts), max(pharmacophore_counts)
                    )
                    if model.use_pharmacophore else " | pharmacophores=off"
                ),
            )

            empty_anchor_pos = batch.protein_pos.new_empty((0, 3))
            empty_anchor_vec = batch.protein_pos.new_empty((0, 3))
            empty_anchor_type = batch.protein_element.new_empty((0,), dtype=torch.long)
            empty_anchor_batch = batch.protein_element_batch.new_empty((0,), dtype=torch.long)
            empty_anchor_confidence = batch.protein_pos.new_empty((0,))
            outputs = model.sample(
                n_graphs=n_graphs,
                protein_node=batch.protein_atom_feat.float(),
                protein_pos=batch.protein_pos,
                protein_batch=batch.protein_element_batch,
                anchor_type=empty_anchor_type,
                anchor_pos=empty_anchor_pos,
                anchor_batch=empty_anchor_batch,
                anchor_confidence=empty_anchor_confidence,
                anchor_vec=empty_anchor_vec,
                ligand_batch=ligand_batch,
                halfedge_index=halfedge_index,
                halfedge_batch=halfedge_batch,
                phore_batch=phore_batch,
            )
            split = separate_outputs(
                outputs, n_graphs, ligand_batch, halfedge_index,
                halfedge_batch, phore_batch,
            )
            for batch_item_id, item in enumerate(split):
                ligand_pred = tensor_to_numpy_list(item["pred"])
                terminal_atom_state = item.get("terminal_ligand_node_state")
                if terminal_atom_state is None:
                    terminal_atom_state = item["traj"][0][-1]
                terminal_atom_logits = terminal_real_atom_logits(
                    terminal_atom_state,
                    item["pred"][0],
                    model.real_atom_type_count,
                ).cpu().numpy()
                molecule_info = featurizer.decode_output(
                    pred_node=terminal_atom_logits,
                    pred_pos=ligand_pred[1],
                    pred_halfedge=ligand_pred[2],
                    halfedge_index=item["halfedge_index"].cpu().numpy(),
                )
                if len(molecule_info["element"]) != int(atom_counts[batch_item_id]):
                    raise RuntimeError("Exact atom-count decoding invariant was violated")

                phore_probability = torch.softmax(
                    item["phore_pred"][0].float(), dim=-1
                )
                halfedge_probability = torch.softmax(
                    item["pred"][2].float(), dim=-1
                ).cpu()
                record = {
                    "ligand": molecule_info,
                    "atom_count": int(atom_counts[batch_item_id]),
                    "atom_count_candidate_bin": int(batch_item_id % count_bins),
                    "candidate_index": int(produced),
                    "ligand_halfedge_index": item["halfedge_index"].long().cpu(),
                    "ligand_halfedge_probability": halfedge_probability,
                    "phore_type": model.phore_types_to_raw(
                        phore_probability.argmax(dim=-1)
                    ).cpu(),
                    "phore_probability": phore_probability.cpu(),
                    "phore_pos": item["phore_pred"][1].float().cpu(),
                    "phore_vec": item["phore_pred"][2].float().cpu(),
                    "protein_filename": getattr(data, "protein_filename", None),
                    "ligand_filename": getattr(data, "ligand_filename", None),
                    "generation_metadata": {
                        "conditioning": "protein_pocket_only",
                        "stages": 1,
                        "use_pharmacophore": bool(model.use_pharmacophore),
                        "pharmacophore_schema": (
                            "legacy_11_raw" if model.legacy_phore_schema
                            else "active_6_compact"
                        ),
                        "joint_outputs": ["ligand_atoms", "ligand_bonds"] + (
                            ["pharmacophores"] if model.use_pharmacophore else []
                        ),
                        "atom_count_mode": "pocket_prior_exact",
                        "cardinality_prior_version": int(
                            cardinality_prior.get("version", -1)
                        ),
                        "atom_count_candidate_bins": int(count_bins),
                        "candidate_sampling": (
                            "direct_one_shot"
                            if sampling_settings["strategy"] == "direct"
                            else "remaining_quota_batches"
                        ),
                        "quota_batch_index": int(batch_index),
                        "quota_batch_requested_candidates": int(
                            candidate_target - batch_candidate_start
                        ),
                        "quota_required_feasible_candidates": int(requested),
                        "quota_max_batches": int(
                            sampling_settings["max_batches"]
                        ),
                        "bond_decode": "rdkit_sanitize_no_edge_insertion",
                    },
                }
                assignment = item.get("phore_atom_assignment")
                if model.use_pharmacophore and assignment is not None:
                    record.update({
                        "phore_atom_assignment_index": assignment["index"].long().cpu(),
                        "phore_atom_assignment_probability": torch.sigmoid(
                            assignment["logits"].float()
                        ).cpu(),
                        "atom_phore_capability_probability": torch.sigmoid(
                            assignment["atom_type_logits"].float()
                        ).cpu(),
                    })
                try:
                    molecule = reconstruct_from_generated_with_edges(
                        molecule_info,
                        atom_is_aromatic=molecule_info.get('atom_is_aromatic'),
                    )
                except (MolReconsError, ValueError, RuntimeError) as error:
                    message = str(error).strip()
                    record["reconstruction_error"] = (
                        "%s: %s" % (type(error).__name__, message)
                        if message else type(error).__name__
                    )
                    molecule = None
                if molecule is not None:
                    record["smiles"] = Chem.MolToSmiles(molecule)
                    record["rdmol"] = molecule
                generated.append(record)
                produced += 1
            if produced >= candidate_target:
                batch_status = quota_batch_status(
                    generated,
                    requested,
                    count_bins,
                    screen_config,
                    sampling_settings,
                    batch_index,
                )
                batch_status["generated_this_batch"] = int(
                    produced - batch_candidate_start
                )
                batch_status["elapsed_seconds"] = float(
                    time.perf_counter() - quota_batch_started_at
                )
                if sampling_settings["strategy"] == "direct":
                    batch_status.update({
                        "stop": True,
                        "stop_reason": "direct_one_shot",
                    })
                batch_history.append(batch_status)
                logger.info(
                    "[%s candidates] batch=%d batch-size=%d total=%d "
                    "unique-feasible=%d/%d remaining=%d duplicates=%d "
                    "top-score=%.4f elapsed=%.2fs stop=%s",
                    "Direct" if sampling_settings["strategy"] == "direct" else "Quota",
                    batch_status["batch_index"],
                    batch_status["generated_this_batch"],
                    batch_status["generated_candidates"],
                    batch_status["unique_feasible_candidates"],
                    batch_status["required_feasible_candidates"],
                    batch_status["remaining_feasible_candidates"],
                    batch_status["duplicate_feasible_candidates"],
                    batch_status["top_selected_mean_score"],
                    batch_status["elapsed_seconds"],
                    batch_status["stop_reason"] or "continue",
                )
                if batch_status["stop"]:
                    stop_reason = batch_status["stop_reason"]
                    break
                batch_index += 1
                quota_batch_started_at = time.perf_counter()
                batch_candidate_start = produced
                candidate_target = min(
                    final_candidate_target,
                    produced + batch_status["remaining_feasible_candidates"],
                )
        if sampling_settings["strategy"] == "direct":
            selected, selection_quota = select_direct_candidates(
                generated,
                requested,
                screen_config=screen_config,
            )
        else:
            selected, selection_quota = select_quota_candidates(
                generated,
                requested,
                count_bins,
                screen_config=screen_config,
                deduplicate_smiles=sampling_settings["deduplicate_smiles"],
            )
        save_raw_candidates = bool(getattr(
            config.sample, "save_raw_candidates", True
        ))
        save_raw_sdf = bool(getattr(config.sample, "save_raw_sdf", False))
        if save_raw_candidates:
            torch.save(generated, os.path.join(case_dir, "raw_candidates.pt"))
        if save_raw_sdf:
            raw_writer = Chem.SDWriter(os.path.join(case_dir, "raw_candidates.sdf"))
            for record in generated:
                if record.get("rdmol") is not None:
                    raw_writer.write(record["rdmol"])
            raw_writer.close()
        sdf_writer = Chem.SDWriter(os.path.join(case_dir, "generated.sdf"))
        for record in selected:
            if record.get("rdmol") is not None:
                sdf_writer.write(record["rdmol"])
        sdf_writer.close()
        torch.save(selected, os.path.join(case_dir, "joint_samples.pt"))
        sampling_audit = {
            **sampling_settings,
            **selection_quota,
            "used_batches": int(batch_index),
            "generated_candidates": int(len(generated)),
            "case_elapsed_seconds": float(
                time.perf_counter() - case_started_at
            ),
            "stop_reason": stop_reason or "max_batches",
            "batch_history": batch_history,
        }
        audit = write_case_candidate_audit(
            case_dir, generated, selected,
            sampling_audit=sampling_audit,
        )
        logger.info(
            "Selected and saved %d/%d pocket-only candidates to %s | "
            "sampling=%s batches=%d/%s strict/fallback=%d/%d | "
            "raw valid/connected/sanitize/repair=%.3f/%.3f/%.3f/%.3f | "
            "selected=%.3f/%.3f/%.3f/%.3f",
            len(selected), len(generated), case_dir,
            sampling_settings["strategy"],
            sampling_audit["used_batches"], sampling_audit["stop_reason"],
            sampling_audit["strict_selected"], sampling_audit["fallback_selected"],
            audit["raw"]["reconstruction_rate"],
            audit["raw"]["connectivity_rate_all"],
            audit["raw"]["sanitize_success_rate"],
            audit["raw"]["kekulize_repair_rate"],
            audit["selected"]["reconstruction_rate"],
            audit["selected"]["connectivity_rate_all"],
            audit["selected"]["sanitize_success_rate"],
            audit["selected"]["kekulize_repair_rate"],
        )

    run_audit = write_run_candidate_audit(log_dir)
    logger.info(
        "[Candidate audit] pockets=%d | raw total=%d valid=%.3f connected=%.3f "
        "sanitize=%.3f repair=%.3f | selected total=%d valid=%.3f "
        "connected=%.3f sanitize=%.3f repair=%.3f",
        run_audit["pockets"],
        run_audit["raw"]["total"],
        run_audit["raw"]["reconstruction_rate"],
        run_audit["raw"]["connectivity_rate_all"],
        run_audit["raw"]["sanitize_success_rate"],
        run_audit["raw"]["kekulize_repair_rate"],
        run_audit["selected"]["total"],
        run_audit["selected"]["reconstruction_rate"],
        run_audit["selected"]["connectivity_rate_all"],
        run_audit["selected"]["sanitize_success_rate"],
        run_audit["selected"]["kekulize_repair_rate"],
    )
    sampling_run_audit = run_audit.get("candidate_sampling")
    if sampling_run_audit is not None:
        logger.info(
            "[Sampling audit] protocols=%s | cases=%d | "
            "batches mean/min/max=%.2f/%d/%d "
            "| mean candidates=%.2f | mean unique feasible=%.2f "
            "| fallback selected=%d | mean case time=%.2fs | stop reasons=%s",
            ",".join(sampling_run_audit["protocols"]),
            sampling_run_audit["cases"],
            sampling_run_audit["mean_used_batches"],
            sampling_run_audit["min_used_batches"],
            sampling_run_audit["max_used_batches"],
            sampling_run_audit["mean_generated_candidates"],
            sampling_run_audit["mean_unique_feasible_candidates"],
            sampling_run_audit["total_fallback_selected"],
            sampling_run_audit["mean_case_elapsed_seconds"],
            json.dumps(sampling_run_audit["stop_reasons"], sort_keys=True),
        )
    return {
        "run_dir": os.path.abspath(log_dir),
        "candidate_audit": run_audit,
        "resumed": bool(resumed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample.yml")
    parser.add_argument("--outdir", default="sample_outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Override model.checkpoint from the YAML config.",
    )
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument(
        "--num_mols", type=int, default=None,
        help="Override sample.num_mols from the YAML config.",
    )
    parser.add_argument(
        "--max_cases", type=int, default=None,
        help="Override sample.max_cases from the YAML config.",
    )
    parser.add_argument(
        "--min-qed", "--min_qed", dest="min_qed", type=float, default=None,
        help="Override the default candidate QED threshold for this run.",
    )
    parser.add_argument(
        "--min-sa", "--min_sa", dest="min_sa", type=float, default=None,
        help="Override the default candidate SA threshold for this run.",
    )
    parser.add_argument(
        "--max-logp", "--max_logp", dest="max_logp", type=float, default=None,
        help="Override the default candidate LogP threshold for this run.",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--hard-property-filter", dest="hard_property_filter",
        action="store_true",
        help="Require QED/SA/LogP thresholds when counting quota candidates.",
    )
    filter_group.add_argument(
        "--soft-property-filter", dest="hard_property_filter",
        action="store_false",
        help="Use QED/SA/LogP only for audit/ranking, not quota feasibility.",
    )
    parser.set_defaults(hard_property_filter=None)
    main(parser.parse_args())
