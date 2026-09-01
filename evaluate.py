import argparse
import csv
import json
import os
import pickle
import sys
sys.path.append('.')
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
import yaml
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
from utils.misc import load_config
from utils.dataset import get_dataset
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length, eval_all
from utils import misc
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from utils.evaluation.generation_metrics import (
    evaluate_generated_records,
    load_or_build_reference_scaffolds,
)
from utils.phore_realization import evaluate_pharmacophore_realization
from utils.candidate_screen import (
    DEFAULT_MAX_LOGP, DEFAULT_MIN_QED, DEFAULT_MIN_SA,
)

VINA_DOCKING_SEED = 2023


def _case_study_config(config):
    settings = getattr(config, 'case_study', None)
    return settings if settings and bool(settings.get('enabled', False)) else None


def _case_study_docking_kwargs(config):
    settings = _case_study_config(config)
    if settings is None:
        return {}
    center = list(settings.get('docking_center') or [])
    box_size = list(settings.get('docking_box_size') or [])
    if len(center) != 3 or len(box_size) != 3:
        raise ValueError(
            'case_study.docking_center and docking_box_size must each contain 3 values'
        )
    return {
        'center': [float(value) for value in center],
        'box_size': [float(value) for value in box_size],
    }


def _vina_task(config, mol, ligand_filename=None, pdbid=None):
    settings = _case_study_config(config)
    if settings is not None:
        receptor_path = os.path.abspath(os.path.expanduser(
            str(settings['receptor_path'])
        ))
        return VinaDockingTask.from_generated_mol(
            mol, receptor_path, **_case_study_docking_kwargs(config)
        )
    if config.dataset == 'pdbbind':
        return VinaDockingTask.from_generated_mol_pdbbind(
            mol, pdbid, protein_root=config.protein_root
        )
    if config.dataset == 'crossdocked':
        return VinaDockingTask.from_generated_mol_crossdocked(
            mol, ligand_filename, protein_root=config.protein_root
        )
    raise ValueError('Unsupported dataset for Vina docking: %s' % config.dataset)


def _reference_docking_cache_key(config, ligand_filename, affinity_mode):
    settings = _case_study_config(config)
    if settings is None:
        identity = '%s|%s' % (config.dataset, ligand_filename)
    else:
        docking = _case_study_docking_kwargs(config)
        identity = '%s|%s|%s|%s' % (
            settings.get('id', 'external_case'),
            os.path.abspath(os.path.expanduser(str(settings['receptor_path']))),
            ','.join('%.4f' % value for value in docking['center']),
            ','.join('%.4f' % value for value in docking['box_size']),
        )
    return 'same_mode_v4|%s|%s|%s|seed=%s' % (
        identity, affinity_mode, config.exhaustiveness, VINA_DOCKING_SEED
    )

def _load_generated_file(path):
    """Load either legacy sample files or PADiff joint_samples.pt."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'finished' in payload and 'failed' in payload:
        finished, failed = payload['finished'], payload['failed']
        for sample_idx, mol_info in enumerate(finished):
            mol_info.setdefault('_sample_index', sample_idx)
        for failed_idx, mol_info in enumerate(failed):
            mol_info.setdefault('_sample_index', len(finished) + failed_idx)
        return finished, failed, None
    if not isinstance(payload, list):
        raise ValueError(f'Unsupported sample format: {path}')

    finished, failed = [], []
    ligand_filename = None
    for sample_idx, record in enumerate(payload):
        ligand_filename = ligand_filename or record.get('ligand_filename')
        mol_info = dict(record.get('ligand', {}))
        mol_info['_sample_index'] = sample_idx
        mol_info['_protein_filename'] = record.get('protein_filename')
        phore_type = record.get('phore_type')
        mol_info['_num_pharmacophores'] = (
            int(phore_type.numel()) if hasattr(phore_type, 'numel')
            else len(phore_type) if phore_type is not None else None
        )
        mol_info['_generated_phore_type'] = phore_type
        mol_info['_generated_phore_pos'] = record.get('phore_pos')
        mol_info['_generated_phore_vec'] = record.get('phore_vec')
        mol_info['_phore_atom_assignment_index'] = record.get(
            'phore_atom_assignment_index')
        mol_info['_phore_atom_assignment_probability'] = record.get(
            'phore_atom_assignment_probability')
        if 'rdmol' in record and 'smiles' in record and '.' not in record['smiles']:
            mol_info['rdmol'] = record['rdmol']
            mol_info['smiles'] = record['smiles']
            finished.append(mol_info)
        else:
            if 'rdmol' in record:
                mol_info['rdmol'] = record['rdmol']
            if 'smiles' in record:
                mol_info['smiles'] = record['smiles']
            if 'reconstruction_error' in record:
                mol_info['reconstruction_error'] = record['reconstruction_error']
            failed.append(mol_info)
    return finished, failed, ligand_filename



def _validate_sampling_provenance(config, logger):
    sample_path = getattr(config, 'sample_path', None)
    if not sample_path:
        raise ValueError('sample_path is required; pass --sample_path.')

    config_paths = sorted(glob(os.path.join(sample_path, '*.yml')))
    if not config_paths:
        raise ValueError(
            'No saved sampling YAML found under %s; sampling provenance cannot '
            'be verified.' % sample_path
        )

    for path in config_paths:
        try:
            with open(path) as handle:
                saved = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                'Could not inspect sampling config %s: %s' % (path, exc)
            ) from exc

        sample = saved.get('sample', {}) or {}
        protocol = str(sample.get('protocol', '')).lower()
        if protocol not in {'adaptive', 'direct'}:
            raise ValueError(
                "Unsupported sampling provenance in %s: sample.protocol=%r"
                % (path, sample.get('protocol'))
            )
        logger.info('Verified %s sampling provenance: %s', protocol, path)

def _reference_ligand_path(config, ligand_filename):
    if config.dataset == 'crossdocked':
        if not ligand_filename:
            raise ValueError('PADiff sample record is missing ligand_filename')
        return os.path.join(config.protein_root, ligand_filename)
    raise ValueError('Explicit ligand filename is only used for CrossDocked samples')


def _dataset_ligand_path(config, data):
    root = config.protein_root if config.dataset == 'crossdocked' else config.data.path
    return os.path.join(root, data.ligand_filename)


_REFERENCE_CACHE_VERSION = 2


def _reference_cache_path(config):
    configured_path = getattr(config, 'reference_cache', None)
    if configured_path:
        return os.path.abspath(os.path.expanduser(configured_path))
    return os.path.abspath(os.path.expanduser(config.data.path + '.eval_reference_cache.pkl'))


def _reference_cache_metadata(config, train_set):
    return {
        'version': _REFERENCE_CACHE_VERSION,
        'dataset': config.dataset,
        'data_path': os.path.abspath(os.path.expanduser(config.data.path)),
        'split_path': os.path.abspath(os.path.expanduser(config.data.split)),
        'protein_root': os.path.abspath(os.path.expanduser(config.protein_root)),
        'train_size': len(train_set),
        'reference_scope': 'train_only',
        'fingerprint': 'morgan_radius2_2048bits',
    }


def _load_reference_cache(cache_path, expected_metadata, logger):
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('metadata') != expected_metadata:
            logger.info('Reference cache metadata changed; rebuilding %s', cache_path)
            return None
        smiles = cache['smiles']
        fingerprints = cache['fingerprints']
        if len(smiles) != len(fingerprints):
            raise ValueError('SMILES/fingerprint count mismatch')
        logger.info('Loaded reference cache: %d molecules from %s', len(smiles), cache_path)
        return smiles, fingerprints
    except Exception as exc:
        logger.warning('Failed to load reference cache %s (%s); rebuilding it', cache_path, exc)
        return None


def _save_reference_cache(cache_path, metadata, smiles, fingerprints, logger):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    temporary_path = cache_path + '.tmp.%d' % os.getpid()
    try:
        with open(temporary_path, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'smiles': smiles,
                'fingerprints': fingerprints,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, cache_path)
        logger.info('Saved reference cache: %d molecules to %s', len(smiles), cache_path)
    except Exception as exc:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        logger.warning('Failed to save reference cache %s: %s', cache_path, exc)


def _morgan_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _canonical_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _build_reference_cache(config, train_set, logger):
    reference_smiles = []
    reference_fingerprints = []
    for data in tqdm(train_set, desc='Get train mols'):
        ligand_rdmol = Chem.SDMolSupplier(_dataset_ligand_path(config, data))[0]
        if ligand_rdmol is not None:
            reference_smiles.append(_canonical_smiles(ligand_rdmol))
            reference_fingerprints.append(_morgan_fingerprint(ligand_rdmol))
    return reference_smiles, reference_fingerprints


def _get_reference_data(config, train_set, logger):
    cache_path = _reference_cache_path(config)
    metadata = _reference_cache_metadata(config, train_set)
    cached = _load_reference_cache(cache_path, metadata, logger)
    if cached is not None:
        return cached

    logger.info('Building train-only ECFP4 reference cache from %d molecules',
                len(train_set))
    smiles, fingerprints = _build_reference_cache(config, train_set, logger)
    _save_reference_cache(cache_path, metadata, smiles, fingerprints, logger)
    return smiles, fingerprints

_MAIN_MOLECULE_CSV_FIELDS = [
    'pocket_id', 'ligand_id', 'status', 'smiles',
    'affinity', 'affinity_reference',
    'qed', 'reference_qed',
    'sa', 'reference_sa',
    'logp', 'reference_logp',
    'lipinski', 'reference_lipinski',
    'tpsa', 'reference_tpsa',
    'phore_realization_ratio', 'all_phores_realized',
    'any_phore_realized', 'generated_phore_count',
    'molecule_extracted_phore_count', 'matched_phore_count',
    'validity', 'connectivity', 'evaluation_success',
    'novelty', 'uniqueness', 'diversity',
    'scaffold_smiles', 'scaffold_novel',
    'internal_diversity_2d', 'internal_diversity_3d',
    'better_than_reference', 'passes_topk_filter', 'topk_rank',
]

_DETAIL_MOLECULE_CSV_FIELDS = [
    'pocket_id', 'ligand_id', 'failure_reason',
    'reconstruction_success', 'num_atoms', 'num_pharmacophores',
    'distance_inferred_mol_stable',
    'distance_inferred_stable_atoms',
    'distance_inferred_atom_stable_fraction',
    'generated_graph_valence_valid',
    'generated_graph_num_overvalent_atoms',
    'generated_graph_overvalent_atom_fraction',
    'generated_graph_num_components',
    'generated_graph_selected_bonds',
    'generated_graph_positive_candidate_bonds',
    'generated_graph_added_connectivity_bonds',
    'sim_with_ref',
    'mean_phore_match_distance', 'mean_phore_direction_similarity',
    'mean_max_assignment_probability', 'assignment_coverage_ratio',
    'phore_realization_by_type', 'phore_matches',
    'ring_size_3', 'ring_size_4', 'ring_size_5', 'ring_size_6',
    'ring_size_7', 'ring_size_8', 'ring_size_9',
    'vina_score', 'vina_minimize', 'vina_dock', 'qvina',
]

_POCKET_MAIN_CSV_FIELDS = [
    'pocket_id', 'num_samples',
    'num_sanitized_valid', 'num_unique_structures',
    'num_scaffold_bearing', 'num_unique_scaffolds',
    'num_novel_scaffolds',
    'affinity', 'affinity_reference',
    'qed', 'reference_qed',
    'sa', 'reference_sa',
    'logp', 'reference_logp',
    'lipinski', 'reference_lipinski',
    'tpsa', 'reference_tpsa',
    'phore_realization_ratio', 'all_phores_realized',
    'any_phore_realized',
    'validity', 'novelty', 'uniqueness', 'diversity',
    'internal_diversity_2d', 'internal_diversity_3d',
    'high_affinity_ratio', 'num_topk_filtered',
    'affinity_filtered_top10',
]

_POCKET_DETAIL_CSV_FIELDS = [
    'pocket_id', 'num_reconstructed', 'num_connected', 'num_evaluated',
    'recon_success', 'eval_success', 'connectivity',
    'distance_inferred_mol_stable', 'distance_inferred_atom_stable',
    'generated_graph_valence_valid_ratio',
    'generated_graph_overvalent_atom_fraction',
    'generated_graph_mean_components',
    'generated_graph_mean_selected_bonds',
    'generated_graph_mean_positive_candidate_bonds',
    'sim_with_ref',
    'mean_phore_match_distance', 'mean_phore_direction_similarity',
    'mean_max_assignment_probability', 'assignment_coverage_ratio',
    'vina_score', 'vina_minimize', 'vina_dock', 'qvina',
]


def _bond_decode_metrics(mol_info):
    stats = mol_info.get('bond_decode_stats', {}) or {}
    mol = mol_info.get('rdmol')
    component_count = stats.get('num_components')
    if mol is not None:
        try:
            component_count = len(Chem.GetMolFrags(
                mol, asMols=False, sanitizeFrags=False
            ))
        except Exception:
            pass

    def _value(key):
        value = stats.get(key)
        if hasattr(value, 'item'):
            value = value.item()
        return value

    return {
        'generated_graph_valence_valid': _value('valence_valid'),
        'generated_graph_num_overvalent_atoms': _value('num_overvalent_atoms'),
        'generated_graph_overvalent_atom_fraction': _value(
            'overvalent_atom_fraction'
        ),
        'generated_graph_num_components': component_count,
        'generated_graph_selected_bonds': _value('selected_bonds'),
        'generated_graph_positive_candidate_bonds': _value(
            'positive_candidate_bonds'
        ),
        'generated_graph_added_connectivity_bonds': _value(
            'added_connectivity_bonds'
        ),
    }


def _assignment_summary(mol_info, threshold=0.5):
    index = mol_info.get('_phore_atom_assignment_index')
    probability = mol_info.get('_phore_atom_assignment_probability')
    num_phore = mol_info.get('_num_pharmacophores')
    if index is None or probability is None or not num_phore:
        return {
            'mean_max_assignment_probability': None,
            'assignment_coverage_ratio': None,
        }
    if hasattr(index, 'detach'):
        index = index.detach().cpu().numpy()
    if hasattr(probability, 'detach'):
        probability = probability.detach().cpu().numpy()
    index = np.asarray(index)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    maxima = []
    for phore_id in range(int(num_phore)):
        values = probability[index[0] == phore_id] if index.size else np.asarray([])
        maxima.append(float(values.max()) if values.size else 0.0)
    return {
        'mean_max_assignment_probability': float(np.mean(maxima)),
        'assignment_coverage_ratio': float(np.mean(
            np.asarray(maxima) >= float(threshold)
        )),
    }


def _first_affinity(vina_results, key):
    try:
        value = float(vina_results[key][0]['affinity'])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value < 0 else None


def _numbered_molecule_row(row):
    sample_index = row.get('sample_index')
    ligand_id = sample_index + 1 if sample_index is not None else None
    status = row.get('status')
    if status == 'reconstruction_failed':
        validity = 0
        connectivity = None
    elif status == 'disconnected':
        validity = 1
        connectivity = 0
    else:
        validity = row.get('validity')
        if validity is None:
            validity = 1
        connectivity = 1 if row.get('reconstruction_success') else None

    numbered = dict(row)
    numbered.update({
        'pocket_id': row.get('pocket_index'),
        'ligand_id': ligand_id,
        'validity': validity,
        'connectivity': connectivity,
        'uniqueness': row.get('pocket_uniqueness'),
        'diversity': row.get('pocket_diversity'),
        'internal_diversity_2d': row.get('pocket_internal_diversity_2d'),
        'internal_diversity_3d': row.get('pocket_internal_diversity_3d'),
        'affinity': _docking_score(row),
        'affinity_reference': row.get('reference_vina_dock'),
    })
    return numbered


def _write_molecule_csv(csv_path, molecule_rows, logger, topk=False,
                        details=False):
    fieldnames = (_DETAIL_MOLECULE_CSV_FIELDS if details
                  else _MAIN_MOLECULE_CSV_FIELDS)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        if topk:
            ordered_rows = sorted(
                molecule_rows,
                key=lambda row: (row['pocket_index'], row.get('topk_rank') or 10 ** 9))
        else:
            ordered_rows = sorted(
                molecule_rows,
                key=lambda row: (row['pocket_index'], row['sample_index']))
        for molecule_row in ordered_rows:
            writer.writerow(_numbered_molecule_row(molecule_row))
    table_kind = 'detail' if details else 'main'
    logger.info('Saved per-molecule %s CSV: %d rows to %s',
                table_kind, len(molecule_rows), csv_path)


def _finite_values(values):
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite.append(value)
    return finite


def _stats(values):
    values = _finite_values(values)
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _summary_row(metric, values):
    mean, std = _stats(values)
    return {'Metric': metric, 'Mean': mean, 'Std': std}


def _write_dict_rows(csv_path, rows, fieldnames, logger, description):
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    logger.info('Saved %s: %d rows to %s', description, len(rows), csv_path)


def _numbered_pocket_rows(rows):
    numbered_rows = []
    for row in rows:
        numbered = dict(row)
        numbered['pocket_id'] = row.get('pocket_index')
        numbered['affinity'] = row.get('affinity_raw_mean')
        numbered_rows.append(numbered)
    return numbered_rows


def _docking_score(row):
    for key in ('vina_dock', 'qvina', 'vina_minimize', 'vina_score'):
        values = _finite_values([row.get(key)])
        if values and values[0] < 0:
            return values[0]
    return None


def _max_similarity_with_reference(mols, reference_fingerprints):
    similarities = []
    for mol in tqdm(mols, desc='Calculate similarity with train'):
        fingerprint = _morgan_fingerprint(mol)
        reference_similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, reference_fingerprints
        )
        similarities.append(
            float(max(reference_similarities)) if reference_similarities else None
        )
    return similarities


def _format_report_metric(label, values, scale=1.0, suffix=''):
    mean, std = _stats(values)
    if mean is None:
        return f'  - {label:30s}: N/A'
    return f'  - {label:30s}: {mean * scale:.3f} ± {std * scale:.3f}{suffix}'


def log_metric_dict(metrics, logger):
    for k, v in metrics.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def log_ring_distribution(all_ring_sizes, logger):
    sizes_count = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for counter in all_ring_sizes:
        for size in sizes_count:
            sizes_count[size] += counter.get(size, 0)
    total = sum(sizes_count.values())
    ratios = {}
    for size in sizes_count:
        ratio = sizes_count[size] / total if total else 0.0
        ratios[size] = ratio
        logger.info(f'ring size: {size} ratio: {ratio:.3f}')
    return ratios


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/eval.yml')
    parser.add_argument(
        '--sample-path', '--sample_path', dest='sample_path', type=str,
        default=None,
        help='Override sample_path from the YAML config (useful for A/B runs).'
    )
    parser.add_argument(
        '--result-path', '--result_path', dest='result_path', type=str,
        default=None,
        help='Override result_path from the YAML config.'
    )
    parser.add_argument(
        '--record-filename', '--record_filename', dest='record_filename',
        type=str, default=None,
        help='Evaluate joint_samples.pt (selected) or raw_candidates.pt (raw).'
    )
    parser.add_argument(
        '--min-qed', '--min_qed', dest='min_qed', type=float, default=None,
        help='Override the default Top-k QED threshold for this run.'
    )
    parser.add_argument(
        '--min-sa', '--min_sa', dest='min_sa', type=float, default=None,
        help='Override the default Top-k SA threshold for this run.'
    )
    parser.add_argument(
        '--max-logp', '--max_logp', dest='max_logp', type=float, default=None,
        help='Override the default Top-k LogP threshold for this run.'
    )
    parser.add_argument(
        '--top-k', '--top_k', dest='top_k', type=int, default=None,
        help='Override the number of top-ranked molecules retained per pocket.'
    )
    topk_group = parser.add_mutually_exclusive_group()
    topk_group.add_argument(
        '--filtered-top-k', dest='topk_filter_enabled', action='store_true',
        help='Apply the configured QED/SA/LogP/negative-affinity Top-k filter.'
    )
    topk_group.add_argument(
        '--unfiltered-top-k', dest='topk_filter_enabled', action='store_false',
        help='Rank all successfully evaluated molecules by affinity only.'
    )
    parser.set_defaults(topk_filter_enabled=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if (_case_study_config(config) is not None
            and config.docking_mode == 'qvina'):
        raise ValueError(
            'External case studies currently require vina_score or vina_dock; '
            'QVina path resolution is dataset-specific.'
        )
    if args.sample_path is not None:
        config.sample_path = args.sample_path
    if args.result_path is not None:
        config.result_path = args.result_path
    if args.record_filename is not None:
        config.record_filename = args.record_filename
    if getattr(config, 'topk_filter', None) is None:
        config.topk_filter = {}
    if args.min_qed is not None:
        config.topk_filter['min_qed'] = args.min_qed
    if args.min_sa is not None:
        config.topk_filter['min_sa'] = args.min_sa
    if args.max_logp is not None:
        config.topk_filter['max_logp'] = args.max_logp
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.topk_filter_enabled is not None:
        config.topk_filter['enabled'] = args.topk_filter_enabled
    topk_filter_enabled = bool(config.topk_filter.get('enabled', True))
    min_qed_override = float(config.topk_filter.get('min_qed', DEFAULT_MIN_QED))
    min_sa_override = float(config.topk_filter.get('min_sa', DEFAULT_MIN_SA))
    max_logp_override = float(config.topk_filter.get('max_logp', DEFAULT_MAX_LOGP))
    if not 0.0 <= min_qed_override <= 1.0:
        raise ValueError('topk_filter.min_qed must be in [0, 1]')
    if not 0.0 <= min_sa_override <= 1.0:
        raise ValueError('topk_filter.min_sa must be in [0, 1]')
    if not np.isfinite(max_logp_override):
        raise ValueError('topk_filter.max_logp must be finite')
    if int(getattr(config, 'top_k', 10)) < 1:
        raise ValueError('top_k must be >= 1')
    result_path = getattr(config, 'result_path', 'eval_results/test1')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    logger.info(
        '[Top-k filter] enabled=%s | min_qed=%.3f | min_sa=%.3f | max_logp=%.3f | top_k=%d',
        topk_filter_enabled, min_qed_override, min_sa_override, max_logp_override,
        int(getattr(config, 'top_k', 10)))
    _validate_sampling_provenance(config, logger)
    if not config.verbose:
        RDLogger.DisableLog('rdApp.*')

    record_filename = str(getattr(config, 'record_filename', 'joint_samples.pt'))
    if os.path.basename(record_filename) != record_filename:
        raise ValueError('record_filename must be a basename, not a path')
    results_fn_list = sorted(glob(
        os.path.join(config.sample_path, '**', record_filename), recursive=True
    ))
    if not results_fn_list and record_filename == 'joint_samples.pt':
        results_fn_list = sorted(glob(os.path.join(config.sample_path, 'samples_*.pt')))
    if config.eval_num_examples is not None:
        results_fn_list = results_fn_list[:config.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')
    if num_examples == 0:
        raise FileNotFoundError(
            f'No {record_filename} found under {config.sample_path}'
        )

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_eval_success = 0
    n_success, n_invalid, n_disconnect = 0, 0, 0
    results = []
    molecule_rows = []
    metric_records_by_pocket = {}
    evaluation_failures = Counter()
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    ligand_ref_list = []
    reference_metrics_by_pocket = {}
    reference_docking_by_pocket = {}
    reference_docking_cache_path = os.path.join(
        result_path, 'reference_docking_cache.pt')
    try:
        reference_docking_cache = torch.load(
            reference_docking_cache_path, map_location='cpu', weights_only=False)
        if not isinstance(reference_docking_cache, dict):
            reference_docking_cache = {}
    except Exception:
        reference_docking_cache = {}

    dataset, subsets = get_dataset(config=config.data)
    train_set = subsets['train']
    reference_smiles, reference_fingerprints = _get_reference_data(
        config, train_set, logger)
    reference_scaffolds, reference_scaffold_counts, reference_scaffold_cache = (
        load_or_build_reference_scaffolds(
            config, reference_smiles, subsets['test'], logger=logger
        )
    )
    logger.info(
        'Scaffold novelty reference: train union test, %d unique >=2-ring scaffolds',
        reference_scaffold_counts['reference_unique_scaffolds'])

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        finished_mols, failed_mols, ligand_filename = _load_generated_file(r_name)
        combined_mols = list(finished_mols) + list(failed_mols)
        metric_records_by_pocket[example_idx] = [
            {
                'sample_index': mol_info.get('_sample_index', sample_index),
                'mol': mol_info.get('rdmol'),
            }
            for sample_index, mol_info in enumerate(combined_mols)
        ]
        pocket_name = (os.path.basename(os.path.dirname(r_name))
                       if os.path.basename(r_name) == record_filename
                       else os.path.basename(r_name))
        case_settings = _case_study_config(config)
        if case_settings is not None:
            ligand_ref_path = os.path.abspath(os.path.expanduser(
                str(case_settings['reference_ligand_path'])
            ))
            ligand_filename = str(case_settings.get(
                'reference_label', os.path.basename(ligand_ref_path)
            ))
            pocket_name = str(case_settings.get('id', pocket_name))
            pdbid = None
        elif config.dataset == 'pdbbind':
            pdbid = r_name.split('/')[-1].split('.')[0].split('_')[1]
            ligand_ref_path = os.path.join(config.protein_root, pdbid, pdbid + '_ligand.sdf')
        elif config.dataset == 'crossdocked':
            pdbid = None
            if ligand_filename is None:  # legacy samples_<encoded pocket>.pt format
                ligand_filename = r_name.split('/')[-1].split('.')[0].split('samples_')[1].replace('-', '/')
                ligand_ref_path = os.path.join(config.protein_root, ligand_filename[:-9] + '.sdf')
            else:
                ligand_ref_path = _reference_ligand_path(config, ligand_filename)
        else:
            raise ValueError('Unsupported dataset: %s' % config.dataset)
        n_success += len(finished_mols)
        num_samples += len(finished_mols) + len(failed_mols)
        for failed_idx, mol_info in enumerate(failed_mols):
            smiles = mol_info.get('smiles')
            if smiles is not None:
                assert '.' in smiles
                n_disconnect += 1
                status = 'disconnected'
            else:
                n_invalid += 1
                status = 'reconstruction_failed'
            element = mol_info.get('element', [])
            failed_row = {
                'pocket_index': example_idx,
                'pocket_name': pocket_name,
                'protein_filename': mol_info.get('_protein_filename'),
                'reference_ligand': ligand_filename,
                'sample_index': mol_info.get('_sample_index', failed_idx),
                'status': status,
                'reconstruction_success': int(status == 'disconnected'),
                'evaluation_success': 0,
                'failure_reason': mol_info.get('reconstruction_error'),
                'smiles': smiles,
                'num_atoms': len(element),
                'num_pharmacophores': mol_info.get('_num_pharmacophores'),
            }
            failed_row.update(_bond_decode_metrics(mol_info))
            molecule_rows.append(failed_row)

        ligand_ref_rdmol = Chem.SDMolSupplier(ligand_ref_path)[0]
        ligand_ref_list.append(ligand_ref_rdmol)
        if ligand_ref_rdmol is not None:
            try:
                reference_metrics_by_pocket[example_idx] = scoring_func.get_chem(ligand_ref_rdmol)
            except Exception as exc:
                logger.warning('Reference ligand chemistry failed for pocket %d: %s',
                               example_idx, exc)

            evaluate_reference_docking = bool(
                getattr(config, 'evaluate_reference_docking', False))
            if evaluate_reference_docking and config.docking_mode != 'none':
                reference_affinity_mode = (
                    'qvina' if config.docking_mode == 'qvina'
                    else 'dock' if config.docking_mode == 'vina_dock'
                    else 'minimize'
                )
                cache_key = _reference_docking_cache_key(
                    config, ligand_filename, reference_affinity_mode
                )
                cached_score = reference_docking_cache.get(cache_key)
                cached_score_is_valid = (
                    cached_score is not None
                    and np.isfinite(float(cached_score))
                    and float(cached_score) < 0
                )
                if cached_score_is_valid:
                    reference_docking_by_pocket[example_idx] = float(cached_score)
                else:
                    try:
                        if config.docking_mode == 'qvina':
                            if config.dataset == 'pdbbind':
                                reference_task = QVinaDockingTask.from_generated_mol_pdbbind(
                                    ligand_ref_rdmol, pdbid,
                                    protein_root=config.protein_root)
                            else:
                                reference_task = QVinaDockingTask.from_generated_mol_crossdocked(
                                    ligand_ref_rdmol, ligand_filename,
                                    protein_root=config.protein_root)
                            reference_result = reference_task.run_sync(
                                exhaustiveness=config.exhaustiveness
                            )
                        else:
                            reference_task = _vina_task(
                                config, ligand_ref_rdmol, ligand_filename, pdbid
                            )
                            reference_result = reference_task.run(
                                mode=reference_affinity_mode,
                                exhaustiveness=config.exhaustiveness,
                                seed=VINA_DOCKING_SEED)
                        if reference_result:
                            reference_score = float(reference_result[0]['affinity'])
                            if not np.isfinite(reference_score) or reference_score >= 0:
                                raise RuntimeError(
                                    'Invalid reference docking affinity: %.3f'
                                    % reference_score
                                )
                            reference_docking_by_pocket[example_idx] = reference_score
                            reference_docking_cache[cache_key] = reference_score
                            torch.save(reference_docking_cache,
                                       reference_docking_cache_path)
                    except Exception as exc:
                        logger.warning(
                            'Reference ligand docking failed for pocket %d: %s',
                            example_idx, exc)

        for sample_idx, mol_info in enumerate(finished_mols):
            pred_atom_type = np.asarray(mol_info['element'])
            pred_pos = np.asarray(mol_info['atom_pos'])

            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            molecule_row = {
                'pocket_index': example_idx,
                'pocket_name': pocket_name,
                'protein_filename': mol_info.get('_protein_filename'),
                'reference_ligand': ligand_filename,
                'sample_index': mol_info.get('_sample_index', sample_idx),
                'status': 'reconstructed',
                'reconstruction_success': 1,
                'evaluation_success': 0,
                'failure_reason': None,
                'smiles': mol_info.get('smiles'),
                'num_atoms': int(r_stable[2]),
                'num_pharmacophores': mol_info.get('_num_pharmacophores'),
                'mol_stable': int(r_stable[0]),
                'stable_atoms': int(r_stable[1]),
                'stable_atom_fraction': (
                    float(r_stable[1]) / r_stable[2] if r_stable[2] else None
                ),
                'distance_inferred_mol_stable': int(r_stable[0]),
                'distance_inferred_stable_atoms': int(r_stable[1]),
                'distance_inferred_atom_stable_fraction': (
                    float(r_stable[1]) / r_stable[2] if r_stable[2] else None
                ),
            }
            molecule_row.update(_bond_decode_metrics(mol_info))

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            mol = mol_info['rdmol']
            smiles = mol_info['smiles']

            assignment_metrics = _assignment_summary(
                mol_info,
                threshold=float(getattr(
                    config, 'phore_assignment_probability_threshold', 0.5
                )),
            )
            molecule_row.update(assignment_metrics)
            realization_metrics = None
            if (mol_info.get('_generated_phore_type') is not None
                    and mol_info.get('_generated_phore_pos') is not None):
                try:
                    realization_metrics = evaluate_pharmacophore_realization(
                        mol,
                        mol_info.get('_generated_phore_type'),
                        mol_info.get('_generated_phore_pos'),
                        mol_info.get('_generated_phore_vec'),
                        distance_threshold=float(getattr(
                            config, 'phore_match_distance_threshold', 1.5
                        )),
                        direction_cosine_threshold=float(getattr(
                            config, 'phore_direction_cosine_threshold', 0.7
                        )),
                    )
                    molecule_row.update({
                        key: realization_metrics.get(key)
                        for key in (
                            'generated_phore_count',
                            'molecule_extracted_phore_count',
                            'matched_phore_count',
                            'phore_realization_ratio',
                            'all_phores_realized',
                            'any_phore_realized',
                            'mean_phore_match_distance',
                            'mean_phore_direction_similarity',
                            'phore_realization_by_type',
                        )
                    })
                    molecule_row['phore_matches'] = json.dumps(
                        realization_metrics.get('phore_matches', []),
                        ensure_ascii=False, sort_keys=True,
                    )
                except Exception as exc:
                    logger.warning(
                        'Pharmacophore realization evaluation failed for %s_%s: %s',
                        example_idx, sample_idx, exc,
                    )

            try:
                chem_results = scoring_func.get_chem(mol)
                if config.docking_mode == 'qvina':
                    if config.dataset == 'pdbbind':
                        vina_task = QVinaDockingTask.from_generated_mol_pdbbind(
                            mol, pdbid, protein_root=config.protein_root)
                    elif config.dataset == 'crossdocked':
                        vina_task = QVinaDockingTask.from_generated_mol_crossdocked(
                            mol, ligand_filename, protein_root=config.protein_root)
                    vina_results = vina_task.run_sync(
                        exhaustiveness=config.exhaustiveness
                    )
                elif config.docking_mode in ['vina_score', 'vina_dock']:
                    vina_task = _vina_task(
                        config, mol, ligand_filename, pdbid
                    )
                    score_only_results = vina_task.run(
                        mode='score_only',
                        exhaustiveness=config.exhaustiveness,
                        seed=VINA_DOCKING_SEED,
                    )
                    minimize_results = vina_task.run(
                        mode='minimize',
                        exhaustiveness=config.exhaustiveness,
                        seed=VINA_DOCKING_SEED,
                    )
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if config.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(
                            mode='dock',
                            exhaustiveness=config.exhaustiveness,
                            seed=VINA_DOCKING_SEED,
                        )
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except Exception as exc:
                failure_message = '%s: %s' % (type(exc).__name__, str(exc))
                evaluation_failures[failure_message] += 1
                if config.verbose:
                    logger.warning('Evaluation failed for %s_%s: %s',
                                   example_idx, sample_idx, failure_message, exc_info=True)
                molecule_row['status'] = 'evaluation_failed'
                molecule_row['failure_reason'] = failure_message
                molecule_rows.append(molecule_row)
                continue

            molecule_row.update({
                'status': 'evaluated',
                'evaluation_success': 1,
                'qed': chem_results['qed'],
                'sa': chem_results['sa'],
                'logp': chem_results['logp'],
                'lipinski': chem_results['lipinski'],
                'tpsa': chem_results['tpsa'],
                'ring_size_3': chem_results['ring_size'].get(3, 0),
                'ring_size_4': chem_results['ring_size'].get(4, 0),
                'ring_size_5': chem_results['ring_size'].get(5, 0),
                'ring_size_6': chem_results['ring_size'].get(6, 0),
                'ring_size_7': chem_results['ring_size'].get(7, 0),
                'ring_size_8': chem_results['ring_size'].get(8, 0),
                'ring_size_9': chem_results['ring_size'].get(9, 0),
                'vina_score': _first_affinity(vina_results, 'score_only'),
                'vina_minimize': _first_affinity(vina_results, 'minimize'),
                'vina_dock': _first_affinity(vina_results, 'dock'),
                'qvina': (
                    float(vina_results[0]['affinity'])
                    if (config.docking_mode == 'qvina' and vina_results
                        and np.isfinite(float(vina_results[0]['affinity']))
                        and float(vina_results[0]['affinity']) < 0)
                    else None
                ),
            })
            molecule_rows.append(molecule_row)

            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'pocket_index': example_idx,
                'pocket_name': pocket_name,
                'sample_index': mol_info.get('_sample_index', sample_idx),
                'protein_filename': mol_info.get('_protein_filename'),
                'ligand_filename': ligand_filename,
                'mol': mol,
                'smiles': smiles,
                'pred_pos': pred_pos,
                'chem_results': chem_results,
                'vina': vina_results,
                'phore_realization': realization_metrics,
            })
    logger.info(f'Evaluate done! {n_success} samples in total.')
    logger.info('Number of evaluated mols: %d' % (len(results)))
    if evaluation_failures:
        failure_summary = '; '.join(
            '%d x %s' % (count, message)
            for message, count in evaluation_failures.most_common(5)
        )
        logger.warning('Chemical/docking evaluation failures: %s', failure_summary)

    reconstructed_total = n_success + n_disconnect
    fraction_mol_stable = all_mol_stable / n_success if n_success else 0.0
    fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom else 0.0
    fraction_recon = reconstructed_total / num_samples if num_samples else 0.0
    fraction_eval = n_eval_success / n_success if n_success else 0.0
    stability_dict = {
        'distance_inferred_mol_stable': fraction_mol_stable,
        'distance_inferred_atom_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
    }
    log_metric_dict(stability_dict, logger)

    generation_metrics = evaluate_generated_records(
        metric_records_by_pocket, reference_scaffolds)
    validity = generation_metrics['global_metrics']['validity']
    connectivity = n_success / reconstructed_total if reconstructed_total else 0.0
    validity_dict = {'validity': validity, 'connectivity': connectivity}
    log_metric_dict(validity_dict, logger)

    if not results:
        logger.error(
            'No reconstructed molecule passed chemical/docking evaluation; '
            'molecule-quality metrics cannot be computed. See the failure summary above.'
        )
        sys.exit(1)

    ligand_gen_list = [r['mol'] for r in results]
    sim_with_ref_values = _max_similarity_with_reference(
        ligand_gen_list, reference_fingerprints
    )
    sim_with_ref = _stats(sim_with_ref_values)[0]
    global_generation_metrics = generation_metrics['global_metrics']
    sim_dict = {
        'novelty': global_generation_metrics['novelty'],
        'uniqueness': global_generation_metrics['uniqueness'],
        'diversity': global_generation_metrics['diversity'],
        'internal_diversity_2d': global_generation_metrics['internal_diversity_2d'],
        'internal_diversity_3d': global_generation_metrics['internal_diversity_3d'],
        'sim_with_ref': sim_with_ref,
    }
    log_metric_dict(sim_dict, logger)

    quality_by_sample = {
        key: {
            'validity': values['validity'],
            'scaffold_smiles': values['scaffold_smiles'],
            'scaffold_novel': values['scaffold_novel'],
            'novelty': values['scaffold_novel'],
        }
        for key, values in generation_metrics['molecule_metrics'].items()
    }
    for result, similarity in zip(results, sim_with_ref_values):
        quality_by_sample.setdefault(
            (result['pocket_index'], result['sample_index']), {}
        )['sim_with_ref'] = similarity
    for molecule_row in molecule_rows:
        quality = quality_by_sample.get((
            molecule_row['pocket_index'], molecule_row['sample_index']
        ))
        if quality is not None:
            molecule_row.update(quality)


    gen_predicted_rmsd = []
    gen_energy_diff = []
    gen_optimized_rmsd = []
    ref_predicted_rmsd = []
    ref_energy_diff = []
    ref_optimized_rmsd = []

    if config.eval_mode == 'bond_only':
        c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
        c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
        logger.info('JS bond distances of complete mols: ')
        log_metric_dict(c_bond_length_dict, logger)

    elif config.eval_mode == 'all':
        bond_length_dict = eval_all.calculate_bond_jsd(
            ligand_gen_list, ligand_ref_list
        )
        angle_dict = eval_all.calculate_angle_jsd(
            ligand_gen_list, ligand_ref_list
        )
        dihedral_dict = eval_all.calculate_dihedral_jsd(
            ligand_gen_list, ligand_ref_list
        )
        logger.info('JS bond distances of complete mols: ')
        log_metric_dict(bond_length_dict, logger)
        logger.info('JS bond angles of complete mols: ')
        log_metric_dict(angle_dict, logger)
        logger.info('JS dihedrals of complete mols: ')
        log_metric_dict(dihedral_dict, logger)

        gen_predicted_rmsd = eval_all.calculate_predicted_rmsd(ligand_gen_list)
        gen_energy_diff, gen_optimized_rmsd = (
            eval_all.calculate_optimized_rmsd(ligand_gen_list)
        )
        valid_reference_mols = [mol for mol in ligand_ref_list if mol is not None]
        ref_predicted_rmsd = eval_all.calculate_predicted_rmsd(
            valid_reference_mols
        )
        ref_energy_diff, ref_optimized_rmsd = eval_all.calculate_optimized_rmsd(
            valid_reference_mols
        )
        gen_energy_diff = [diff if diff is not None else 0 for diff in gen_energy_diff]
        ref_energy_diff = [diff if diff is not None else 0 for diff in ref_energy_diff]
        if config.save:
            if gen_predicted_rmsd:
                eval_all.plot_rmsd_violin(
                    gen_predicted_rmsd,
                    save_path=os.path.join(result_path, 'predicted_rmsd_violin.png'))
            if gen_optimized_rmsd:
                eval_all.plot_rmsd_violin(
                    gen_optimized_rmsd,
                    save_path=os.path.join(result_path, 'optimized_rmsd_violin.png'))

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    log_metric_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if config.save:
        eval_bond_length.plot_distance_hist(
            success_pair_length_profile,
            metrics=success_js_metrics,
            save_path=os.path.join(result_path, 'pair_dist_hist.png'))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logp = [r['chem_results']['logp'] for r in results]
    lipinski = [r['chem_results']['lipinski'] for r in results]
    tpsa = [r['chem_results']['tpsa'] for r in results]
    vina_score_only, vina_min, vina_dock, qvina = [], [], [], []
    if config.docking_mode == 'qvina':
        qvina = [r['vina'][0]['affinity'] for r in results if r.get('vina')]
    elif config.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [
            r['vina']['score_only'][0]['affinity'] for r in results
            if r.get('vina') and r['vina'].get('score_only')
        ]
        vina_min = [
            r['vina']['minimize'][0]['affinity'] for r in results
            if r.get('vina') and r['vina'].get('minimize')
        ]
        if config.docking_mode == 'vina_dock':
            vina_dock = [
                r['vina']['dock'][0]['affinity'] for r in results
                if r.get('vina') and r['vina'].get('dock')
            ]

    ring_ratios = log_ring_distribution(
        [r['chem_results']['ring_size'] for r in results], logger)

    rows_by_pocket = {}
    for row in molecule_rows:
        rows_by_pocket.setdefault(row['pocket_index'], []).append(row)

    filter_config = getattr(config, 'topk_filter', {}) or {}
    topk_filter_enabled = bool(filter_config.get('enabled', True))
    min_sa = float(filter_config.get('min_sa', DEFAULT_MIN_SA))
    min_qed = float(filter_config.get('min_qed', DEFAULT_MIN_QED))
    max_logp = float(filter_config.get('max_logp', DEFAULT_MAX_LOGP))
    top_k = int(getattr(config, 'top_k', 10))

    pocket_metric_rows = []
    topk_rows = []
    for pocket_index in sorted(rows_by_pocket):
        pocket_rows = rows_by_pocket[pocket_index]
        evaluated_rows = [row for row in pocket_rows if row.get('evaluation_success')]
        reconstructed_rows = [row for row in pocket_rows if row.get('reconstruction_success')]
        connected_rows = [
            row for row in reconstructed_rows if row.get('status') != 'disconnected'
        ]

        total_count = len(pocket_rows)
        reconstructed_count = len(reconstructed_rows)
        connected_count = len(connected_rows)
        evaluated_count = len(evaluated_rows)
        pocket_generation_metrics = generation_metrics['pocket_metrics'][pocket_index]
        pocket_novelty = pocket_generation_metrics['novelty']
        pocket_uniqueness = pocket_generation_metrics['uniqueness']
        pocket_diversity = pocket_generation_metrics['diversity']
        pocket_internal_diversity_2d = pocket_generation_metrics['internal_diversity_2d']
        pocket_internal_diversity_3d = pocket_generation_metrics['internal_diversity_3d']
        pocket_sim_with_ref = _stats([
            row.get('sim_with_ref') for row in evaluated_rows
        ])[0]

        reference_metrics = reference_metrics_by_pocket.get(pocket_index, {})
        reference_docking_score = reference_docking_by_pocket.get(pocket_index)
        for row in pocket_rows:
            row['pocket_novelty'] = pocket_novelty
            row['pocket_uniqueness'] = pocket_uniqueness
            row['pocket_diversity'] = pocket_diversity
            row['pocket_internal_diversity_2d'] = pocket_internal_diversity_2d
            row['pocket_internal_diversity_3d'] = pocket_internal_diversity_3d
            row['reference_qed'] = reference_metrics.get('qed')
            row['reference_sa'] = reference_metrics.get('sa')
            row['reference_logp'] = reference_metrics.get('logp')
            row['reference_lipinski'] = reference_metrics.get('lipinski')
            row['reference_tpsa'] = reference_metrics.get('tpsa')
            row['reference_vina_dock'] = reference_docking_score
            docking_score = _docking_score(row)
            row['better_than_reference'] = (
                int(docking_score <= reference_docking_score)
                if docking_score is not None and reference_docking_score is not None
                else None
            )
            row['passes_topk_filter'] = 0
            row['topk_rank'] = None

        filtered_rows = []
        for row in evaluated_rows:
            docking_score = _docking_score(row)
            sa_value = _finite_values([row.get('sa')])
            qed_value = _finite_values([row.get('qed')])
            logp_value = _finite_values([row.get('logp')])
            if (docking_score is not None and (
                    not topk_filter_enabled or (
                        docking_score < 0 and
                        sa_value and sa_value[0] >= min_sa and
                        qed_value and qed_value[0] >= min_qed and
                        logp_value and logp_value[0] <= max_logp
                    ))):
                row['passes_topk_filter'] = 1
                filtered_rows.append(row)
        filtered_rows.sort(key=_docking_score)
        selected_rows = filtered_rows[:top_k]
        for rank, row in enumerate(selected_rows, 1):
            row['topk_rank'] = rank
            topk_rows.append(row)

        stable_atoms = sum(row.get('stable_atoms', 0) or 0 for row in connected_rows)
        total_atoms = sum(row.get('num_atoms', 0) or 0 for row in connected_rows)
        graph_rows = [
            row for row in reconstructed_rows
            if row.get('generated_graph_valence_valid') is not None
        ]
        graph_atoms = sum(row.get('num_atoms', 0) or 0 for row in graph_rows)
        graph_overvalent_atoms = sum(
            row.get('generated_graph_num_overvalent_atoms', 0) or 0
            for row in graph_rows
        )
        raw_docking_scores = _finite_values([_docking_score(row) for row in evaluated_rows])
        better_than_reference = [
            row.get('better_than_reference') for row in evaluated_rows
            if row.get('better_than_reference') is not None
        ]
        topk_docking_scores = _finite_values([_docking_score(row) for row in selected_rows])
        pocket_metric_rows.append({
            'pocket_index': pocket_index,
            'pocket_name': pocket_rows[0].get('pocket_name'),
            'protein_filename': pocket_rows[0].get('protein_filename'),
            'reference_ligand': pocket_rows[0].get('reference_ligand'),
            'num_samples': total_count,
            'num_sanitized_valid': pocket_generation_metrics['num_sanitized_valid'],
            'num_unique_structures': pocket_generation_metrics['num_unique_structures'],
            'num_scaffold_bearing': pocket_generation_metrics['num_scaffold_bearing'],
            'num_unique_scaffolds': pocket_generation_metrics['num_unique_scaffolds'],
            'num_novel_scaffolds': pocket_generation_metrics['num_novel_scaffolds'],
            'num_reconstructed': reconstructed_count,
            'num_connected': connected_count,
            'num_evaluated': evaluated_count,
            'recon_success': (float(reconstructed_count) / total_count
                              if total_count else None),
            'eval_success': (float(evaluated_count) / connected_count
                             if connected_count else None),
            'validity': pocket_generation_metrics['validity'],
            'connectivity': (float(connected_count) / reconstructed_count
                             if reconstructed_count else None),
            'mol_stable': (float(sum(row.get('mol_stable', 0) or 0
                                    for row in connected_rows)) /
                           connected_count if connected_count else None),
            'atom_stable': float(stable_atoms) / total_atoms if total_atoms else None,
            'distance_inferred_mol_stable': (
                float(sum(row.get('mol_stable', 0) or 0
                          for row in connected_rows)) /
                connected_count if connected_count else None
            ),
            'distance_inferred_atom_stable': (
                float(stable_atoms) / total_atoms if total_atoms else None
            ),
            'generated_graph_valence_valid_ratio': _stats([
                float(row.get('generated_graph_valence_valid'))
                for row in graph_rows
            ])[0],
            'generated_graph_overvalent_atom_fraction': (
                float(graph_overvalent_atoms) / graph_atoms
                if graph_atoms else None
            ),
            'generated_graph_mean_components': _stats([
                row.get('generated_graph_num_components') for row in reconstructed_rows
            ])[0],
            'generated_graph_mean_selected_bonds': _stats([
                row.get('generated_graph_selected_bonds') for row in reconstructed_rows
            ])[0],
            'generated_graph_mean_positive_candidate_bonds': _stats([
                row.get('generated_graph_positive_candidate_bonds')
                for row in reconstructed_rows
            ])[0],
            'novelty': pocket_novelty,
            'uniqueness': pocket_uniqueness,
            'diversity': pocket_diversity,
            'internal_diversity_2d': pocket_internal_diversity_2d,
            'internal_diversity_3d': pocket_internal_diversity_3d,
            'sim_with_ref': pocket_sim_with_ref,
            'qed': _stats([row.get('qed') for row in evaluated_rows])[0],
            'sa': _stats([row.get('sa') for row in evaluated_rows])[0],
            'logp': _stats([row.get('logp') for row in evaluated_rows])[0],
            'lipinski': _stats([row.get('lipinski') for row in evaluated_rows])[0],
            'tpsa': _stats([row.get('tpsa') for row in evaluated_rows])[0],
            'phore_realization_ratio': _stats([
                row.get('phore_realization_ratio') for row in evaluated_rows
            ])[0],
            'all_phores_realized': _stats([
                row.get('all_phores_realized') for row in evaluated_rows
            ])[0],
            'any_phore_realized': _stats([
                row.get('any_phore_realized') for row in evaluated_rows
            ])[0],
            'mean_phore_match_distance': _stats([
                row.get('mean_phore_match_distance') for row in evaluated_rows
            ])[0],
            'mean_phore_direction_similarity': _stats([
                row.get('mean_phore_direction_similarity') for row in evaluated_rows
            ])[0],
            'mean_max_assignment_probability': _stats([
                row.get('mean_max_assignment_probability') for row in evaluated_rows
            ])[0],
            'assignment_coverage_ratio': _stats([
                row.get('assignment_coverage_ratio') for row in evaluated_rows
            ])[0],
            'vina_score': _stats([row.get('vina_score') for row in evaluated_rows])[0],
            'vina_minimize': _stats([row.get('vina_minimize') for row in evaluated_rows])[0],
            'vina_dock': _stats([row.get('vina_dock') for row in evaluated_rows])[0],
            'qvina': _stats([row.get('qvina') for row in evaluated_rows])[0],
            'affinity_raw_mean': _stats(raw_docking_scores)[0],
            'affinity_reference': reference_docking_score,
            'high_affinity_ratio': _stats(better_than_reference)[0],
            'num_topk_filtered': len(selected_rows),
            'affinity_filtered_top10': _stats(topk_docking_scores)[0],
            'reference_qed': reference_metrics.get('qed'),
            'reference_sa': reference_metrics.get('sa'),
            'reference_logp': reference_metrics.get('logp'),
            'reference_lipinski': reference_metrics.get('lipinski'),
            'reference_tpsa': reference_metrics.get('tpsa'),
        })


    pocket_value = lambda key: [row.get(key) for row in pocket_metric_rows]
    summary_csv_rows = [
        _summary_row('num_pockets', [len(pocket_metric_rows)]),
        _summary_row('num_samples', [num_samples]),
        _summary_row('num_reconstructed', [reconstructed_total]),
        _summary_row('num_connected', [n_success]),
        _summary_row('num_evaluated', [len(results)]),
        _summary_row('distance_inferred_mol_stable',
                     pocket_value('distance_inferred_mol_stable')),
        _summary_row('distance_inferred_atom_stable',
                     pocket_value('distance_inferred_atom_stable')),
        _summary_row('generated_graph_valence_valid_ratio',
                     pocket_value('generated_graph_valence_valid_ratio')),
        _summary_row('generated_graph_overvalent_atom_fraction',
                     pocket_value('generated_graph_overvalent_atom_fraction')),
        _summary_row('generated_graph_mean_components',
                     pocket_value('generated_graph_mean_components')),
        _summary_row('generated_graph_mean_selected_bonds',
                     pocket_value('generated_graph_mean_selected_bonds')),
        _summary_row('generated_graph_mean_positive_candidate_bonds',
                     pocket_value('generated_graph_mean_positive_candidate_bonds')),
        _summary_row('recon_success', pocket_value('recon_success')),
        _summary_row('eval_success', pocket_value('eval_success')),
        _summary_row('validity', pocket_value('validity')),
        _summary_row('connectivity', pocket_value('connectivity')),
        _summary_row('novelty', pocket_value('novelty')),
        _summary_row('uniqueness', pocket_value('uniqueness')),
        _summary_row('diversity', pocket_value('diversity')),
        _summary_row('internal_diversity_2d',
                     pocket_value('internal_diversity_2d')),
        _summary_row('internal_diversity_3d',
                     pocket_value('internal_diversity_3d')),
        _summary_row('sim_with_ref', sim_with_ref_values),
        _summary_row('qed', pocket_value('qed')),
        _summary_row('sa', pocket_value('sa')),
        _summary_row('logp', pocket_value('logp')),
        _summary_row('lipinski', pocket_value('lipinski')),
        _summary_row('tpsa', pocket_value('tpsa')),
        _summary_row('phore_realization_ratio',
                     pocket_value('phore_realization_ratio')),
        _summary_row('all_phores_realized',
                     pocket_value('all_phores_realized')),
        _summary_row('any_phore_realized',
                     pocket_value('any_phore_realized')),
        _summary_row('mean_phore_match_distance',
                     pocket_value('mean_phore_match_distance')),
        _summary_row('mean_phore_direction_similarity',
                     pocket_value('mean_phore_direction_similarity')),
        _summary_row('mean_max_assignment_probability',
                     pocket_value('mean_max_assignment_probability')),
        _summary_row('assignment_coverage_ratio',
                     pocket_value('assignment_coverage_ratio')),
        _summary_row('ref_qed', pocket_value('reference_qed')),
        _summary_row('ref_sa', pocket_value('reference_sa')),
        _summary_row('ref_logp', pocket_value('reference_logp')),
        _summary_row('ref_lipinski', pocket_value('reference_lipinski')),
        _summary_row('ref_tpsa', pocket_value('reference_tpsa')),
        _summary_row('vina_score', pocket_value('vina_score')),
        _summary_row('vina_minimize', pocket_value('vina_minimize')),
        _summary_row('vina_dock', pocket_value('vina_dock')),
        _summary_row('qvina', pocket_value('qvina')),
        _summary_row('affinity_raw_mean', pocket_value('affinity_raw_mean')),
        _summary_row('affinity_reference', pocket_value('affinity_reference')),
        _summary_row('high_affinity_ratio', pocket_value('high_affinity_ratio')),
        _summary_row('affinity_filtered_top10',
                     pocket_value('affinity_filtered_top10')),
        _summary_row('predicted_rmsd', gen_predicted_rmsd),
        _summary_row('optimized_rmsd', gen_optimized_rmsd),
        _summary_row('optimization_energy_difference', gen_energy_diff),
        _summary_row('reference_predicted_rmsd', ref_predicted_rmsd),
        _summary_row('reference_optimized_rmsd', ref_optimized_rmsd),
        _summary_row('reference_optimization_energy_difference', ref_energy_diff),
        _summary_row('atom_type_js', [atom_type_js]),
    ]
    ring_counts = {
        ring_size: sum(
            result['chem_results']['ring_size'].get(ring_size, 0)
            for result in results
        )
        for ring_size in sorted(ring_ratios)
    }
    ring_csv_rows = [
        {
            'ring_size': ring_size,
            'count': ring_counts[ring_size],
            'ratio': ring_ratios[ring_size],
        }
        for ring_size in sorted(ring_ratios)
    ]

    print('\n' + '=' * 64)
    print('Final Evaluation Report')
    print('=' * 64)
    print('[0. Counts]')
    print('  - Pockets                       : %d' % len(pocket_metric_rows))
    print('  - Generated records             : %d' % num_samples)
    print('  - Chemically reconstructed      : %d' % reconstructed_total)
    print('  - Connected molecules           : %d' % n_success)
    print('  - Fully evaluated molecules     : %d' % len(results))

    print('\n[1. Reconstruction, Validity & Stability]')
    print(_format_report_metric('Reconstruction success', pocket_value('recon_success')))
    print(_format_report_metric('Evaluation success', pocket_value('eval_success')))
    print(_format_report_metric('Validity', pocket_value('validity')))
    print(_format_report_metric('Connectivity', pocket_value('connectivity')))
    print(_format_report_metric(
        'Distance-inferred molecule stability',
        pocket_value('distance_inferred_mol_stable'),
    ))
    print(_format_report_metric(
        'Distance-inferred atom stability',
        pocket_value('distance_inferred_atom_stable'),
    ))
    print(_format_report_metric(
        'Generated-graph valence-valid ratio',
        pocket_value('generated_graph_valence_valid_ratio'),
    ))
    print(_format_report_metric(
        'Generated-graph overvalent atoms',
        pocket_value('generated_graph_overvalent_atom_fraction'),
    ))
    print(_format_report_metric(
        'Generated-graph mean components',
        pocket_value('generated_graph_mean_components'),
    ))
    print(_format_report_metric(
        'Generated-graph selected bonds',
        pocket_value('generated_graph_mean_selected_bonds'),
    ))
    print(_format_report_metric(
        'Generated-graph positive candidates',
        pocket_value('generated_graph_mean_positive_candidate_bonds'),
    ))

    print('\n[2. Chemical Properties & Distributions]')
    print(_format_report_metric('QED', pocket_value('qed')))
    print(_format_report_metric('Reference QED', pocket_value('reference_qed')))
    print(_format_report_metric('SA', pocket_value('sa')))
    print(_format_report_metric('Reference SA', pocket_value('reference_sa')))
    print(_format_report_metric('LogP', pocket_value('logp')))
    print(_format_report_metric('Reference LogP', pocket_value('reference_logp')))
    print(_format_report_metric('Lipinski', pocket_value('lipinski')))
    print(_format_report_metric('Reference Lipinski',
                                pocket_value('reference_lipinski')))
    print(_format_report_metric('TPSA', pocket_value('tpsa')))
    print(_format_report_metric('Reference TPSA',
                                pocket_value('reference_tpsa')))

    print('\n[3. Novelty & Diversity]')
    print(_format_report_metric('Scaffold novelty', pocket_value('novelty')))
    print(_format_report_metric('Uniqueness', pocket_value('uniqueness')))
    print(_format_report_metric('Scaffold diversity', pocket_value('diversity')))
    print(_format_report_metric('2D internal diversity',
                                pocket_value('internal_diversity_2d')))
    print(_format_report_metric('3D internal diversity',
                                pocket_value('internal_diversity_3d')))
    print(_format_report_metric('Global scaffold diversity',
                                [sim_dict.get('diversity')]))
    print(_format_report_metric('Similarity with reference',
                                sim_with_ref_values))

    print('\n[4. Pharmacophore Realization]')
    print(_format_report_metric('Realization ratio',
                                pocket_value('phore_realization_ratio')))
    print(_format_report_metric('All pharmacophores realized',
                                pocket_value('all_phores_realized'),
                                scale=100.0, suffix='%'))
    print(_format_report_metric('Any pharmacophore realized',
                                pocket_value('any_phore_realized'),
                                scale=100.0, suffix='%'))
    print(_format_report_metric('Mean match distance',
                                pocket_value('mean_phore_match_distance'),
                                suffix=' Å'))
    print(_format_report_metric('Direction similarity',
                                pocket_value('mean_phore_direction_similarity')))
    print(_format_report_metric('Assignment coverage',
                                pocket_value('assignment_coverage_ratio'),
                                scale=100.0, suffix='%'))

    if config.docking_mode == 'vina_dock':
        dock_or_qvina_values = pocket_value('vina_dock')
        ranking_affinity_label = 'Vina dock'
    elif config.docking_mode == 'qvina':
        dock_or_qvina_values = pocket_value('qvina')
        ranking_affinity_label = 'QVina dock'
    elif config.docking_mode == 'vina_score':
        dock_or_qvina_values = []
        ranking_affinity_label = 'Vina minimize'
    else:
        dock_or_qvina_values = []
        ranking_affinity_label = 'N/A'

    topk_title = (
        'Filtered Top-%d' % top_k if topk_filter_enabled
        else 'Unfiltered Top-%d' % top_k
    )
    print('\n[5. Binding Affinity & %s]' % topk_title)
    print(_format_report_metric('Vina score-only', pocket_value('vina_score')))
    print(_format_report_metric('Vina minimize', pocket_value('vina_minimize')))
    print(_format_report_metric('Vina dock / QVina', dock_or_qvina_values))
    print(_format_report_metric('Affinity used for ranking (%s)' % ranking_affinity_label,
                                pocket_value('affinity_raw_mean')))
    print(_format_report_metric('Reference affinity (%s)' % ranking_affinity_label,
                                pocket_value('affinity_reference')))
    print(_format_report_metric('High-affinity ratio',
                                pocket_value('high_affinity_ratio'),
                                scale=100.0, suffix='%'))
    print(_format_report_metric('%s affinity' % topk_title,
                                pocket_value('affinity_filtered_top10')))
    if topk_filter_enabled:
        print('  - Filter                        : SA >= %.2f, QED >= %.2f, LogP <= %.2f, affinity < 0'
              % (min_sa, min_qed, max_logp))
    else:
        print('  - Filter                        : disabled; rank all evaluated molecules by affinity')

    if config.eval_mode == 'all':
        print('\n[6. 3D Conformation Diagnostics]')
        print(_format_report_metric(
            'Generated -> RDKit UFF min RMSD', gen_predicted_rmsd))
        print(_format_report_metric(
            'Reference -> RDKit UFF min RMSD', ref_predicted_rmsd))
        print(_format_report_metric(
            'Generated MMFF relaxation displacement', gen_optimized_rmsd))
        print(_format_report_metric(
            'Reference MMFF relaxation displacement', ref_optimized_rmsd))
        print(_format_report_metric(
            'Generated MMFF energy decrease', gen_energy_diff))
        print(_format_report_metric(
            'Reference MMFF energy decrease', ref_energy_diff))
        print('  - Note                          : These are free-conformer/relaxation diagnostics, not RMSD to the pocket-bound reference pose.')
    print('=' * 64)

    if config.save:
        molecules_path = os.path.join(result_path, 'molecules.csv')
        molecule_details_path = os.path.join(result_path, 'molecule_details.csv')
        pocket_metrics_path = os.path.join(result_path, 'pocket_metrics.csv')
        pocket_details_path = os.path.join(result_path, 'pocket_details.csv')
        summary_path = os.path.join(result_path, 'summary_metrics.csv')
        ring_statistics_path = os.path.join(result_path, 'ring_statistics.csv')
        topk_path = os.path.join(result_path, 'molecules_top10.csv')
        _write_molecule_csv(molecules_path, molecule_rows, logger)
        _write_molecule_csv(molecule_details_path, molecule_rows, logger,
                            details=True)
        _write_molecule_csv(topk_path, topk_rows, logger, topk=True)
        numbered_pocket_rows = _numbered_pocket_rows(pocket_metric_rows)
        _write_dict_rows(pocket_metrics_path, numbered_pocket_rows,
                         _POCKET_MAIN_CSV_FIELDS, logger,
                         'compact per-pocket CSV')
        _write_dict_rows(pocket_details_path, numbered_pocket_rows,
                         _POCKET_DETAIL_CSV_FIELDS, logger,
                         'per-pocket detail CSV')
        _write_dict_rows(summary_path, summary_csv_rows,
                         ['Metric', 'Mean', 'Std'], logger, 'summary CSV')
        _write_dict_rows(ring_statistics_path, ring_csv_rows,
                         ['ring_size', 'count', 'ratio'], logger,
                         'ring statistics CSV')
        torch.save({
            'stability': stability_dict,
            'validity': validity_dict,
            'similarity': sim_dict,
            'generation_metrics': generation_metrics,
            'scaffold_reference': {
                'scope': 'CrossDocked train union test known ligands',
                'counts': reference_scaffold_counts,
                'cache': reference_scaffold_cache,
            },
            'bond_length': all_bond_dist,
            'pair_distance': success_pair_dist,
            'pocket_metrics': pocket_metric_rows,
            'summary_metrics': summary_csv_rows,
            'all_results': results
        }, os.path.join(result_path, 'metrics.pt'))

        print('\nSaved files:')
        print('  - Main ligand metrics   : %s' % molecules_path)
        print('  - Ligand details/rings  : %s' % molecule_details_path)
        print('  - Per-pocket metrics    : %s' % pocket_metrics_path)
        print('  - Per-pocket details    : %s' % pocket_details_path)
        print('  - Summary mean/std      : %s' % summary_path)
        print('  - Ring statistics       : %s' % ring_statistics_path)
        print('  - %s Top-%d       : %s' % (
            'Filtered' if topk_filter_enabled else 'Unfiltered',
            top_k, topk_path,
        ))
