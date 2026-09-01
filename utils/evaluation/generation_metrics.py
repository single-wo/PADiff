import math
import os
import pickle

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


MORGAN_RADIUS = 2
MORGAN_BITS = 1024
MIN_SCAFFOLD_RINGS = 2
REFERENCE_CACHE_VERSION = 1


def sanitize_generated_molecule(mol):
    """Sanitize the full molecule, then return its largest fragment."""
    if mol is None:
        return None
    try:
        sanitized = Chem.Mol(mol)
        Chem.SanitizeMol(sanitized)
        fragments = Chem.GetMolFrags(
            sanitized, asMols=True, sanitizeFrags=True)
        if not fragments:
            return None
        sanitized = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
        Chem.SanitizeMol(sanitized)
        return sanitized
    except Exception:
        return None


def canonical_structure_smiles(mol):
    return Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True) if mol is not None else None


def scaffold_smiles(mol):
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
        if rdMolDescriptors.CalcNumRings(scaffold) < MIN_SCAFFOLD_RINGS:
            return None
        value = Chem.MolToSmiles(
            scaffold, canonical=True, isomericSmiles=False)
        return value or None
    except Exception:
        return None


def morgan_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def usrcat_descriptor(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        descriptor = np.asarray(
            rdMolDescriptors.GetUSRCAT(mol), dtype=np.float64)
        if descriptor.shape != (60,) or not np.isfinite(descriptor).all():
            return None
        return descriptor
    except Exception:
        return None


def internal_diversity_2d(fingerprints):
    """Return ``1 - mean pairwise Tanimoto`` for Morgan fingerprints."""
    fingerprints = list(fingerprints)
    if len(fingerprints) < 2:
        return None
    similarity_sum = 0.0
    pair_count = 0
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints[index + 1:])
        similarity_sum += float(sum(similarities))
        pair_count += len(similarities)
    return 1.0 - similarity_sum / pair_count if pair_count else None


def _usrcat_similarity(descriptor, others):
    """Vectorized equivalent of RDKit ``GetUSRScore`` for USRCAT vectors."""
    others = np.asarray(others, dtype=np.float64)
    return 1.0 / (
        1.0 + np.abs(others - descriptor).sum(axis=1) / 12.0)


def internal_diversity_3d(descriptors):
    """Return ``1 - mean pairwise USRCAT similarity``."""
    descriptors = np.asarray(list(descriptors), dtype=np.float64)
    if len(descriptors) < 2:
        return None
    similarity_sum = 0.0
    pair_count = 0
    for index, descriptor in enumerate(descriptors[:-1]):
        similarities = _usrcat_similarity(
            descriptor, descriptors[index + 1:])
        similarity_sum += float(similarities.sum())
        pair_count += int(similarities.size)
    return 1.0 - similarity_sum / pair_count if pair_count else None


def _safe_ratio(numerator, denominator):
    return float(numerator) / denominator if denominator else None


def _finite_stats(values):
    finite = np.asarray([
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ], dtype=np.float64)
    if not finite.size:
        return {'n': 0, 'mean': None, 'std': None}
    return {
        'n': int(finite.size),
        'mean': float(finite.mean()),
        'std': float(finite.std(ddof=0)),
    }


def _calculate_group(records, reference_scaffolds):
    total = len(records)
    valid_records = [record for record in records if record['validity']]
    structures = [record['canonical_smiles'] for record in valid_records]
    scaffolds = [
        record['scaffold_smiles'] for record in valid_records
        if record['scaffold_smiles'] is not None
    ]
    unique_scaffolds = set(scaffolds)
    novel_scaffolds = unique_scaffolds.difference(reference_scaffolds)
    fingerprints = [record['fingerprint'] for record in valid_records]
    descriptors = [
        record['usrcat'] for record in valid_records
        if record['usrcat'] is not None
    ]
    return {
        'num_samples': total,
        'num_sanitized_valid': len(valid_records),
        'num_unique_structures': len(set(structures)),
        'num_scaffold_bearing': len(scaffolds),
        'num_unique_scaffolds': len(unique_scaffolds),
        'num_novel_scaffolds': len(novel_scaffolds),
        'num_usrcat': len(descriptors),
        'validity': _safe_ratio(len(valid_records), total),
        'uniqueness': _safe_ratio(len(set(structures)), len(valid_records)),
        'diversity': _safe_ratio(len(unique_scaffolds), len(valid_records)),
        'novelty': _safe_ratio(len(novel_scaffolds), len(unique_scaffolds)),
        'internal_diversity_2d': internal_diversity_2d(fingerprints),
        'internal_diversity_3d': internal_diversity_3d(descriptors),
    }


def evaluate_generated_records(records_by_pocket, reference_scaffolds):
    """Evaluate generated molecules globally and per pocket."""
    reference_scaffolds = set(reference_scaffolds or ())
    prepared_by_pocket = {}
    molecule_metrics = {}
    all_records = []

    for pocket_id in sorted(records_by_pocket):
        prepared = []
        for record in records_by_pocket[pocket_id]:
            sample_index = int(record['sample_index'])
            mol = sanitize_generated_molecule(record.get('mol'))
            valid = int(mol is not None)
            canonical_smiles = canonical_structure_smiles(mol)
            generated_scaffold = scaffold_smiles(mol)
            fingerprint = morgan_fingerprint(mol) if mol is not None else None
            descriptor = usrcat_descriptor(mol)
            scaffold_novel = (
                int(generated_scaffold not in reference_scaffolds)
                if generated_scaffold is not None else None
            )
            item = {
                'pocket_id': pocket_id,
                'sample_index': sample_index,
                'validity': valid,
                'canonical_smiles': canonical_smiles,
                'scaffold_smiles': generated_scaffold,
                'scaffold_novel': scaffold_novel,
                'fingerprint': fingerprint,
                'usrcat': descriptor,
            }
            prepared.append(item)
            molecule_metrics[(pocket_id, sample_index)] = {
                'validity': valid,
                'canonical_smiles': canonical_smiles,
                'scaffold_smiles': generated_scaffold,
                'scaffold_novel': scaffold_novel,
            }
        prepared_by_pocket[pocket_id] = prepared
        all_records.extend(prepared)

    pocket_metrics = {
        pocket_id: _calculate_group(records, reference_scaffolds)
        for pocket_id, records in prepared_by_pocket.items()
    }
    global_metrics = _calculate_group(all_records, reference_scaffolds)
    macro_metrics = {}
    for name in (
            'validity', 'uniqueness', 'diversity', 'novelty',
            'internal_diversity_2d', 'internal_diversity_3d'):
        macro_metrics[name] = _finite_stats([
            metrics.get(name) for metrics in pocket_metrics.values()
        ])

    return {
        'definitions': {
            'validity': 'RDKit sanitization success / generated outputs',
            'uniqueness': 'unique canonical structures / sanitized-valid outputs',
            'diversity': 'unique >=2-ring Murcko scaffolds / sanitized-valid outputs',
            'novelty': 'unique generated scaffolds absent from reference / unique generated scaffolds',
            'internal_diversity_2d': '1 - mean pairwise Morgan radius-2 1024-bit Tanimoto',
            'internal_diversity_3d': '1 - mean pairwise USRCAT similarity',
        },
        'molecule_metrics': molecule_metrics,
        'pocket_metrics': pocket_metrics,
        'macro_metrics': macro_metrics,
        'global_metrics': global_metrics,
    }


def reference_scaffolds_from_smiles(smiles_values):
    scaffolds = set()
    valid_molecules = 0
    for smiles in smiles_values:
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        valid_molecules += 1
        value = scaffold_smiles(mol)
        if value is not None:
            scaffolds.add(value)
    return scaffolds, valid_molecules

def _reference_scaffold_cache_path(config):
    configured = getattr(config, 'reference_cache', None)
    if configured:
        base, _ = os.path.splitext(os.path.abspath(os.path.expanduser(configured)))
    else:
        base = os.path.abspath(os.path.expanduser(config.data.path))
    return base + '.scaffold_reference.pkl'


def load_or_build_reference_scaffolds(config, train_smiles, test_set, logger=None):
    """Load or build the train-and-test scaffold reference cache."""
    cache_path = _reference_scaffold_cache_path(config)
    split_path = os.path.abspath(os.path.expanduser(config.data.split))
    train_cache_path = os.path.abspath(os.path.expanduser(getattr(
        config, 'reference_cache',
        config.data.path + '.eval_reference_cache.pkl')))
    metadata = {
        'version': REFERENCE_CACHE_VERSION,
        'dataset': config.dataset,
        'split_path': split_path,
        'split_mtime_ns': os.stat(split_path).st_mtime_ns,
        'train_cache_path': train_cache_path,
        'train_cache_mtime_ns': (
            os.stat(train_cache_path).st_mtime_ns
            if os.path.isfile(train_cache_path) else None
        ),
        'train_smiles': len(train_smiles),
        'test_size': len(test_set),
        'reference_scope': 'crossdocked_train_union_test_known_ligands',
        'min_scaffold_rings': MIN_SCAFFOLD_RINGS,
    }
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as handle:
                cached = pickle.load(handle)
            if cached.get('metadata') == metadata:
                if logger is not None:
                    logger.info(
                        'Loaded scaffold reference cache: %d scaffolds from %s',
                        len(cached['scaffolds']), cache_path)
                return set(cached['scaffolds']), cached['counts'], cache_path
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    'Failed to load scaffold cache %s (%s); rebuilding it',
                    cache_path, exc)

    if logger is not None:
        logger.info('Building scaffold reference from train + test ligands')
    train_scaffolds, train_valid = reference_scaffolds_from_smiles(train_smiles)
    test_scaffolds = set()
    test_valid = 0
    root = (config.protein_root if config.dataset == 'crossdocked'
            else config.data.path)
    for data in test_set:
        ligand_path = os.path.join(root, data.ligand_filename)
        try:
            supplier = Chem.SDMolSupplier(ligand_path)
            mol = supplier[0] if supplier else None
            if mol is None:
                continue
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        test_valid += 1
        value = scaffold_smiles(mol)
        if value is not None:
            test_scaffolds.add(value)

    scaffolds = train_scaffolds.union(test_scaffolds)
    counts = {
        'train_smiles': len(train_smiles),
        'train_valid': train_valid,
        'train_unique_scaffolds': len(train_scaffolds),
        'test_size': len(test_set),
        'test_valid': test_valid,
        'test_unique_scaffolds': len(test_scaffolds),
        'reference_unique_scaffolds': len(scaffolds),
    }
    payload = {
        'metadata': metadata,
        'counts': counts,
        'scaffolds': sorted(scaffolds),
    }
    directory = os.path.dirname(cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = cache_path + '.tmp.%d' % os.getpid()
    try:
        with open(temporary, 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return scaffolds, counts, cache_path
