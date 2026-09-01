import argparse
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data import PDBProtein, parse_sdf_file
from utils.dataset import ProteinLigandData, lmdb_extent_error, to_torch_dict
from utils.pharmacophore_preprocessing import (
    build_training_pharmacophore_targets,
    load_heavy_ligand,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / 'raw_data' / 'crossdocked_v1.1_rmsd1.0_pocket10'
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data'
PADIFF_SCHEMA_VERSION = 2
PADIFF_METADATA_KEY = b'__padiff_metadata__'
LEGACY_METADATA_KEY = b'__anchor_v2_metadata__'

MOTIF_FIELDS = {
    'atom_cluster_array',
    'atom_vocab_array',
    'motif_pos',
    'node_wid',
    'size_vocab',
    'ph2motif_edge_index',
}

PADIFF_REQUIRED_FIELDS = {
    'phore_pos',
    'phore_vec',
    'phore_type',
    'ph2atom_edge_index',
    'protein_anchor_pos',
    'protein_anchor_vec',
    'protein_anchor_type',
    'protein_anchor_confidence',
    'protein_anchor_source_atom',
    'ph2anchor_edge_index',
    'ph2anchor_contact_label',
    'ph2anchor_contact_type',
    'ph2anchor_contact_confidence',
    'phore_contact_schema_version',
}


def _empty_edge():
    return torch.zeros((2, 0), dtype=torch.long)


def _tensor_size0(value):
    return int(value.size(0)) if torch.is_tensor(value) and value.dim() > 0 else 0


def validate_padiff_record(record):
    """Return a list of schema/index errors for one serialized LMDB record."""
    errors = []
    missing = sorted(PADIFF_REQUIRED_FIELDS.difference(record.keys()))
    if missing:
        return ['missing fields: %s' % ', '.join(missing)]

    version = record['phore_contact_schema_version']
    if torch.is_tensor(version):
        version = int(version.reshape(-1)[0].item()) if version.numel() else -1
    else:
        version = int(version)
    if version != PADIFF_SCHEMA_VERSION:
        errors.append('schema version %s != %s' % (version, PADIFF_SCHEMA_VERSION))

    phore_pos = record['phore_pos']
    phore_vec = record['phore_vec']
    phore_type = record['phore_type']
    anchor_pos = record['protein_anchor_pos']
    anchor_vec = record['protein_anchor_vec']
    anchor_type = record['protein_anchor_type']

    if not torch.is_tensor(phore_pos) or phore_pos.dim() != 2 or phore_pos.size(-1) != 3:
        errors.append('phore_pos must have shape [P, 3]')
        num_phore = 0
    else:
        num_phore = int(phore_pos.size(0))
    if not torch.is_tensor(phore_vec) or tuple(phore_vec.shape) != tuple(phore_pos.shape):
        errors.append('phore_vec shape must match phore_pos')
    if not torch.is_tensor(phore_type) or _tensor_size0(phore_type) != num_phore:
        errors.append('phore_type length must match phore_pos')

    if not torch.is_tensor(anchor_pos) or anchor_pos.dim() != 2 or anchor_pos.size(-1) != 3:
        errors.append('protein_anchor_pos must have shape [A, 3]')
        num_anchor = 0
    else:
        num_anchor = int(anchor_pos.size(0))
    if not torch.is_tensor(anchor_vec) or tuple(anchor_vec.shape) != tuple(anchor_pos.shape):
        errors.append('protein_anchor_vec shape must match protein_anchor_pos')
    if not torch.is_tensor(anchor_type) or _tensor_size0(anchor_type) != num_anchor:
        errors.append('protein_anchor_type length must match protein_anchor_pos')
    for key in ('protein_anchor_confidence', 'protein_anchor_source_atom'):
        if not torch.is_tensor(record[key]) or _tensor_size0(record[key]) != num_anchor:
            errors.append('%s length must match protein anchors' % key)

    ligand_element = record.get('ligand_element')
    num_ligand = _tensor_size0(ligand_element)
    for key, upper0, upper1 in (
        ('ph2atom_edge_index', num_phore, num_ligand),
        ('ph2anchor_edge_index', num_phore, num_anchor),
    ):
        edge = record[key]
        if not torch.is_tensor(edge) or edge.dim() != 2 or edge.size(0) != 2:
            errors.append('%s must have shape [2, E]' % key)
            continue
        if edge.numel() > 0:
            if int(edge.min().item()) < 0:
                errors.append('%s contains a negative index' % key)
            if upper0 <= 0 or int(edge[0].max().item()) >= upper0:
                errors.append('%s pharmacophore index out of range' % key)
            if upper1 <= 0 or int(edge[1].max().item()) >= upper1:
                errors.append('%s target index out of range' % key)

    num_contacts = int(record['ph2anchor_edge_index'].size(1))
    for key in ('ph2anchor_contact_label', 'ph2anchor_contact_type', 'ph2anchor_contact_confidence'):
        if not torch.is_tensor(record[key]) or _tensor_size0(record[key]) != num_contacts:
            errors.append('%s length must match ph2anchor edges' % key)
    labels = record['ph2anchor_contact_label']
    if torch.is_tensor(labels) and labels.numel() > 0:
        bad = set(int(x) for x in labels.unique().tolist()).difference({-1, 0, 1})
        if bad:
            errors.append('invalid contact labels: %s' % sorted(bad))

    stale_motif = sorted(MOTIF_FIELDS.intersection(record.keys()))
    if stale_motif:
        errors.append('legacy motif fields present: %s' % ', '.join(stale_motif))
    return errors


def inspect_padiff_lmdb(lmdb_path, max_records=3):
    report = {'valid': False, 'count': 0, 'checked': 0, 'errors': []}
    if not os.path.isfile(lmdb_path):
        report['errors'].append('LMDB file does not exist')
        return report
    try:
        env = lmdb.open(
            lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        extent_error = lmdb_extent_error(lmdb_path, env)
        if extent_error:
            report['errors'].append(extent_error)
            env.close()
            return report
        with env.begin() as txn:
            total_entries = int(txn.stat().get('entries', 0))
            metadata_blobs = [
                txn.get(key) for key in (PADIFF_METADATA_KEY, LEGACY_METADATA_KEY)
            ]
            metadata_blob = next((blob for blob in metadata_blobs if blob), None)
            metadata_count = sum(blob is not None for blob in metadata_blobs)
            metadata = None
            if metadata_blob is None:
                report['errors'].append('missing completion metadata (dataset may be legacy or partially written)')
            else:
                try:
                    metadata = pickle.loads(metadata_blob)
                    if not metadata.get('completed', False):
                        report['errors'].append('completion metadata is not marked completed')
                    if int(metadata.get('schema_version', -1)) != PADIFF_SCHEMA_VERSION:
                        report['errors'].append('metadata schema version mismatch')
                except Exception as exc:
                    report['errors'].append('cannot decode completion metadata: %s' % exc)
            report['count'] = total_entries - metadata_count
            if metadata is not None and int(metadata.get('processed_count', -1)) != report['count']:
                report['errors'].append(
                    'metadata processed_count %s != LMDB data records %s' %
                    (metadata.get('processed_count'), report['count'])
                )
            cursor = txn.cursor()
            for key, value in cursor:
                if bytes(key).startswith(b'__'):
                    continue
                if report['checked'] >= max_records:
                    break
                try:
                    record = pickle.loads(value)
                    errors = validate_padiff_record(record)
                    if errors:
                        report['errors'].append('%s: %s' % (key.decode(errors='replace'), '; '.join(errors)))
                except Exception as exc:
                    report['errors'].append('%s: cannot decode record: %s' % (key, exc))
                report['checked'] += 1
        env.close()
    except Exception as exc:
        report['errors'].append('cannot open LMDB: %s' % exc)
    if report['count'] <= 0:
        report['errors'].append('LMDB contains zero records')
    report['valid'] = not report['errors']
    return report


class PocketLigandPairDataset(Dataset):
    """Build or inspect the canonical CrossDocked PADiff LMDB."""

    def __init__(
        self,
        raw_path,
        transform=None,
        limit=None,
        custom_lmdb_path=None,
        overwrite=False,
        map_size=35 * (1024 ** 3),
    ):
        super().__init__()

        self.raw_path = os.path.abspath(raw_path.rstrip('/'))
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        if custom_lmdb_path is not None:
            self.processed_path = os.path.abspath(custom_lmdb_path)
        elif limit is not None:
            self.processed_path = str(
                DEFAULT_DATA_ROOT /
                ('crossdocked_v1.1_rmsd1.0_pocket10_processed_smoke_%d.lmdb' % limit)
            )
        else:
            self.processed_path = str(
                DEFAULT_DATA_ROOT / 'crossdocked_v1.1_rmsd1.0_pocket10_processed.lmdb'
            )

        self.transform = transform
        self.db = None
        self.keys = None
        self.limit = limit
        self.map_size = int(map_size)

        if overwrite and os.path.exists(self.processed_path):
            os.remove(self.processed_path)
            lock_path = self.processed_path + '-lock'
            if os.path.exists(lock_path):
                os.remove(lock_path)
            print('[INFO] Removed existing LMDB because --overwrite was specified: %s' % self.processed_path)

        if not os.path.exists(self.processed_path):
            print('[INFO] %s does not exist, begin processing data...' % self.processed_path)
            if self.limit:
                print('[INFO] Smoke mode: processing only %d successful samples.' % self.limit)
            self._process()
        else:
            report = inspect_padiff_lmdb(self.processed_path)
            if not report['valid']:
                raise RuntimeError(
                    'Existing LMDB is not a valid PADiff dataset and will not be silently reused: %s\n'
                    'Errors: %s\nUse --overwrite to rebuild it.' %
                    (self.processed_path, ' | '.join(report['errors']))
                )
            print('[INFO] Found valid PADiff dataset at %s (%d entries). Skipping processing.' %
                  (self.processed_path, report['count']))

    def _connect_db(self):
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = [
                key for key in txn.cursor().iternext(values=False)
                if not bytes(key).startswith(b'__')
            ]

    def _close_db(self):
        if self.db is not None:
            self.db.close()
        self.db = None
        self.keys = None

    @staticmethod
    def _attach_pharmacophore_targets(data, phore_data):
        data.phore_pos = phore_data['phore_pos']
        data.phore_vec = phore_data['phore_vec']
        data.phore_type = phore_data['phore_type']
        data.ph2atom_edge_index = phore_data['ph2atom_idx']

        for key in (
            'protein_anchor_pos', 'protein_anchor_vec', 'protein_anchor_type',
            'protein_anchor_confidence', 'protein_anchor_source_atom',
            'ph2anchor_edge_index', 'ph2anchor_contact_label',
            'ph2anchor_contact_type', 'ph2anchor_contact_confidence',
        ):
            setattr(data, key, phore_data[key])

        data.ph2protein_edge_index = _empty_edge()
        data.ph2protein_contact_label = torch.zeros((0,), dtype=torch.long)
        data.ph2protein_contact_type = torch.zeros((0,), dtype=torch.long)
        data.phore_contact_schema_version = torch.tensor([PADIFF_SCHEMA_VERSION], dtype=torch.long)

    def _process(self):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        db = lmdb.open(
            self.processed_path,
            map_size=self.map_size,
            create=True,
            subdir=False,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
        )

        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        processed_count = 0
        positive_count = 0
        negative_count = 0
        ignore_count = 0
        error_log_path = os.path.join(self.raw_path, 'padiff_preprocessing_errors.log')
        print('[INFO] Preprocessing errors will be written to: %s' % error_log_path)

        txn = db.begin(write=True)
        try:
            with open(error_log_path, 'w') as log_f:
                pbar = tqdm(index, desc='Building PADiff LMDB')
                for i, item in enumerate(pbar):
                    pocket_fn, ligand_fn = item[0], item[1]
                    pbar.set_postfix(
                        success=processed_count,
                        skipped=num_skipped,
                        pos=positive_count,
                        neg=negative_count,
                        ign=ignore_count,
                    )
                    if self.limit is not None and processed_count >= self.limit:
                        print('[INFO] Reached limit of %d successful samples. Stopping.' % self.limit)
                        break
                    if pocket_fn is None or ligand_fn is None:
                        continue

                    pocket_path = os.path.join(self.raw_path, pocket_fn)
                    ligand_path = os.path.join(self.raw_path, ligand_fn)
                    try:
                        pdb = PDBProtein(pocket_path)
                        protein_dict = pdb.to_dict_atom()
                        ligand_dict = parse_sdf_file(ligand_path, extract_phore=False)
                        heavy_mol, heavy_pos = load_heavy_ligand(ligand_path)
                        ligand_pos = np.asarray(ligand_dict['pos'], dtype=np.float32)
                        if heavy_mol.GetNumAtoms() != ligand_pos.shape[0]:
                            raise ValueError(
                                'ligand heavy-atom count mismatch: extractor=%d dataset=%d' %
                                (heavy_mol.GetNumAtoms(), ligand_pos.shape[0])
                            )
                        if heavy_pos.shape != ligand_pos.shape or not np.allclose(heavy_pos, ligand_pos, atol=1e-3, rtol=0.0):
                            max_err = float(np.max(np.abs(heavy_pos - ligand_pos))) if heavy_pos.shape == ligand_pos.shape else float('inf')
                            raise ValueError('ligand heavy-atom coordinates are not aligned (max error %.6f)' % max_err)

                        phore_data = build_training_pharmacophore_targets(heavy_mol)

                        data = ProteinLigandData.protein_ligand_dicts(
                            protein_dict=to_torch_dict(protein_dict),
                            ligand_dict=to_torch_dict(ligand_dict),
                        )
                        data.protein_filename = pocket_fn
                        data.ligand_filename = ligand_fn
                        self._attach_pharmacophore_targets(data, phore_data)

                        data_dict = data.to_dict()
                        for field in MOTIF_FIELDS:
                            data_dict.pop(field, None)

                        data_dict.pop('protein_anchor_metadata', None)
                        data_dict.pop('ph2anchor_geometry', None)

                        errors = validate_padiff_record(data_dict)
                        if errors:
                            raise ValueError('PADiff schema validation failed: %s' % '; '.join(errors))

                        labels = data_dict['ph2anchor_contact_label']
                        positive_count += int((labels == 1).sum().item())
                        negative_count += int((labels == 0).sum().item())
                        ignore_count += int((labels == -1).sum().item())

                        txn.put(key=str(i).encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                        processed_count += 1
                        if processed_count % 1000 == 0:
                            txn.commit()
                            txn = db.begin(write=True)

                    except Exception as exc:
                        num_skipped += 1
                        if num_skipped <= 5:
                            print('\n[ALERT] Sample %d skipped: %s' % (i, exc))
                        log_f.write(
                            'Index %d | Pocket: %s | Ligand: %s | Error: %s\n' %
                            (i, pocket_fn, ligand_fn, exc)
                        )
                        log_f.flush()
                        continue

                metadata = {
                    'schema_version': PADIFF_SCHEMA_VERSION,
                    'completed': True,
                    'source_index_count': len(index),
                    'limit': self.limit,
                    'processed_count': processed_count,
                    'skipped_count': num_skipped,
                    'positive_count': positive_count,
                    'hard_negative_count': negative_count,
                    'ignore_count': ignore_count,
                }
                txn.put(PADIFF_METADATA_KEY, pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))
                txn.commit()
                txn = None
        finally:
            if txn is not None:
                txn.abort()
            db.sync()
            db.close()

        print('[INFO] Finished PADiff preprocessing.')
        print('       processed=%d skipped=%d' % (processed_count, num_skipped))
        print('       positive=%d hard-negative=%d ignore=%d' %
              (positive_count, negative_count, ignore_count))
        if processed_count == 0:
            raise RuntimeError('No records were written; inspect %s' % error_log_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_raw_record(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_raw_record(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data


def verify_padiff_dataset(lmdb_path, max_records=3):
    print('\n%s VERIFICATION REPORT %s' % ('=' * 20, '=' * 20))
    print('Checking LMDB file: %s' % lmdb_path)
    report = inspect_padiff_lmdb(lmdb_path, max_records=max_records)
    print('Entries: %d; checked: %d' % (report['count'], report['checked']))
    if report['valid']:
        print('✅ PADiff schema/index validation: PASS')
    else:
        print('❌ PADiff schema/index validation: FAIL')
        for error in report['errors']:
            print('   - %s' % error)
    print('=' * 61)
    return report['valid']


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build the CrossDocked PADiff LMDB with six-class ligand '
            'pharmacophore supervision.'
        )
    )
    parser.add_argument('--path', type=str, default=str(DEFAULT_RAW_ROOT), help='Raw CrossDocked root containing index.pkl.')
    parser.add_argument('--output', type=str, default=None, help='Explicit output LMDB path.')
    parser.add_argument('--limit', type=int, default=None, help='Process only N successful complexes for a smoke test.')
    parser.add_argument('--overwrite', action='store_true', help='Delete and rebuild an existing output LMDB.')
    parser.add_argument('--verify_records', type=int, default=3, help='Number of records to validate after preprocessing.')
    parser.add_argument(
        '--map-size-gb', type=float, default=35.0,
        help='Maximum LMDB size while writing; reduce this for small smoke tests.',
    )
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(
        args.path,
        limit=args.limit,
        custom_lmdb_path=args.output,
        overwrite=args.overwrite,
        map_size=int(args.map_size_gb * (1024 ** 3)),
    )

    print('\n[TEST] Reading record 0 to validate PADiff fields...')
    if len(dataset) > 0:
        sample_data = dataset[0]
        print('Phore nodes: %d' % sample_data.phore_type.size(0))
        print('Protein anchors: %d' % sample_data.protein_anchor_type.size(0))
        print('Phore-anchor edges: %d' % sample_data.ph2anchor_edge_index.size(1))

    if not verify_padiff_dataset(dataset.processed_path, max_records=args.verify_records):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
