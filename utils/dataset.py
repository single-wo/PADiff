import os
import pickle
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data

METADATA_PREFIX = b'__'


def lmdb_extent_error(path, environment):
    page_size = int(environment.stat()["psize"])
    last_page = int(environment.info()["last_pgno"])
    minimum_size = (last_page + 1) * page_size
    actual_size = os.path.getsize(path)
    if actual_size < minimum_size:
        return (
            f"LMDB file is truncated: {path} has {actual_size} bytes, but "
            f"its page table requires at least {minimum_size} bytes"
        )
    return None


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def to_torch_dict(data):
    output = {}
    for key, value in data.items():
        output[key] = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


class ProteinLigandData(Data):
    @staticmethod
    def protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)
        if protein_dict is not None:
            for key, value in protein_dict.items():
                instance['protein_' + key] = value
        if ligand_dict is not None:
            for key, value in ligand_dict.items():
                instance['ligand_' + key] = value
        if 'ligand_bond_index' in instance:
            instance['ligand_nbh_list'] = {
                i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                           if instance.ligand_bond_index[0, k].item() == i]
                for i in instance.ligand_bond_index[0]
            }
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key in ('ligand_bond_index', 'ligand_halfedge_index', 'ligand_torsion_index', 'ligand_rotatable_torsion_index', 'ligand_shape_pair_index', 'ligand_angle_pair_index', 'ligand_clash_pair_index'):
            return int(self['ligand_element'].size(0))
        if key in ('ph2atom_edge_index', 'ph2atom_idx'):
            return torch.tensor([[self['phore_type'].size(0)], [self['ligand_element'].size(0)]])
        if key == 'ph2protein_edge_index':
            return torch.tensor([[self['phore_type'].size(0)], [self['protein_element'].size(0)]])
        if key == 'ph2anchor_edge_index':
            return torch.tensor([[self['phore_type'].size(0)], [self['protein_anchor_type'].size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('ligand_bond_index', 'ligand_halfedge_index', 'ligand_torsion_index', 'ligand_rotatable_torsion_index', 'ligand_shape_pair_index', 'ligand_angle_pair_index', 'ligand_clash_pair_index', 'ph2atom_edge_index',
                   'ph2atom_idx', 'ph2protein_edge_index', 'ph2anchor_edge_index'):
            return 1
        if key in ('protein_diagrams', 'ligand_diagrams'):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class ProteinLigandDataset(Dataset):

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = str(path)
        self.transform = transform
        self.database = None
        self.keys = None
        if not self.path.endswith('.lmdb'):
            raise ValueError(
                "ProteinLigandDataset expects a processed .lmdb file; run "
                "`python -m datasets.preprocess_crossdocked` for raw PDB/SDF data"
            )
        self.processed_path = self.path
        if not os.path.exists(self.processed_path):
            raise FileNotFoundError(self.processed_path)

    def _build_db(self):
        if self.database is not None:
            return
        self.database = lmdb.open(
            self.processed_path, create=False, subdir=False,
            readonly=True, lock=False, readahead=False, meminit=False, max_readers=512,
        )
        extent_error = lmdb_extent_error(self.processed_path, self.database)
        if extent_error:
            self.database.close()
            self.database = None
            raise RuntimeError(extent_error)
        with self.database.begin() as db:
            self.keys = [key for key in db.cursor().iternext(values=False)
                         if not bytes(key).startswith(METADATA_PREFIX)]

    def __len__(self):
        self._build_db()
        return len(self.keys)

    def get_raw_record(self, idx):
        self._build_db()
        value = self.database.begin(buffers=False).get(self.keys[idx])
        if value is None:
            raise IndexError(idx)
        record = pickle.loads(value)
        if isinstance(record, Data):
            data = record
        else:
            if 'ph2atom_idx' in record and 'ph2atom_edge_index' not in record:
                record['ph2atom_edge_index'] = record.pop('ph2atom_idx')
            data = ProteinLigandData(**record)
        data.id = idx
        if data.protein_pos.size(0) == 0 or data.ligand_pos.size(0) == 0:
            raise ValueError(f'Empty protein/ligand in record {idx}')
        return data

    def __getitem__(self, idx):
        data = self.get_raw_record(idx)
        return self.transform(data) if self.transform is not None else data


def get_dataset(config, *args, **kwargs):
    name = _config_value(config, 'name')
    if name != 'protein_ligand':
        raise NotImplementedError(f'Unknown dataset name: {name}')
    dataset = ProteinLigandDataset(
        _config_value(config, 'path'), *args, **kwargs
    )
    split_path = _config_value(config, 'split')
    if not split_path:
        return dataset, None

    split = torch.load(split_path, map_location='cpu', weights_only=False)
    split = {key: list(value) for key, value in split.items()}
    subsets = {key: Subset(dataset, indices=value) for key, value in split.items()}
    return dataset, subsets
