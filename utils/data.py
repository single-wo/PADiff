import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

atom_families = [
    "Acceptor", "Donor", "Aromatic", "Hydrophobe", 
    "LumpedHydrophobe", "NegIonizable", "PosIonizable", "ZnBinder"
]
atom_families_id = {f: i for i, f in enumerate(atom_families)}
bond_types = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

class PDBProtein(object):
    aa_name_sym = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
        "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
        "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    aa_name_number = {k: i for i, (k, _) in enumerate(aa_name_sym.items())}
    backbone_names = ["CA", "C", "N", "O"]

    def __init__(self, data, mode="auto"):
        super().__init__()
        if (data[-4:].lower() == ".pdb" and mode == "auto") or mode == "path":
            with open(data, "r") as f:
                self.block = f.read()
        else:
            self.block = data
        self.periodtable = Chem.GetPeriodicTable()
        self.title = None
        self.element = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == "ATOM":
                element_sym = line[76:78].strip().capitalize()
                if len(element_sym) == 0:
                    element_sym = line[13:14]
                if element_sym != 'H':
                    yield {
                        "type": "ATOM",
                        "element_sym": element_sym,
                        "atom_name": line[12:16].strip(),
                        "res_name": line[17:20].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                    }
            elif line[0:6].strip() == "HETATM" and line[17:20].strip() != "HOH":
                element_sym = line[76:78].strip().capitalize()
                if len(element_sym) == 0:
                    element_sym = line[12:14]
                if element_sym != 'H':
                    yield {
                        "type": "HETATM",
                        "element_sym": element_sym,
                        "atom_name": line[12:16].strip(),
                        "res_name": line[17:20].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                    }
            elif line[0:6].strip() == "HEADER":
                yield {
                    "type": "HEADER",
                    "value": line[10:].strip()
                }
            elif line[0:6].strip() == "ENDMDL":
                break

    def _parse(self):
        for atom in self._enum_formatted_atom_lines():
            if atom["type"] == "HEADER":
                self.title = atom["value"].lower()
                continue
            elif atom["type"] == "ATOM":
                atomic_number = self.periodtable.GetAtomicNumber(atom["element_sym"])
                self.element.append(atomic_number)
                self.pos.append(np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float32))
                self.atom_name.append(atom["atom_name"])
                self.is_backbone.append(atom["atom_name"] in self.backbone_names)
                self.atom_to_aa_type.append(self.aa_name_number[atom["res_name"]])
            elif atom["type"] == "HETATM":
                atomic_number = self.periodtable.GetAtomicNumber(atom["element_sym"])
                self.element.append(atomic_number)
                self.pos.append(np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float32))
                self.atom_name.append(atom["atom_name"])
                self.is_backbone.append(atom["atom_name"] in self.backbone_names)
                self.atom_to_aa_type.append(int(len(self.aa_name_sym)))

    def to_dict_atom(self):
        return {
            "element": np.array(self.element, dtype=np.int64),
            "molecule_name": self.title,
            "pos": np.array(self.pos, dtype=np.float32),
            "is_backbone": np.array(self.is_backbone, dtype=np.bool_),
            "atom_name": self.atom_name,
            "atom_to_aa_type": np.array(self.atom_to_aa_type, dtype=np.int64),
        }

def load_ligand_molecule(path):
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix == ".sdf":
        mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError("Unknown ligand file; expected an SDF or MOL2 file")
    if mol is None:
        raise ValueError("RDKit could not read ligand file: %s" % path)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand has no 3D conformer: %s" % path)
    return mol


def parse_sdf_file(path, extract_phore=False):
    if extract_phore:
        raise ValueError(
            "parse_sdf_file no longer extracts pharmacophores; use "
            "build_training_pharmacophore_targets instead"
        )
    mol = load_ligand_molecule(path)
    num_atoms = mol.GetNumAtoms()
    positions = np.asarray(
        mol.GetConformer().GetPositions(), dtype=np.float32
    )
    elements = np.asarray(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64
    )
    hybridization = [str(atom.GetHybridization()) for atom in mol.GetAtoms()]

    feature_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(feature_path)
    atom_features = np.zeros(
        (num_atoms, len(atom_families)), dtype=np.int64
    )
    for feature in factory.GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family in atom_families_id:
            atom_features[list(feature.GetAtomIds()), atom_families_id[family]] = 1

    rows, columns, edge_types = [], [], []
    for bond in mol.GetBonds():
        bond_type = bond_types.get(bond.GetBondType())
        if bond_type is None:
            raise ValueError("Unsupported RDKit bond type: %s" % bond.GetBondType())
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows.extend((start, end))
        columns.extend((end, start))
        edge_types.extend((bond_type, bond_type))
    bond_index = np.asarray([rows, columns], dtype=np.int64)
    bond_type = np.asarray(edge_types, dtype=np.int64)
    if bond_index.shape[1]:
        order = (bond_index[0] * num_atoms + bond_index[1]).argsort()
        bond_index = bond_index[:, order]
        bond_type = bond_type[order]

    masses = np.asarray(
        [atom.GetMass() for atom in mol.GetAtoms()], dtype=np.float64
    )
    center_of_mass = np.average(positions, axis=0, weights=masses).astype(
        np.float32
    )
    return {
        "smiles": Chem.MolToSmiles(mol),
        "element": elements,
        "pos": positions,
        "bond_index": bond_index,
        "bond_type": bond_type,
        "num_atoms": num_atoms,
        "num_bonds": mol.GetNumBonds(),
        "center_of_mass": center_of_mass,
        "atom_feature": atom_features,
        "hybridization": hybridization,
    }
