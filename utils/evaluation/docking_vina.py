try:
    from openbabel import pybel
except ImportError:  
try:
    from meeko import MoleculePreparation, obutils
except ImportError:  
    MoleculePreparation = None
    obutils = None
try:
    from vina import Vina
except ImportError: 
    Vina = None
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    import AutoDockTools
except ImportError:  
    AutoDockTools = None
import os
import contextlib
import shutil
import sys

from utils.evaluation.docking_qvina import (
    get_random_id, BaseDockingTask, resolve_crossdocked_protein_path
)


_PDBQT_ELEMENT_MAP = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
    "P": "P", "F": "F", "Cl": "Cl", "CL": "Cl",
    "Br": "Br", "BR": "Br", "I": "I", "H": "H", "HD": "H",
}


def _require_dependency(value, package_name):
    if value is None:
        raise ImportError(
            "%s is required for AutoDock Vina preparation/docking" % package_name
        )
    return value


def _pdbqt_atoms(pdbqt_text):
    """Return PDBQT atoms in file order as element/coordinate dictionaries."""
    atoms = []
    for line in str(pdbqt_text or "").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        atom_type = fields[-1]
        element = _PDBQT_ELEMENT_MAP.get(atom_type, atom_type)
        try:
            coordinates = np.asarray([
                float(line[30:38]), float(line[38:46]), float(line[46:54]),
            ], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        atoms.append({
            "serial": int(fields[1]),
            "atom_type": atom_type,
            "element": element,
            "coordinates": coordinates,
        })
    return atoms


def rdmol_from_pdbqt_pose(reference_mol, input_pdbqt, pose_pdbqt,
                           coordinate_tolerance=0.10):
    """Map a Vina PDBQT pose back onto ``reference_mol`` atom ordering.

    Open Babel may reorder atoms while preparing PDBQT.  The input PDBQT is
    therefore matched to the original heavy atoms by element and their
    unchanged pre-docking coordinates.  Vina preserves that PDBQT atom order,
    allowing the docked coordinates to be transferred without guessing bond
    orders from the pose file.
    """
    if reference_mol is None or reference_mol.GetNumConformers() == 0:
        raise ValueError("reference_mol must contain a conformer")
    original_indices = [
        atom.GetIdx() for atom in reference_mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    input_atoms = [
        atom for atom in _pdbqt_atoms(input_pdbqt) if atom["element"] != "H"
    ]
    pose_atoms = [
        atom for atom in _pdbqt_atoms(pose_pdbqt) if atom["element"] != "H"
    ]
    if len(input_atoms) != len(original_indices):
        raise ValueError(
            "Input PDBQT heavy-atom count %d differs from RDKit count %d"
            % (len(input_atoms), len(original_indices))
        )
    if len(pose_atoms) != len(input_atoms):
        raise ValueError(
            "Docked PDBQT heavy-atom count %d differs from input count %d"
            % (len(pose_atoms), len(input_atoms))
        )

    conformer = reference_mol.GetConformer(0)
    costs = np.full((len(original_indices), len(input_atoms)), 1.0e6)
    for row, atom_index in enumerate(original_indices):
        atom = reference_mol.GetAtomWithIdx(atom_index)
        position = np.asarray(conformer.GetAtomPosition(atom_index), dtype=np.float64)
        for column, pdbqt_atom in enumerate(input_atoms):
            if atom.GetSymbol() == pdbqt_atom["element"]:
                costs[row, column] = np.linalg.norm(
                    position - pdbqt_atom["coordinates"]
                )
    rows, columns = linear_sum_assignment(costs)
    assigned_costs = costs[rows, columns]
    if len(assigned_costs) != len(original_indices) or (
        assigned_costs.size and assigned_costs.max() > float(coordinate_tolerance)
    ):
        maximum = float(assigned_costs.max()) if assigned_costs.size else None
        raise ValueError(
            "Could not map PDBQT atoms to RDKit order within %.3f A; max=%s"
            % (float(coordinate_tolerance), maximum)
        )

    pdbqt_to_rdkit = {
        int(column): int(original_indices[row])
        for row, column in zip(rows, columns)
    }
    docked = Chem.Mol(reference_mol)
    docked.RemoveAllConformers()
    docked_conformer = Chem.Conformer(docked.GetNumAtoms())
    for atom_index in range(docked.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        docked_conformer.SetAtomPosition(atom_index, point)
    for pdbqt_index, pose_atom in enumerate(pose_atoms):
        atom_index = pdbqt_to_rdkit[pdbqt_index]
        x, y, z = pose_atom["coordinates"]
        docked_conformer.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
    docked.AddConformer(docked_conformer, assignId=True)
    return docked


def suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class LigandPreparer(object):
    def __init__(self, input_mol, mol_format):
        _require_dependency(pybel, "Open Babel")
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        _require_dependency(obutils, "Meeko")
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        _require_dependency(obutils, "Meeko")
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @suppress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        _require_dependency(MoleculePreparation, "Meeko")
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None: 
            preparator.write_pdbqt_file(lig_pdbqt)
            return 
        else: 
            return preparator.write_pdbqt_string()
        

class ReceptorPreparer(object):
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): 
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):
        self.prot_pqr = prot_pqr
        executable = shutil.which('pdb2pqr30')
        if executable is None:
            candidate = os.path.join(os.path.dirname(sys.executable), 'pdb2pqr30')
            executable = candidate if os.path.isfile(candidate) else None
        if executable is None:
            raise FileNotFoundError('pdb2pqr30 was not found in PATH or the active Python environment')
        subprocess.run(
            [executable, '--ff=AMBER', self.prot, self.prot_pqr],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True,
        )

    def get_pdbqt(self, prot_pdbqt):
        _require_dependency(AutoDockTools, "AutoDockTools_py3")
        prepare_receptor = os.path.join(
            AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py'
        )
        subprocess.run(
            [sys.executable, prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True,
        )


class VinaDockingRunner(object):
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        _require_dependency(Vina, "vina")
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_mol_crossdocked(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        protein_path = resolve_crossdocked_protein_path(protein_root, ligand_filename)
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol_pdbbind(cls, ligand_rdmol, pdbid, protein_root='./data/pdbbind', **kwargs):
        # load original pdb
        protein_fn =  os.path.join(pdbid, pdbid + '_protein.pdb')  # PDBId_protein.pdb
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)
    
    @classmethod
    def from_generated_mol(cls, ligand_rdmol, protein_path, **kwargs):
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./vina_tmp', center=None,
                 size_factor=1., buffer=5.0, box_size=None):
        super().__init__(protein_path, ligand_rdmol)
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if box_size is not None:
            if len(box_size) != 3:
                raise ValueError('box_size must contain exactly three dimensions')
            self.size_x, self.size_y, self.size_z = map(float, box_size)
        elif size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = LigandPreparer(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        prot = ReceptorPreparer(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDockingRunner(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(
            score_func='vina', mode=mode, exhaustiveness=exhaustiveness,
            save_pose=True, **kwargs
        )
        result = {'affinity': score, 'pose': pose}
        if pose is not None:
            try:
                with open(ligand_pdbqt) as handle:
                    input_pdbqt = handle.read()
                result['rdmol'] = rdmol_from_pdbqt_pose(
                    self.recon_ligand_mol, input_pdbqt, pose
                )
            except Exception as exc:
                result['pose_conversion_error'] = '%s: %s' % (
                    type(exc).__name__, str(exc)
                )
        return [result]
