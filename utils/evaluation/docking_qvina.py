import os
import shutil
import subprocess
import random
import string
import sys
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def resolve_crossdocked_protein_path(protein_root, ligand_filename):
    """Resolve either the original receptor PDB or this dataset's pocket10 PDB."""
    receptor_fn = os.path.join(
        os.path.dirname(ligand_filename),
        os.path.basename(ligand_filename)[:10] + '.pdb'
    )
    receptor_path = os.path.join(protein_root, receptor_fn)
    if os.path.isfile(receptor_path):
        return receptor_path

    pocket_fn = os.path.splitext(ligand_filename)[0] + '_pocket10.pdb'
    pocket_path = os.path.join(protein_root, pocket_fn)
    if os.path.isfile(pocket_path):
        return pocket_path

    raise FileNotFoundError(
        'CrossDocked receptor/pocket file not found. Tried: %s and %s'
        % (receptor_path, pocket_path)
    )


def parse_qvina_outputs(docked_sdf_path):
    suppl = Chem.SDMolSupplier(docked_sdf_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
        }))

    return results


class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_mol_crossdocked(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        protein_path = resolve_crossdocked_protein_path(protein_root, ligand_filename)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        return cls(pdb_block, ligand_rdmol, **kwargs)
    
    @classmethod
    def from_generated_mol_pdbbind(cls, ligand_rdmol, pdbid, protein_root='./data/pdbbind', **kwargs):
        # load original pdb
        protein_fn =  os.path.join(pdbid, pdbid + '_protein.pdb')  # PDBId_protein.pdb
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        return cls(pdb_block, ligand_rdmol, **kwargs)

    def __init__(self, pdb_block, ligand_rdmol, tmp_dir='./vina_tmp', use_uff=True, center=None,
                 size_factor=1.):
        super().__init__(pdb_block, ligand_rdmol)
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if use_uff:
            UFFOptimizeMolecule(ligand_rdmol)
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor

        self.results = None
        self.docked_sdf_path = None

    def run(self, exhaustiveness=16):
        """Run QVina using tools from the active environment/PATH."""
        receptor_preparer = shutil.which('prepare_receptor4.py')
        if receptor_preparer is None:
            try:
                import AutoDockTools
            except ImportError:
                AutoDockTools = None
            if AutoDockTools is not None:
                candidate = os.path.join(
                    AutoDockTools.__path__[0],
                    'Utilities24',
                    'prepare_receptor4.py',
                )
                if os.path.isfile(candidate):
                    receptor_preparer = candidate
        if receptor_preparer is None:
            raise FileNotFoundError(
                'prepare_receptor4.py was not found in PATH or AutoDockTools_py3'
            )
        obabel = shutil.which('obabel')
        qvina = shutil.which('qvina2')
        if obabel is None:
            raise FileNotFoundError('obabel was not found in PATH')
        if qvina is None:
            raise FileNotFoundError('qvina2 was not found in PATH')

        receptor_pdbqt = os.path.join(
            self.tmp_dir, self.receptor_id + '.pdbqt'
        )
        ligand_pdbqt = os.path.join(self.tmp_dir, self.ligand_id + '.pdbqt')
        docked_pdbqt = os.path.join(
            self.tmp_dir, self.ligand_id + '_out.pdbqt'
        )
        self.docked_sdf_path = os.path.join(
            self.tmp_dir, self.ligand_id + '_out.sdf'
        )
        subprocess.run(
            [sys.executable, receptor_preparer, '-r', self.receptor_path,
             '-o', receptor_pdbqt],
            cwd=self.tmp_dir,
            check=True,
        )
        subprocess.run(
            [obabel, self.ligand_path, '-O', ligand_pdbqt],
            cwd=self.tmp_dir,
            check=True,
        )
        subprocess.run(
            [
                qvina,
                '--receptor', receptor_pdbqt,
                '--ligand', ligand_pdbqt,
                '--center_x', '%.4f' % self.center[0],
                '--center_y', '%.4f' % self.center[1],
                '--center_z', '%.4f' % self.center[2],
                '--size_x', str(float(self.size_x)),
                '--size_y', str(float(self.size_y)),
                '--size_z', str(float(self.size_z)),
                '--exhaustiveness', str(int(exhaustiveness)),
                '--out', docked_pdbqt,
            ],
            cwd=self.tmp_dir,
            check=True,
        )
        subprocess.run(
            [obabel, docked_pdbqt, '-O', self.docked_sdf_path, '-h'],
            cwd=self.tmp_dir,
            check=True,
        )
        self.results = parse_qvina_outputs(self.docked_sdf_path)
        return self.results

    def run_sync(self, exhaustiveness=16):
        results = self.run(exhaustiveness=exhaustiveness)
        if results:
            print('Best affinity:', results[0]['affinity'])
        return results
