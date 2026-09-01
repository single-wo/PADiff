import pickle
from rdkit import Chem
import numpy as np
from tqdm import tqdm
from utils.evaluation.scoring_func import get_rdkit_optimize_rmsd, get_rdkit_rmsd
from multiprocessing import Pool
from functools import partial
from scipy import spatial as sci_spatial
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils.data as utils_data


def calculate_3d_geometry(mol, obj, geometry_type):
    assert geometry_type in ['length', 'angle', 'dihedral']
    if geometry_type == 'length':
        func = Chem.rdMolTransforms.GetBondLength
    elif geometry_type == 'angle':
        func = Chem.rdMolTransforms.GetAngleDeg
    elif geometry_type == 'dihedral':
        func = Chem.rdMolTransforms.GetDihedralDeg
    
    matches = mol.GetSubstructMatches(obj)
    results = []
    for match in matches:
        value = func(mol.GetConformer(), *match)
        results.append(value)
    return results

class LocalGeometryStatistics:
    def __init__(self, bonds=None, angles=None, dihedrals=None):
        if bonds is not None:
            self.bonds = [Chem.MolFromSmarts(b) for b in bonds]
        if angles is not None:
            self.angles = [Chem.MolFromSmarts(a) for a in angles]
        if dihedrals is not None:
            self.dihedrals = [Chem.MolFromSmarts(d) for d in dihedrals]

    def define_default_patterns(self):
        """Frequent bonds/angles/dihedrals"""
        bonds_smarts = ['c:c', '[#6]-[#6]', '[#6]-[#7]', '[#6]-O', 'c:n', '[#6]=O', '[#6]-S', 'O=S','c:o', 'c:s',
                '[#6]-F', 'n:n', '[#6]-Cl', '[#6]=[#6]', '[#7]-S', '[#6]=[#7]', '[#7]-[#7]', '[#7]-O', '[#6]=S', '[#7]=O']
        angles_smarts = ['c:c:c', '[#6]-[#6]-[#6]', '[#6]-[#7]-[#6]', '[#7]-[#6]-[#6]', 'c:c-[#6]', '[#6]-O-[#6]', 'O=[#6]-[#6]', '[#7]-c:c',
                'n:c:c', 'c:c-O', 'c:n:c', '[#6]-[#6]-O', 'O=[#6]-[#7]', ]
        dihedrals_smarts = ['c:c:c:c', '[#6]-[#6]-[#6]-[#6]', '[#6]-[#7]-[#6]-[#6]', '[#6]-c:c:c', '[#7]-[#6]-[#6]-[#6]', '[#7]-c:c:c', 'O-c:c:c',
                  '[#6]-[#7]-c:c', '[#7]-[#6]-c:c', 'n:c:c:c', '[#6]-[#7]-[#6]=O', '[#6]-[#6]-c:c', 'c:c-[#7]-[#6]', 'c:n:c:c', '[#6]-O-c:c']

        self.bonds = [Chem.MolFromSmarts(b) for b in bonds_smarts]
        self.angles = [Chem.MolFromSmarts(a) for a in angles_smarts]
        self.dihedrals = [Chem.MolFromSmarts(d) for d in dihedrals_smarts]

    def calculate_distributions(self, mols, geometry_type, parallel=False):
        assert geometry_type in ['length', 'angle', 'dihedral']
        if geometry_type == 'length':
            obj_list = self.bonds
        elif geometry_type == 'angle':
            obj_list = self.angles
        elif geometry_type == 'dihedral':
            obj_list = self.dihedrals

        results = {}
        for obj in obj_list:
            tmp_results = []
            if not parallel:
                for mol in tqdm(mols):
                    tmp_results.append(calculate_3d_geometry(
                        mol, obj, geometry_type
                    ))
            else:
                with Pool(20) as pool:
                    func = partial(
                        calculate_3d_geometry,
                        obj=obj,
                        geometry_type=geometry_type,
                    )
                    tmp_results = list(tqdm(pool.imap(func, mols), total=len(mols)))
            tmp_results = np.concatenate(tmp_results)
            results[Chem.MolToSmarts(obj)] = tmp_results
        return results

def compare_with_ref(values_list, width=None, num_bins=50, discrete=False):

    all_list = np.concatenate(values_list)
    all_list = all_list[~np.isnan(all_list)]
    all_list_sort = np.sort(all_list)
    if len(all_list_sort) >= 10:
        max_value = all_list_sort[-5]
        min_value = all_list_sort[5]
    else:
        n = len(all_list_sort) // 2
        max_value = all_list_sort[-n]
        min_value = all_list_sort[n]
    if not discrete:
        if width is not None:
            bins = np.arange(min_value, max_value+width, width)
        else:
            bins = np.linspace(min_value, max_value, num_bins)
    else:
        bins = np.arange(min_value, max_value+1.5) - 0.5

    hist_list = []
    for values in values_list:
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        hist_list.append(hist)

    jsd = sci_spatial.distance.jensenshannon(hist_list[-1], hist_list[0])
        
    return jsd

def _format_bond_type(bond_obj):
    for bond in bond_obj.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        bond_type = utils_data.bond_types[bond.GetBondType()]
        if s_sym > e_sym:
            s_sym, e_sym = e_sym, s_sym
    return f'{s_sym}-{e_sym}|{bond_type}'

def _format_angle_type(angle_obj):
    bond_list = []
    for bond in angle_obj.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        bond_type = utils_data.bond_types[bond.GetBondType()]
        bond_list.append([s_sym, e_sym, bond_type])
    return f'{bond_list[0][0]}-{bond_list[0][1]}-{bond_list[1][1]}|{bond_list[0][2]} {bond_list[1][2]}'

def _format_dihedral_type(dihedral_obj):
    bond_list = []
    for bond in dihedral_obj.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        bond_type = utils_data.bond_types[bond.GetBondType()]
        bond_list.append([s_sym, e_sym, bond_type])
    return f'{bond_list[0][0]}-{bond_list[0][1]}-{bond_list[1][1]}-{bond_list[2][1]}|{bond_list[0][2]} {bond_list[1][2]} {bond_list[2][2]}'

def calculate_bond_jsd(mols, refs):
    local3d = LocalGeometryStatistics()
    local3d.define_default_patterns()
    mol_lengths = local3d.calculate_distributions(mols, geometry_type='length')
    ref_lengths = local3d.calculate_distributions(refs, geometry_type='length')
    metrics = {}
    for bond_obj in local3d.bonds:
        bond_smart = Chem.MolToSmarts(bond_obj)
        if len(mol_lengths[bond_smart]) != 0 and len(ref_lengths[bond_smart]) != 0:
            values_list = [mol_lengths[bond_smart], ref_lengths[bond_smart]]
            metrics[f'JSD_{_format_bond_type(bond_obj)}'] = compare_with_ref(values_list, width=0.02, discrete=False)
        else:
            metrics[f'JSD_{_format_bond_type(bond_obj)}'] = None
    return metrics

def calculate_angle_jsd(mols, refs):
    local3d = LocalGeometryStatistics()
    local3d.define_default_patterns()
    mol_angles = local3d.calculate_distributions(mols, geometry_type='angle')
    ref_angles = local3d.calculate_distributions(refs, geometry_type='angle')
    metrics = {}
    for angle_obj in local3d.angles:
        angle_smart = Chem.MolToSmarts(angle_obj)
        if len(mol_angles[angle_smart]) != 0 and len(ref_angles[angle_smart]) != 0:
            values_list = [mol_angles[angle_smart], ref_angles[angle_smart]]
            metrics[f'JSD_{_format_angle_type(angle_obj)}'] = compare_with_ref(values_list, width=5, discrete=False)
        else:
            metrics[f'JSD_{_format_angle_type(angle_obj)}'] = None
    return metrics

def calculate_dihedral_jsd(mols, refs):
    local3d = LocalGeometryStatistics()
    local3d.define_default_patterns()
    mol_dihedrals = local3d.calculate_distributions(
        mols, geometry_type='dihedral'
    )
    ref_dihedrals = local3d.calculate_distributions(
        refs, geometry_type='dihedral'
    )
    metrics = {}
    for dihedral_obj in local3d.dihedrals:
        dihedral_smart = Chem.MolToSmarts(dihedral_obj)
        if len(mol_dihedrals[dihedral_smart]) != 0 and len(ref_dihedrals[dihedral_smart]) != 0:
            values_list = [mol_dihedrals[dihedral_smart], ref_dihedrals[dihedral_smart]]
            metrics[f'JSD_{_format_dihedral_type(dihedral_obj)}'] = compare_with_ref(values_list, width=5, discrete=False)
        else:
            metrics[f'JSD_{_format_dihedral_type(dihedral_obj)}'] = None
    return metrics

def calculate_predicted_rmsd(mols):
    results = []
    for mol in tqdm(mols, desc='Predict RMSD'):
        try:
            rmsd_list = get_rdkit_rmsd(mol)
            results.append(rmsd_list)
        except:
            continue
    return [rmsd for rmsd in results if rmsd]

def calculate_optimized_rmsd(mols):
    all_energy_diff, all_rmsd = [], []
    for mol in tqdm(mols, desc='Optimize RMSD'):
        try:
            result = get_rdkit_optimize_rmsd(mol)
            all_energy_diff.append(result[0])
            all_rmsd.append(result[1])
        except:
            continue
    return all_energy_diff, all_rmsd

def plot_rmsd_violin(gen_rmsd, save_path=None):
    data_dict = {'PADiff': gen_rmsd}
    df = pd.DataFrame(data_dict)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, palette="muted")
    plt.xticks(fontsize=12)
    plt.xlabel('Method', fontsize=16)
    plt.ylabel(r'RMSD ($\AA$)', fontsize=16)

    if save_path is not None:
        plt.savefig(save_path)
        with open(save_path.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        plt.show()
    plt.close()
