from collections import Counter
from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolAlign
from rdkit.Chem.QED import qed
from utils.evaluation.sascorer import compute_sa_score

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rules = (
        Descriptors.ExactMolWt(mol) < 500,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10,
        -2 <= get_logp(mol) <= 5,
        Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10,
    )
    return sum(int(rule) for rule in rules)


def _rmsd_graph_is_pathological(mol, max_cycle_rank=20):
    """Reject highly cyclic graphs that can stall symmetry-aware alignment."""
    num_atoms = int(mol.GetNumAtoms())
    if num_atoms == 0:
        return True
    num_components = len(Chem.GetMolFrags(mol))
    cycle_rank = max(0, int(mol.GetNumBonds()) - num_atoms + num_components)
    return cycle_rank > int(max_cycle_rank)


def get_rdkit_rmsd(
    mol,
    n_conf=20,
    random_seed=42,
    max_cycle_rank=20,
    max_embed_attempts=50,
    max_matches=10000,
):
    """Return the minimum RMSD to a bounded set of RDKit conformers."""
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    if _rmsd_graph_is_pathological(mol, max_cycle_rank=max_cycle_rank):
        return []
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    try:
        conf_ids = AllChem.EmbedMultipleConfs(
            mol3d,
            numConfs=int(n_conf),
            maxAttempts=int(max_embed_attempts),
            randomSeed=int(random_seed),
            numThreads=1,
        )
        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol3d, confId=conf_id, maxIters=200)
            rmsd_list.append(
                Chem.rdMolAlign.GetBestRMS(
                    mol, mol3d, refId=conf_id, maxMatches=int(max_matches)
                )
            )
        return float(np.min(np.asarray(rmsd_list))) if rmsd_list else []
    except Exception:
        return []


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    ring_info = mol.GetRingInfo()
    return {
        "qed": qed(mol),
        "sa": compute_sa_score(mol),
        "logp": get_logp(mol),
        "lipinski": obey_lipinski(mol),
        "tpsa": Chem.rdMolDescriptors.CalcTPSA(mol),
        "ring_size": Counter(len(ring) for ring in ring_info.AtomRings()),
    }


def get_rdkit_optimize_rmsd(
    ori_mol,
    addHs=False,
    enable_torsion=False,
    max_cycle_rank=20,
    max_matches=10000,
):
    """Return energy change and RMSD after bounded MMFF optimization."""
    if _rmsd_graph_is_pathological(ori_mol, max_cycle_rank=max_cycle_rank):
        return []
    mol = deepcopy(ori_mol)
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)
    properties = AllChem.MMFFGetMoleculeProperties(
        mol, mmffVariant="MMFF94s"
    )
    if properties is None:
        return (None,)

    properties.SetMMFFOopTerm(enable_torsion)
    properties.SetMMFFAngleTerm(True)
    properties.SetMMFFTorsionTerm(enable_torsion)
    properties.SetMMFFStretchBendTerm(True)
    properties.SetMMFFBondTerm(True)
    properties.SetMMFFVdWTerm(True)
    properties.SetMMFFEleTerm(True)

    try:
        force_field = AllChem.MMFFGetMoleculeForceField(mol, properties)
        energy_before = force_field.CalcEnergy()
        force_field.Minimize()
        energy_change = energy_before - force_field.CalcEnergy()
        Chem.SanitizeMol(ori_mol)
        Chem.SanitizeMol(mol)
        rmsd = rdMolAlign.GetBestRMS(
            ori_mol, mol, maxMatches=int(max_matches)
        )
    except Exception:
        return []
    return [energy_change, rmsd, mol]
