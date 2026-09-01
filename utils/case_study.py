from __future__ import annotations
import json
import os
import string
from functools import lru_cache
from pathlib import Path
import numpy as np
import yaml
from Bio.PDB import MMCIFParser, PDBParser, is_aa
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem
from rdkit.Chem import FilterCatalog

_CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def load_case_config(path):
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def resolve_case_path(config, value):
    if value is None:
        return None
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return str(path.resolve())


def _load_structure(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure("case", path)
    if suffix in {".pdb", ".ent"}:
        return PDBParser(QUIET=True).get_structure("case", path)
    raise ValueError("Structure must be PDB or mmCIF: %s" % path)


def _atom_element(atom):
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element
    return "".join(char for char in atom.name if char.isalpha())[:1].upper()


def _heavy_atoms(residue):
    return [atom for atom in residue.get_atoms() if _atom_element(atom) != "H"]


def _find_ligand(model, resname, chain_id=None):
    matches = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            if residue.resname.strip().upper() == resname.upper():
                matches.append((chain, residue))
    if not matches:
        available = sorted({
            residue.resname.strip() for chain in model for residue in chain
            if residue.id[0] != " " and residue.resname.strip() not in {"HOH", "WAT"}
        })
        raise ValueError(
            "Ligand %s%s was not found; non-water components are: %s"
            % (resname, " in chain %s" % chain_id if chain_id else "", available)
        )
    if len(matches) > 1:
        labels = ["%s:%s:%s" % (chain.id, residue.resname, residue.id[1])
                  for chain, residue in matches]
        raise ValueError(
            "Ligand selection is ambiguous (%s); set ligand.chain explicitly"
            % ", ".join(labels)
        )
    return matches[0]


def _protein_residues(model):
    for chain in model:
        for residue in chain:
            if residue.id[0] == " " and is_aa(residue, standard=False):
                yield chain, residue


def _selected_atom(atom):
    if not atom.is_disordered() or not hasattr(atom, "disordered_get_list"):
        return atom
    choices = list(atom.disordered_get_list())
    return max(
        choices,
        key=lambda item: (
            float(item.get_occupancy() or 0.0),
            item.get_altloc() in {" ", "A"},
        ),
    )


def _format_pdb_atom(serial, atom, residue, chain_id):
    atom = _selected_atom(atom)
    name = atom.get_name().strip()
    element = _atom_element(atom)
    if len(name) < 4 and (not name or not name[0].isdigit()):
        name = " " + name
    name = name[:4].ljust(4)
    coord = atom.coord
    occupancy = float(atom.get_occupancy() or 1.0)
    bfactor = float(atom.get_bfactor() or 0.0)
    resseq = int(residue.id[1])
    icode = str(residue.id[2] or " ")[:1]
    return (
        "ATOM  {serial:5d} {name}{altloc:1s}{resname:>3s} {chain:1s}"
        "{resseq:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        "{occ:6.2f}{bfac:6.2f}          {element:>2s}\n"
    ).format(
        serial=serial,
        name=name,
        altloc=" ",
        resname=residue.resname.strip()[:3],
        chain=chain_id,
        resseq=resseq,
        icode=icode,
        x=float(coord[0]),
        y=float(coord[1]),
        z=float(coord[2]),
        occ=occupancy,
        bfac=bfactor,
        element=element[:2],
    )


def write_protein_pdb(path, residues):
    residues = list(residues)
    source_chain_ids = []
    for chain, _ in residues:
        if chain.id not in source_chain_ids:
            source_chain_ids.append(chain.id)
    if len(source_chain_ids) > len(_CHAIN_IDS):
        raise ValueError("Too many protein chains for PDB output")
    chain_map = {
        source: _CHAIN_IDS[index] for index, source in enumerate(source_chain_ids)
    }

    serial = 1
    previous_chain = None
    with open(path, "w") as handle:
        for chain, residue in residues:
            output_chain = chain_map[chain.id]
            if previous_chain is not None and output_chain != previous_chain:
                handle.write("TER\n")
            for atom in residue.get_atoms():
                atom = _selected_atom(atom)
                if _atom_element(atom) == "H":
                    continue
                handle.write(_format_pdb_atom(serial, atom, residue, output_chain))
                serial += 1
            previous_chain = output_chain
        handle.write("TER\nEND\n")
    return chain_map


def _as_list(value):
    return value if isinstance(value, list) else [value]


def ligand_from_component_cif(component_cif, ligand_residue):
    data = MMCIF2Dict(component_cif)
    atom_ids = _as_list(data["_chem_comp_atom.atom_id"])
    elements = _as_list(data["_chem_comp_atom.type_symbol"])
    charges = _as_list(data.get("_chem_comp_atom.charge", ["0"] * len(atom_ids)))
    aromatic_atoms = _as_list(data.get(
        "_chem_comp_atom.pdbx_aromatic_flag", ["N"] * len(atom_ids)
    ))

    experimental = {
        atom.get_name().strip(): atom.coord
        for atom in ligand_residue.get_atoms()
        if _atom_element(atom) != "H"
    }
    rw = Chem.RWMol()
    index_by_name = {}
    for atom_id, element, charge, aromatic in zip(
            atom_ids, elements, charges, aromatic_atoms):
        if str(element).upper() == "H":
            continue
        atom = Chem.Atom(str(element))
        atom.SetFormalCharge(int(float(charge)))
        atom.SetIsAromatic(str(aromatic).upper() == "Y")
        index_by_name[str(atom_id)] = rw.AddAtom(atom)

    bond_a = _as_list(data["_chem_comp_bond.atom_id_1"])
    bond_b = _as_list(data["_chem_comp_bond.atom_id_2"])
    orders = _as_list(data["_chem_comp_bond.value_order"])
    aromatic_bonds = _as_list(data.get(
        "_chem_comp_bond.pdbx_aromatic_flag", ["N"] * len(bond_a)
    ))
    order_map = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "QUAD": Chem.BondType.QUADRUPLE,
    }
    for first, second, order, aromatic in zip(
            bond_a, bond_b, orders, aromatic_bonds):
        if first not in index_by_name or second not in index_by_name:
            continue
        bond_type = (
            Chem.BondType.AROMATIC if str(aromatic).upper() == "Y"
            else order_map.get(str(order).upper(), Chem.BondType.SINGLE)
        )
        rw.AddBond(index_by_name[first], index_by_name[second], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            rw.GetAtomWithIdx(index_by_name[first]).SetIsAromatic(True)
            rw.GetAtomWithIdx(index_by_name[second]).SetIsAromatic(True)

    mol = rw.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    missing = []
    for atom_id, atom_index in index_by_name.items():
        position = experimental.get(atom_id)
        if position is None:
            missing.append(atom_id)
            continue
        conformer.SetAtomPosition(atom_index, tuple(float(value) for value in position))
    if missing:
        raise ValueError(
            "Experimental ligand is missing component atoms: %s" % ", ".join(missing)
        )
    mol.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(mol)
    mol.SetProp("_Name", ligand_residue.resname.strip())
    return mol


def standardize_reference_ligand(mol, options=None):
    options = options or {}
    standardized = Chem.RWMol(mol)
    audit = {
        "deprotonated_sulfonic_acids": 0,
        "formal_charge_before": int(Chem.GetFormalCharge(mol)),
    }
    if bool(options.get("deprotonate_sulfonic_acids", False)):
        pattern = Chem.MolFromSmarts("[S;X4](=[O;X1])(=[O;X1])[O;H1]")
        matches = standardized.GetSubstructMatches(pattern)
        oxygen_indices = sorted({int(match[3]) for match in matches})
        for atom_index in oxygen_indices:
            atom = standardized.GetAtomWithIdx(atom_index)
            atom.SetFormalCharge(-1)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
        audit["deprotonated_sulfonic_acids"] = len(oxygen_indices)
        if not oxygen_indices:
            raise ValueError(
                "reference_standardization requested sulfonic-acid "
                "deprotonation, but no sulfonic acid was found"
            )
    standardized = standardized.GetMol()
    Chem.SanitizeMol(standardized)
    audit["formal_charge_after"] = int(Chem.GetFormalCharge(standardized))
    expected_charge = options.get("expected_formal_charge")
    if expected_charge is not None and audit["formal_charge_after"] != int(expected_charge):
        raise ValueError(
            "Standardized reference formal charge is %d; expected %d"
            % (audit["formal_charge_after"], int(expected_charge))
        )
    standardized.SetProp("reference_standardization", json.dumps(audit, sort_keys=True))
    return standardized, audit


def _distance_to_ligand(residue, ligand_positions):
    atoms = _heavy_atoms(residue)
    if not atoms:
        return float("inf")
    positions = np.asarray([atom.coord for atom in atoms], dtype=np.float64)
    delta = positions[:, None, :] - ligand_positions[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=-1)).min())


def prepare_case_study(case_config_path, force=False):
    config = load_case_config(case_config_path)
    case_id = str(config.get("id") or Path(case_config_path).stem)
    structure_path = resolve_case_path(config, config["structure"])
    component_cif = resolve_case_path(config, config.get("component_cif"))
    output_dir = Path(resolve_case_path(
        config, config.get("prepared_dir", "prepared")
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    pocket_path = output_dir / (case_id + "_pocket.pdb")
    receptor_path = output_dir / (case_id + "_receptor.pdb")
    reference_path = output_dir / (case_id + "_reference.sdf")
    manifest_path = output_dir / "manifest.json"
    if (not force and pocket_path.is_file() and receptor_path.is_file()
            and reference_path.is_file() and manifest_path.is_file()):
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        manifest.update({
            "source_structure": str(Path(structure_path).resolve()),
            "source_component_cif": (
                str(Path(component_cif).resolve()) if component_cif else None
            ),
            "pocket_path": str(pocket_path.resolve()),
            "receptor_path": str(receptor_path.resolve()),
            "reference_ligand_path": str(reference_path.resolve()),
        })
        return manifest

    structure = _load_structure(structure_path)
    model = next(structure.get_models())
    ligand_cfg = config.get("ligand", {}) or {}
    ligand_resname = str(ligand_cfg["resname"])
    ligand_chain = ligand_cfg.get("chain")
    source_ligand_chain, ligand_residue = _find_ligand(
        model, ligand_resname, ligand_chain
    )
    ligand_atoms = _heavy_atoms(ligand_residue)
    if not ligand_atoms:
        raise ValueError("Selected ligand has no heavy atoms")
    ligand_positions = np.asarray(
        [atom.coord for atom in ligand_atoms], dtype=np.float64
    )

    radius = float(config.get("pocket_radius", 10.0))
    protein_residues = list(_protein_residues(model))
    pocket_residues = []
    contacts = []
    for chain, residue in protein_residues:
        distance = _distance_to_ligand(residue, ligand_positions)
        if distance <= radius:
            pocket_residues.append((chain, residue))
        if distance <= 6.0:
            contacts.append({
                "source_chain": chain.id,
                "resname": residue.resname.strip(),
                "resseq": int(residue.id[1]),
                "distance": distance,
            })
    if not pocket_residues:
        raise ValueError("No protein residue was found within %.2f Å" % radius)

    receptor_chain_map = write_protein_pdb(receptor_path, protein_residues)
    pocket_chain_map = write_protein_pdb(pocket_path, pocket_residues)

    if component_cif is None:
        raise ValueError("component_cif is required to preserve ligand bond orders")
    reference_mol = ligand_from_component_cif(component_cif, ligand_residue)
    reference_mol, reference_standardization = standardize_reference_ligand(
        reference_mol, config.get("reference_standardization")
    )
    writer = Chem.SDWriter(str(reference_path))
    writer.write(reference_mol)
    writer.close()

    extent = ligand_positions.max(axis=0) - ligand_positions.min(axis=0)
    center = (ligand_positions.max(axis=0) + ligand_positions.min(axis=0)) / 2.0
    box_padding = float(config.get("docking_box_padding", 12.0))
    minimum_box = float(config.get("minimum_docking_box_size", 20.0))
    box_size = np.maximum(extent + box_padding, minimum_box)

    mapped_contacts = []
    for contact in sorted(contacts, key=lambda item: item["distance"]):
        mapped = dict(contact)
        source_chain = mapped.pop("source_chain")
        mapped["chain"] = pocket_chain_map[source_chain]
        mapped_contacts.append(mapped)

    manifest = {
        "id": case_id,
        "source_structure": structure_path,
        "source_component_cif": component_cif,
        "source_ligand": {
            "resname": ligand_resname,
            "chain": source_ligand_chain.id,
            "resseq": int(ligand_residue.id[1]),
        },
        "pocket_radius": radius,
        "pocket_residues": len(pocket_residues),
        "pocket_atoms": int(sum(len(_heavy_atoms(residue)) for _, residue in pocket_residues)),
        "protein_chains": receptor_chain_map,
        "pocket_chains": pocket_chain_map,
        "pocket_path": str(pocket_path.resolve()),
        "receptor_path": str(receptor_path.resolve()),
        "reference_ligand_path": str(reference_path.resolve()),
        "reference_standardization": reference_standardization,
        "docking_center": [float(value) for value in center],
        "docking_box_size": [float(value) for value in box_size],
        "contacts_within_6A": mapped_contacts,
        "hotspots": config.get("hotspots", []),
    }
    persisted_manifest = dict(manifest)
    for key in (
        "source_structure", "source_component_cif", "pocket_path",
        "receptor_path", "reference_ligand_path",
    ):
        value = persisted_manifest.get(key)
        if value:
            persisted_manifest[key] = os.path.relpath(value, output_dir)
    with manifest_path.open("w") as handle:
        json.dump(persisted_manifest, handle, indent=2, sort_keys=True)
    return manifest


@lru_cache(maxsize=16)
def receptor_residue_positions(receptor_path):
    structure = _load_structure(str(receptor_path))
    model = next(structure.get_models())
    positions = {}
    for chain in model:
        for residue in chain:
            atoms = _heavy_atoms(residue)
            if not atoms:
                continue
            positions[(str(chain.id), int(residue.id[1]))] = np.asarray(
                [atom.coord for atom in atoms], dtype=np.float64
            )
    return positions


def evaluate_hotspot_recovery(mol, receptor_path, hotspots):
    from utils.phore_realization import extract_molecule_pharmacophores

    features = extract_molecule_pharmacophores(mol)
    residues = receptor_residue_positions(receptor_path)
    details = []
    recovered_count = 0
    for hotspot in hotspots or []:
        residue_coordinates = []
        missing_residues = []
        for label in hotspot.get("residues", []):
            chain, residue_number = str(label).split(":", 1)
            key = (chain, int(residue_number))
            if key not in residues:
                missing_residues.append(str(label))
                continue
            residue_coordinates.append(residues[key])
        allowed_types = {
            int(value) for value in hotspot.get("allowed_phore_types", [])
        }
        candidate_features = [
            feature for feature in features
            if int(feature["type"]) in allowed_types
        ]
        minimum_distance = None
        if residue_coordinates and candidate_features:
            target = np.concatenate(residue_coordinates, axis=0)
            minimum_distance = min(
                float(np.linalg.norm(
                    target - np.asarray(feature["pos"], dtype=np.float64), axis=1
                ).min())
                for feature in candidate_features
            )
        threshold = float(hotspot.get("distance_threshold", 5.0))
        recovered = bool(
            minimum_distance is not None and minimum_distance <= threshold
        )
        recovered_count += int(recovered)
        details.append({
            "name": str(hotspot["name"]),
            "recovered": int(recovered),
            "minimum_distance": minimum_distance,
            "distance_threshold": threshold,
            "allowed_phore_types": sorted(allowed_types),
            "residues": [str(value) for value in hotspot.get("residues", [])],
            "missing_residues": missing_residues,
        })

    total = len(details)
    return {
        "hotspot_count": total,
        "hotspot_recovered_count": recovered_count,
        "hotspot_recovery_ratio": (
            float(recovered_count) / total if total else None
        ),
        "all_hotspots_recovered": int(total > 0 and recovered_count == total),
        "any_hotspot_recovered": int(recovered_count > 0),
        "hotspot_details": details,
    }

def evaluate_generated_phore_hotspot_recovery(
    phore_positions, phore_types, receptor_path, hotspots
):

    if hasattr(phore_positions, "detach"):
        phore_positions = phore_positions.detach().cpu().numpy()
    if hasattr(phore_types, "detach"):
        phore_types = phore_types.detach().cpu().numpy()
    positions = np.asarray(
        phore_positions if phore_positions is not None else [], dtype=np.float64
    )
    types = np.asarray(
        phore_types if phore_types is not None else [], dtype=np.int64
    ).reshape(-1)
    if positions.size == 0:
        positions = np.empty((0, 3), dtype=np.float64)
    positions = positions.reshape((-1, 3))
    if positions.shape[0] != types.shape[0]:
        raise ValueError(
            "Generated pharmacophore position/type counts differ: %d vs %d"
            % (positions.shape[0], types.shape[0])
        )

    residues = receptor_residue_positions(str(receptor_path))
    details = []
    recovered_count = 0
    for hotspot in hotspots or []:
        residue_coordinates = []
        missing_residues = []
        for label in hotspot.get("residues", []):
            chain, residue_number = str(label).split(":", 1)
            key = (chain, int(residue_number))
            if key not in residues:
                missing_residues.append(str(label))
                continue
            residue_coordinates.append(residues[key])
        allowed_types = {
            int(value) for value in hotspot.get("allowed_phore_types", [])
        }
        candidate_positions = positions[
            np.asarray([int(value) in allowed_types for value in types], dtype=bool)
        ]
        minimum_distance = None
        if residue_coordinates and candidate_positions.size:
            target = np.concatenate(residue_coordinates, axis=0)
            minimum_distance = float(np.linalg.norm(
                candidate_positions[:, None, :] - target[None, :, :], axis=2
            ).min())
        threshold = float(hotspot.get("distance_threshold", 5.0))
        recovered = bool(
            minimum_distance is not None and minimum_distance <= threshold
        )
        recovered_count += int(recovered)
        details.append({
            "name": str(hotspot["name"]),
            "recovered": int(recovered),
            "minimum_distance": minimum_distance,
            "distance_threshold": threshold,
            "allowed_phore_types": sorted(allowed_types),
            "residues": [str(value) for value in hotspot.get("residues", [])],
            "missing_residues": missing_residues,
        })
    total = len(details)
    return {
        "hotspot_count": total,
        "hotspot_recovered_count": recovered_count,
        "hotspot_recovery_ratio": (
            float(recovered_count) / total if total else None
        ),
        "all_hotspots_recovered": int(total > 0 and recovered_count == total),
        "any_hotspot_recovered": int(recovered_count > 0),
        "hotspot_details": details,
    }


@lru_cache(maxsize=1)
def _case_study_filter_catalog():
    params = FilterCatalog.FilterCatalogParams()
    for catalog in (
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    ):
        params.AddCatalog(catalog)
    return FilterCatalog.FilterCatalog(params)


def chemical_quality_flags(mol):
    if mol is None:
        return {
            "catalog_alert_count": None,
            "catalog_alerts": [],
            "has_catalog_alert": None,
            "has_peroxide": None,
            "has_three_membered_ring": None,
            "has_four_membered_ring": None,
            "alert_free": None,
        }
    matches = _case_study_filter_catalog().GetMatches(mol)
    descriptions = sorted({match.GetDescription() for match in matches})
    peroxide = Chem.MolFromSmarts("[O;X1,X2]-[O;X1,X2]")
    rings = mol.GetRingInfo().AtomRings()
    has_peroxide = bool(peroxide is not None and mol.HasSubstructMatch(peroxide))
    has_three = any(len(ring) == 3 for ring in rings)
    has_four = any(len(ring) == 4 for ring in rings)
    return {
        "catalog_alert_count": int(len(descriptions)),
        "catalog_alerts": descriptions,
        "has_catalog_alert": int(bool(descriptions)),
        "has_peroxide": int(has_peroxide),
        "has_three_membered_ring": int(has_three),
        "has_four_membered_ring": int(has_four),
        "alert_free": int(
            not descriptions and not has_peroxide and not has_three and not has_four
        ),
    }


@lru_cache(maxsize=16)
def receptor_atom_data(receptor_path):
    structure = _load_structure(str(receptor_path))
    model = next(structure.get_models())
    positions = []
    elements = []
    for chain in model:
        for residue in chain:
            for atom in _heavy_atoms(residue):
                positions.append(np.asarray(atom.coord, dtype=np.float64))
                elements.append(_atom_element(atom).title())
    return np.asarray(positions, dtype=np.float64), tuple(elements)


def evaluate_pose_clashes(mol, receptor_path, vdw_scale=0.75,
                           severe_distance=1.0):
    if mol is None or mol.GetNumConformers() == 0:
        return {
            "minimum_protein_distance": None,
            "severe_clash_count": None,
            "vdw_clash_count": None,
            "has_severe_clash": None,
            "has_vdw_clash": None,
        }
    protein_positions, protein_elements = receptor_atom_data(str(receptor_path))
    conformer = mol.GetConformer(0)
    ligand_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if not ligand_indices or protein_positions.size == 0:
        return {
            "minimum_protein_distance": None,
            "severe_clash_count": 0,
            "vdw_clash_count": 0,
            "has_severe_clash": 0,
            "has_vdw_clash": 0,
        }
    ligand_positions = np.asarray([
        conformer.GetAtomPosition(index) for index in ligand_indices
    ], dtype=np.float64)
    distances = np.linalg.norm(
        ligand_positions[:, None, :] - protein_positions[None, :, :], axis=2
    )
    severe_count = int(np.sum(distances < float(severe_distance)))
    periodic_table = Chem.GetPeriodicTable()
    protein_radii = np.asarray([
        periodic_table.GetRvdw(element) for element in protein_elements
    ], dtype=np.float64)
    ligand_radii = np.asarray([
        periodic_table.GetRvdw(mol.GetAtomWithIdx(index).GetSymbol())
        for index in ligand_indices
    ], dtype=np.float64)
    thresholds = float(vdw_scale) * (
        ligand_radii[:, None] + protein_radii[None, :]
    )
    vdw_count = int(np.sum(distances < thresholds))
    return {
        "minimum_protein_distance": float(distances.min()),
        "severe_clash_count": severe_count,
        "vdw_clash_count": vdw_count,
        "has_severe_clash": int(severe_count > 0),
        "has_vdw_clash": int(vdw_count > 0),
    }
