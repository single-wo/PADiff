from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
import torch
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils.case_study import prepare_case_study


def parse_seeds(value):
    seeds = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return seeds


def _load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _record_molecule(record):
    if record.get("rdmol") is not None:
        return record["rdmol"]
    return (record.get("ligand") or {}).get("rdmol")


def _scaffold_smiles(mol):
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def _valid_seed_manifest(path, seed, expected_count):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        manifest = _load_json(path)
        if int(manifest.get("seed")) != int(seed):
            return None
        record_path = Path(manifest["record_path"])
        if not record_path.is_file():
            return None
        records = torch.load(
            str(record_path), map_location="cpu", weights_only=False
        )
        if len(records) != int(expected_count):
            return None
        return manifest
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def run_seed(args, case_id, seed):
    seed_dir = Path(args.outdir).resolve() / ("seed_%d" % int(seed))
    seed_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = seed_dir / "run_manifest.json"
    existing = _valid_seed_manifest(run_manifest, seed, args.num_mols_per_seed)
    if existing is not None and not args.rerun_complete:
        print("[seed %d] complete; reusing %s" % (seed, existing["record_path"]))
        return existing

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --merge-only is used")
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("case_study_sample.py")),
        "--case", str(Path(args.case).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--base-config", str(Path(args.base_config).resolve()),
        "--outdir", str(seed_dir / "runs"),
        "--device", args.device,
        "--num-mols", str(int(args.num_mols_per_seed)),
        "--batch-size", str(int(args.batch_size)),
        "--seed", str(int(seed)),
        "--run-manifest", str(run_manifest),
    ]
    if args.max_batches is not None:
        command.extend(["--max-batches", str(int(args.max_batches))])
    if args.min_qed is not None:
        command.extend(["--min-qed", str(float(args.min_qed))])
    if args.min_sa is not None:
        command.extend(["--min-sa", str(float(args.min_sa))])
    if args.max_logp is not None:
        command.extend(["--max-logp", str(float(args.max_logp))])
    if args.hard_property_filter is True:
        command.append("--hard-property-filter")
    elif args.hard_property_filter is False:
        command.append("--soft-property-filter")
    print("[seed %d] %s" % (seed, " ".join(command)))
    subprocess.run(command, check=True)
    manifest = _valid_seed_manifest(
        run_manifest, seed, args.num_mols_per_seed
    )
    if manifest is None:
        raise RuntimeError("Seed %d did not produce a valid run manifest" % seed)
    return manifest


def _audit_for_manifest(manifest):
    path = Path(manifest.get("candidate_audit_path", ""))
    return _load_json(path) if path.is_file() else {}


def _sampling_config_path(manifest):

    candidates = []
    saved = manifest.get("saved_sampling_config")
    if saved:
        candidates.append(Path(saved))
    generated = manifest.get("generated_config")
    run_dir = manifest.get("run_dir")
    if generated and run_dir:
        candidates.append(Path(run_dir) / Path(generated).name)
    if generated:
        candidates.append(Path(generated))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "No saved sampling YAML could be located for seed %s; checked: %s"
        % (manifest.get("seed"), ", ".join(str(path) for path in candidates))
    )


def _validate_and_copy_sampling_provenance(
        manifests, merged_dir, expected_per_seed):
    merged_dir = Path(merged_dir).resolve()
    for stale in merged_dir.glob("sampling_seed_*.yml"):
        stale.unlink()

    copied = []
    for manifest in manifests:
        seed = int(manifest["seed"])
        source = _sampling_config_path(manifest)
        with source.open() as handle:
            config = yaml.safe_load(handle) or {}
        sample = config.get("sample", {}) or {}
        protocol = str(sample.get("protocol", "")).lower()
        mismatches = []
        if protocol not in {"adaptive", "direct"}:
            mismatches.append(
                "sample.protocol=%r (expected 'adaptive' or 'direct')"
                % sample.get("protocol")
            )
        if int(sample.get("seed", -1)) != seed:
            mismatches.append(
                "sample.seed=%r (expected %r)" % (sample.get("seed"), seed)
            )
        if int(sample.get("num_mols", -1)) != int(expected_per_seed):
            mismatches.append(
                "sample.num_mols=%r (expected %r)"
                % (sample.get("num_mols"), int(expected_per_seed))
            )
        if mismatches:
            raise ValueError(
                "Invalid sampling provenance for seed %d in %s: %s"
                % (seed, source, "; ".join(mismatches))
            )
        destination = merged_dir / ("sampling_seed_%d.yml" % seed)
        shutil.copy2(str(source), str(destination))
        copied.append({
            "seed": seed,
            "source": str(source),
            "copied_path": str(destination),
        })
    return copied


def merge_seed_runs(manifests, merged_dir, case_id, expected_per_seed):
    merged_dir = Path(merged_dir).resolve()
    case_dir = merged_dir / ("0000_%s_multiseed" % case_id)
    case_dir.mkdir(parents=True, exist_ok=True)

    merged_records = []
    seed_rows = []
    seed_audits = []
    smiles = []
    scaffolds = []
    for seed_index, manifest in enumerate(manifests):
        seed = int(manifest["seed"])
        records = torch.load(
            manifest["record_path"], map_location="cpu", weights_only=False
        )
        if len(records) != int(expected_per_seed):
            raise ValueError(
                "Seed %d has %d records; expected %d"
                % (seed, len(records), int(expected_per_seed))
            )
        for seed_sample_index, record in enumerate(records):
            record = dict(record)
            record["case_study_seed"] = seed
            record["case_study_seed_index"] = int(seed_index)
            record["case_study_seed_sample_index"] = int(seed_sample_index)
            record["case_study_source_record"] = str(manifest["record_path"])
            pooled_index = len(merged_records)
            merged_records.append(record)
            mol = _record_molecule(record)
            canonical = None
            if mol is not None:
                canonical = Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                )
                smiles.append(canonical)
                scaffold = _scaffold_smiles(mol)
                if scaffold:
                    scaffolds.append(scaffold)
            seed_rows.append({
                "pooled_sample_index": pooled_index,
                "seed": seed,
                "seed_index": seed_index,
                "seed_sample_index": seed_sample_index,
                "smiles": canonical,
            })
        seed_audits.append({
            "seed": seed,
            "run_manifest": manifest,
            "candidate_audit": _audit_for_manifest(manifest),
        })

    record_path = case_dir / "joint_samples.pt"
    torch.save(merged_records, str(record_path))
    sdf_path = case_dir / "generated.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for pooled_index, record in enumerate(merged_records):
        mol = _record_molecule(record)
        if mol is None:
            continue
        mol = Chem.Mol(mol)
        mol.SetProp("pooled_sample_index", str(pooled_index))
        mol.SetProp("case_study_seed", str(record["case_study_seed"]))
        mol.SetProp(
            "case_study_seed_sample_index",
            str(record["case_study_seed_sample_index"]),
        )
        writer.write(mol)
    writer.close()

    index_path = merged_dir / "multiseed_record_index.csv"
    with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(seed_rows)

    sampling_provenance = _validate_and_copy_sampling_provenance(
        manifests, merged_dir, expected_per_seed
    )

    raw_total = selected_total = fallback_total = 0
    elapsed_total = 0.0
    per_seed_budget = []
    for item in seed_audits:
        audit = item["candidate_audit"] or {}
        raw = int((audit.get("raw") or {}).get("total") or 0)
        selected = int((audit.get("selected") or {}).get("total") or 0)
        sampling = audit.get("candidate_sampling") or {}
        fallback = int(sampling.get("total_fallback_selected") or 0)
        elapsed = float(sampling.get("total_case_elapsed_seconds") or 0.0)
        raw_total += raw
        selected_total += selected
        fallback_total += fallback
        elapsed_total += elapsed
        per_seed_budget.append({
            "seed": item["seed"],
            "raw_candidates": raw,
            "selected_candidates": selected,
            "strict_selected": max(0, selected - fallback),
            "fallback_selected": fallback,
            "candidate_multiplier": (
                float(raw) / selected if selected else None
            ),
            "sampling_seconds": elapsed,
        })

    manifest = {
        "case_id": case_id,
        "merge_policy": "concatenate_all_selected_records_without_cross_seed_deduplication",
        "num_seeds": len(manifests),
        "seeds": [int(item["seed"]) for item in manifests],
        "molecules_per_seed": int(expected_per_seed),
        "pooled_molecules": len(merged_records),
        "record_path": str(record_path),
        "generated_sdf": str(sdf_path),
        "record_index_csv": str(index_path),
        "sampling_provenance": sampling_provenance,
        "pooled_unique_smiles": len(set(smiles)),
        "pooled_smiles_uniqueness": (
            float(len(set(smiles))) / len(smiles) if smiles else None
        ),
        "pooled_unique_scaffolds": len(set(scaffolds)),
        "pooled_scaffold_bearing_molecules": len(scaffolds),
        "raw_candidates": raw_total,
        "selected_candidates": selected_total,
        "strict_selected": max(0, selected_total - fallback_total),
        "fallback_selected": fallback_total,
        "candidate_multiplier": (
            float(raw_total) / selected_total if selected_total else None
        ),
        "sampling_seconds": elapsed_total,
        "per_seed_budget": per_seed_budget,
        "seed_runs": seed_audits,
    }
    manifest_path = merged_dir / "multiseed_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-config", default="configs/sample.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds", default="2023,2024,2025,2026,2027,2028,2029,2030,2031,2032"
    )
    parser.add_argument("--num-mols-per-seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-batches", type=int, default=10,
        help="Maximum remaining-quota rounds per seed before audited fallback.",
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--min-qed", type=float, default=None)
    parser.add_argument("--min-sa", type=float, default=None)
    parser.add_argument("--max-logp", type=float, default=None)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--hard-property-filter", dest="hard_property_filter",
        action="store_true",
    )
    filter_group.add_argument(
        "--soft-property-filter", dest="hard_property_filter",
        action="store_false",
    )
    parser.set_defaults(hard_property_filter=None)
    parser.add_argument(
        "--force-prepare", action="store_true",
        help="Rebuild the prepared pocket, receptor, and standardized reference ligand.",
    )
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--rerun-complete", action="store_true",
        help="Create a fresh run even when a valid seed manifest already exists.",
    )
    args = parser.parse_args(argv)
    if args.sample_only and args.merge_only:
        parser.error("--sample-only and --merge-only are mutually exclusive")
    if int(args.max_batches) < 1:
        parser.error("--max-batches must be >= 1")

    case_manifest = prepare_case_study(args.case, force=args.force_prepare)
    case_id = case_manifest["id"]
    if args.outdir is None:
        args.outdir = str(
            Path("sample_outputs/case_studies") / (case_id + "_multiseed")
        )
    args.outdir = str(Path(args.outdir).resolve())
    merged_dir = Path(args.merged_dir).resolve() if args.merged_dir else (
        Path(args.outdir) / "merged"
    )
    seeds = parse_seeds(args.seeds)

    manifests = []
    if not args.merge_only:
        for seed in seeds:
            manifests.append(run_seed(args, case_id, seed))
    else:
        for seed in seeds:
            path = Path(args.outdir) / ("seed_%d" % seed) / "run_manifest.json"
            manifest = _valid_seed_manifest(
                path, seed, args.num_mols_per_seed
            )
            if manifest is None:
                raise FileNotFoundError(
                    "No complete seed manifest for seed %d at %s" % (seed, path)
                )
            manifests.append(manifest)

    if args.sample_only:
        print("Sampling complete; merge skipped by --sample-only")
        return {"seed_runs": manifests}
    return merge_seed_runs(
        manifests, merged_dir, case_id, args.num_mols_per_seed
    )


if __name__ == "__main__":
    main()
