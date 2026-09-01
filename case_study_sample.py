from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
import sample
from utils.case_study import prepare_case_study

def _single_record_path(run_dir, record_filename="joint_samples.pt"):
    paths = sorted(Path(run_dir).glob("**/%s" % record_filename))
    if len(paths) != 1:
        raise ValueError(
            "Expected exactly one %s below %s; found %d"
            % (record_filename, run_dir, len(paths))
        )
    return paths[0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Case-study YAML file.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--base-config", default="configs/sample.yml")
    parser.add_argument("--outdir", default="sample_outputs/case_studies")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-mols", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override sample.seed for an independent case-study replicate.",
    )
    parser.add_argument(
        "--run-manifest", default=None,
        help="Optional JSON path recording the exact output directory and seed.",
    )
    parser.add_argument("--min-qed", type=float, default=None)
    parser.add_argument("--min-sa", type=float, default=None)
    parser.add_argument("--max-logp", type=float, default=None)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help=(
            "Override sample.max_batches for the adaptive protocol. "
            "Each quota round generates only the remaining strict-feasible count."
        ),
    )
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
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    manifest = prepare_case_study(args.case, force=args.force_prepare)
    print("Prepared pocket :", manifest["pocket_path"])
    print("Dock receptor  :", manifest["receptor_path"])
    print("Reference SDF  :", manifest["reference_ligand_path"])
    print("Pocket residues:", manifest["pocket_residues"])
    print("Docking center :", manifest["docking_center"])
    print("Docking box    :", manifest["docking_box_size"])
    if args.prepare_only:
        return {"case_manifest": manifest}
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --prepare-only is used")

    with open(args.base_config) as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("model", {})["target"] = manifest["pocket_path"]
    config["model"]["checkpoint"] = os.path.abspath(args.checkpoint)
    config.setdefault("sample", {})["mode"] = "pocket"
    config["sample"]["max_cases"] = 1
    config["sample"]["num_mols"] = int(args.num_mols)
    if args.max_batches is not None:
        if int(args.max_batches) < 1:
            parser.error("--max-batches must be >= 1")
        if config["sample"].get("protocol") != "adaptive":
            parser.error("--max-batches is only valid for protocol: adaptive")
        config["sample"]["max_batches"] = int(args.max_batches)
    if args.seed is not None:
        config["sample"]["seed"] = int(args.seed)

    prepared_dir = os.path.dirname(manifest["pocket_path"])
    seed_suffix = "_seed%d" % int(config["sample"].get("seed", 2023))
    generated_config = os.path.join(prepared_dir, "sample_case%s.yml" % seed_suffix)
    with open(generated_config, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    run_result = sample.main(SimpleNamespace(
        config=generated_config,
        outdir=args.outdir,
        device=args.device,
        checkpoint=os.path.abspath(args.checkpoint),
        batch_size=int(args.batch_size),
        num_mols=int(args.num_mols),
        max_cases=1,
        min_qed=args.min_qed,
        min_sa=args.min_sa,
        max_logp=args.max_logp,
        hard_property_filter=args.hard_property_filter,
    ))
    run_dir = Path(run_result["run_dir"]).resolve()
    record_path = _single_record_path(run_dir)
    records = torch.load(str(record_path), map_location="cpu", weights_only=False)
    saved_sampling_config = run_dir / Path(generated_config).name
    if not saved_sampling_config.is_file():
        raise FileNotFoundError(
            "Sampling run did not preserve its YAML config at %s"
            % saved_sampling_config
        )
    output = {
        "case_id": manifest["id"],
        "case_config": str(Path(args.case).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": int(config["sample"].get("seed", 2023)),
        "requested_molecules": int(args.num_mols),
        "saved_molecules": int(len(records)),
        "run_dir": str(run_dir),
        "record_path": str(record_path.resolve()),
        "candidate_audit_path": str((run_dir / "candidate_audit.json").resolve()),
        "generated_config": str(Path(generated_config).resolve()),
        "saved_sampling_config": str(saved_sampling_config.resolve()),
        "resumed": bool(run_result.get("resumed", False)),
    }
    manifest_path = Path(args.run_manifest).resolve() if args.run_manifest else (
        run_dir / "case_study_run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    output["run_manifest"] = str(manifest_path)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    main()
