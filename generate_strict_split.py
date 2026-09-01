from __future__ import annotations
import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm
from datasets.preprocess_crossdocked import PocketLigandPairDataset


def _load_torch(path):
    try:
        return torch.load(str(path), weights_only=False)
    except TypeError:  
        return torch.load(str(path))


def build_split(dataset, blueprint):
    name_to_index = {}
    for index in tqdm(range(len(dataset)), desc="Indexing processed dataset"):
        record = dataset.get_raw_record(index)
        key = str(record.protein_filename) + str(record.ligand_filename)
        name_to_index[key] = index

    selected = {}
    missing = 0
    for split_name in ("train", "val", "test"):
        requested = blueprint.get(split_name, [])
        matched = []
        for filenames in tqdm(requested, desc="Mapping %s" % split_name):
            key = str(filenames[0]) + str(filenames[1])
            if key in name_to_index:
                matched.append(name_to_index[key])
            else:
                missing += 1
        selected[split_name] = matched
    return selected, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="raw_data/crossdocked_v1.1_rmsd1.0_pocket10",
        help="CrossDocked directory containing index.pkl and molecule files.",
    )
    parser.add_argument(
        "--lmdb",
        default="data/crossdocked_v1.1_rmsd1.0_pocket10_processed.lmdb",
        help="Processed PADiff LMDB to index.",
    )
    parser.add_argument(
        "--blueprint",
        required=True,
        help="TargetDiff-compatible torch split containing filename pairs.",
    )
    parser.add_argument(
        "--output",
        default="data/crossdocked_pocket10_pose_split.pt",
        help="Destination for PADiff train/val/test index lists.",
    )
    args = parser.parse_args(argv)

    blueprint_path = Path(args.blueprint)
    if not blueprint_path.is_file():
        parser.error("split blueprint does not exist: %s" % blueprint_path)
    dataset = PocketLigandPairDataset(
        args.raw_root,
        custom_lmdb_path=args.lmdb,
    )
    split, missing = build_split(dataset, _load_torch(blueprint_path))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(split, str(output_path))

    print("Saved split to %s" % output_path)
    print(
        "Matched train=%d val=%d test=%d; missing=%d"
        % (len(split["train"]), len(split["val"]), len(split["test"]), missing)
    )
    return split


if __name__ == "__main__":
    main()
