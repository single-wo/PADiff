from collections import Counter, defaultdict
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy import spatial

AA_DIM = 21
DEFAULT_POCKET_SIZE_BOUNDS = (
    26.52218454781356,
    27.875010450039014,
    28.827148928234973,
    29.653040886224233,
    30.439636480866106,
    31.234028070512103,
    32.014619850535716,
    33.035326071776495,
    34.57565983834558,
)
HYDROPHOBIC_AA = {0, 1, 4, 7, 9, 10, 12, 17, 18, 19}
POCKET_DESCRIPTOR_NAMES = [
    "pocket_size", "bbox_volume", "convex_hull_volume",
    "available_space_proxy", "radius_of_gyration", "protein_atom_count",
    "atom_density", "carbon_fraction", "polar_atom_fraction",
    "hydrophobic_aa_atom_fraction", "polar_site_proxy_count",
    "hydrophobic_site_proxy_count",
] + ["aa_atom_fraction_%02d" % index for index in range(AA_DIM)]


def _numpy(value, dtype=None):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def compute_pocket_descriptors(data):
    positions = _numpy(getattr(data, "protein_pos"), np.float64)
    elements = _numpy(getattr(data, "protein_element", None), np.int64)
    aa_type = _numpy(getattr(data, "protein_atom_to_aa_type", None), np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("Pocket descriptors require non-empty protein_pos [N,3]")
    n_atoms = len(positions)
    if elements is None or elements.shape != (n_atoms,):
        elements = np.zeros(n_atoms, dtype=np.int64)
    if aa_type is None or aa_type.shape != (n_atoms,):
        aa_type = np.full(n_atoms, AA_DIM - 1, dtype=np.int64)

    centered = positions - positions.mean(axis=0, keepdims=True)
    extent = np.maximum(np.ptp(positions, axis=0), 1e-3)
    bbox_volume = float(np.prod(extent))
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    hull_volume = bbox_volume
    if n_atoms >= 4:
        try:
            hull_volume = float(spatial.ConvexHull(positions).volume)
        except Exception:
            hull_volume = bbox_volume
    hull_volume = max(hull_volume, 1e-3)

    vdw_radius = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80, 34: 1.90}
    occupied = sum(
        (4.0 / 3.0) * np.pi * vdw_radius.get(int(z), 1.70) ** 3
        for z in elements
    ) * 0.35
    available_space = max(hull_volume - occupied, 0.0)

    aa_valid = (aa_type >= 0) & (aa_type < AA_DIM)
    aa_counts = np.bincount(aa_type[aa_valid], minlength=AA_DIM).astype(np.float64)
    aa_fraction = aa_counts / max(float(aa_counts.sum()), 1.0)
    hydrophobic_mask = np.isin(aa_type, list(HYDROPHOBIC_AA))
    polar_mask = np.isin(elements, [7, 8, 16, 34])
    carbon_mask = elements == 6
    pocket_size = _pocket_size_from_numpy(positions)
    atom_density = float(n_atoms / hull_volume)
    descriptor = [
        pocket_size,
        bbox_volume,
        hull_volume,
        available_space,
        radius_of_gyration,
        float(n_atoms),
        atom_density,
        float(carbon_mask.mean()),
        float(polar_mask.mean()),
        float(hydrophobic_mask.mean()),
        float(polar_mask.sum()),
        float((hydrophobic_mask & carbon_mask).sum()),
    ]
    descriptor.extend(aa_fraction.tolist())
    return np.asarray(descriptor, dtype=np.float64)


def _pocket_size_from_numpy(positions):
    if len(positions) < 2:
        return 10.0
    distances = spatial.distance.pdist(positions, metric="euclidean")
    top_k = min(10, len(distances))
    largest = np.partition(distances, len(distances) - top_k)[-top_k:]
    return float(np.median(largest))


def pocket_size_bin(pocket_size, bounds=None):
    """Map a pocket-size scalar to a monotonically defined bin."""
    bounds = DEFAULT_POCKET_SIZE_BOUNDS if bounds is None else bounds
    return int(
        next((i for i, bound in enumerate(bounds) if bound > pocket_size), len(bounds))
    )


def atom_count_bin(atom_count, width=4):
    """Round an atom count to a coarse bucket for the conditional phore prior."""
    if int(width) <= 0:
        raise ValueError("atom_bucket_width must be positive")
    return int(max(0, round(float(atom_count) / width) * width))


def _distribution(counter):
    if not counter:
        return None
    values = sorted(int(key) for key in counter)
    total = float(sum(counter.values()))
    return {
        "values": values,
        "probabilities": [float(counter[value]) / total for value in values],
    }


def _iter_training_records(train_set):
    base_dataset = getattr(train_set, "dataset", None)
    indices = getattr(train_set, "indices", None)
    if base_dataset is not None and indices is not None and hasattr(
        base_dataset, "get_raw_record"
    ):
        for index in indices:
            yield base_dataset.get_raw_record(int(index))
        return
    for index in range(len(train_set)):
        yield train_set[index]


def _pocket_size(pocket_pos):
    return _pocket_size_from_numpy(_numpy(pocket_pos, np.float64))


def _fit_pocket_bounds(pocket_sizes, num_bins=10):
    if num_bins < 1:
        raise ValueError("num_pocket_bins must be at least 1")
    if num_bins == 1:
        return []
    quantiles = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)[1:-1]
    bounds = np.quantile(np.asarray(pocket_sizes, dtype=np.float64), quantiles)
    return [float(value) for value in np.unique(bounds)]


def build_cardinality_prior(
    train_set, atom_bucket_width=4, num_pocket_bins=10, show_progress=False,
):

    records = []
    iterator = _iter_training_records(train_set)
    if show_progress:
        iterator = tqdm(
            iterator, total=len(train_set), desc="Fit train-only cardinality prior",
            unit="complex", dynamic_ncols=True,
        )
    for data in iterator:
        descriptor = compute_pocket_descriptors(data)
        records.append((
            descriptor,
            int(getattr(data, "ligand_element").numel()),
            int(getattr(data, "phore_type").numel()),
        ))
    if not records:
        raise ValueError("Cannot fit cardinality priors from an empty training split")

    descriptor_matrix = np.stack([record[0] for record in records], axis=0)
    descriptor_mean = descriptor_matrix.mean(axis=0)
    descriptor_scale = descriptor_matrix.std(axis=0)
    descriptor_scale[descriptor_scale < 1e-6] = 1.0
    normalized = (descriptor_matrix - descriptor_mean) / descriptor_scale
    pocket_sizes = descriptor_matrix[:, 0]
    pocket_bounds = _fit_pocket_bounds(pocket_sizes, num_bins=num_pocket_bins)

    phore_conditional = defaultdict(Counter)
    phore_by_pocket = defaultdict(Counter)
    phore_by_atom = defaultdict(Counter)
    phore_global = Counter()
    atom_by_pocket = defaultdict(Counter)
    atom_global = Counter()
    for descriptor, atom_count, phore_count in records:
        pbin = pocket_size_bin(descriptor[0], pocket_bounds)
        abin = atom_count_bin(atom_count, atom_bucket_width)
        phore_conditional[f"{pbin}:{abin}"][phore_count] += 1
        phore_by_pocket[pbin][phore_count] += 1
        phore_by_atom[abin][phore_count] += 1
        phore_global[phore_count] += 1
        atom_by_pocket[pbin][atom_count] += 1
        atom_global[atom_count] += 1

    return {
        "version": 3,
        "num_training_records": len(records),
        "atom_bucket_width": int(atom_bucket_width),
        "descriptor_names": list(POCKET_DESCRIPTOR_NAMES),
        "descriptor_mean": descriptor_mean.tolist(),
        "descriptor_scale": descriptor_scale.tolist(),
        "training_descriptors": normalized.astype(np.float32),
        "training_atom_counts": np.asarray([r[1] for r in records], dtype=np.int16),
        "training_phore_counts": np.asarray([r[2] for r in records], dtype=np.int16),
        "neighbor_count": int(min(64, max(8, round(np.sqrt(len(records)))))),
        "pocket_bounds": pocket_bounds,
        "atom_by_pocket": {
            str(key): _distribution(value) for key, value in atom_by_pocket.items()
        },
        "atom_global": _distribution(atom_global),
        "conditional": {
            key: _distribution(value) for key, value in phore_conditional.items()
        },
        "by_pocket": {
            str(key): _distribution(value) for key, value in phore_by_pocket.items()
        },
        "by_atom": {
            str(key): _distribution(value) for key, value in phore_by_atom.items()
        },
        "global": _distribution(phore_global),
    }


def _file_signature(path):
    absolute = os.path.abspath(os.path.expanduser(path))
    try:
        stat = os.stat(absolute)
        return {
            "path": absolute,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {"path": absolute, "size": None, "mtime_ns": None}


def load_or_build_cardinality_prior(
    train_set, cache_path, data_path, split_path, logger=None,
    atom_bucket_width=4, num_pocket_bins=10,
):
    cache_path = os.path.abspath(os.path.expanduser(cache_path))
    expected_metadata = {
        "cache_version": 2,
        "prior_version": 3,
        "train_size": int(len(train_set)),
        "atom_bucket_width": int(atom_bucket_width),
        "num_pocket_bins": int(num_pocket_bins),
        "data": _file_signature(data_path),
        "split": _file_signature(split_path),
    }
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)
            if (
                isinstance(cached, dict)
                and cached.get("metadata") == expected_metadata
                and isinstance(cached.get("prior"), dict)
                and cached["prior"].get("version") == 3
            ):
                if logger is not None:
                    logger.info("Loaded train-only cardinality prior cache: %s", cache_path)
                return cached["prior"]
            if logger is not None:
                logger.info("Cardinality prior cache metadata changed; rebuilding %s", cache_path)
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to load cardinality prior cache %s (%s); rebuilding", cache_path, exc)

    if logger is not None:
        logger.info(
            "Building train-only atom/pharmacophore cardinality prior from %d complexes; "
            "this one-time preprocessing step can take several minutes.",
            len(train_set),
        )
    prior = build_cardinality_prior(
        train_set, atom_bucket_width=atom_bucket_width,
        num_pocket_bins=num_pocket_bins, show_progress=True,
    )
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    temporary_path = "%s.tmp.%d" % (cache_path, os.getpid())
    try:
        with open(temporary_path, "wb") as handle:
            pickle.dump(
                {"metadata": expected_metadata, "prior": prior},
                handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary_path, cache_path)
        if logger is not None:
            logger.info("Saved train-only cardinality prior cache: %s", cache_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return prior


def _choice(distribution, rng):
    if not distribution:
        raise ValueError("Cardinality prior has no usable fallback distribution")
    values = np.asarray(distribution.get("values", []), dtype=np.int64)
    probabilities = np.asarray(distribution.get("probabilities", []), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or probabilities.shape != values.shape:
        raise ValueError("Malformed cardinality distribution")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Cardinality probabilities must be finite and non-negative")
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("Cardinality probabilities must have positive mass")
    return int(rng.choice(values, p=probabilities / total))


def _neighbor_distribution(prior, descriptor, target, atom_count=None):
    matrix = np.asarray(prior.get("training_descriptors"), dtype=np.float64)
    mean = np.asarray(prior.get("descriptor_mean"), dtype=np.float64)
    scale = np.asarray(prior.get("descriptor_scale"), dtype=np.float64)
    query = np.asarray(descriptor, dtype=np.float64)
    if query.shape != mean.shape or matrix.ndim != 2 or matrix.shape[1] != len(mean):
        raise ValueError("Pocket descriptor schema does not match cardinality prior")
    normalized = (query - mean) / scale
    distances = np.sqrt(np.mean((matrix - normalized[None, :]) ** 2, axis=1))
    if target == "phore" and atom_count is not None:
        training_atoms = np.asarray(prior["training_atom_counts"], dtype=np.float64)
        width = max(float(prior.get("atom_bucket_width", 4)), 1.0)
        distances = distances + 0.35 * np.abs(training_atoms - int(atom_count)) / width
    k = min(int(prior.get("neighbor_count", 32)), len(distances))
    indices = np.argpartition(distances, k - 1)[:k]
    local = distances[indices]
    bandwidth = max(float(np.median(local)), 0.15)
    weights = np.exp(-0.5 * (local / bandwidth) ** 2) + 1e-8
    targets = np.asarray(
        prior["training_atom_counts" if target == "atom" else "training_phore_counts"]
    )[indices]
    counter = defaultdict(float)
    for value, weight in zip(targets, weights):
        counter[int(value)] += float(weight)
    return _distribution(counter)


def _stratified_choice(distribution, stratum, num_strata, rng):
    if int(num_strata) <= 1:
        return _choice(distribution, rng)
    values = np.asarray(distribution["values"], dtype=np.int64)
    probabilities = np.asarray(distribution["probabilities"], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    low = float(stratum) / int(num_strata)
    high = float(stratum + 1) / int(num_strata)
    quantile = low + (high - low) * float(rng.random())
    return int(values[np.searchsorted(np.cumsum(probabilities), quantile, side="right")])


def sample_atom_counts(
    prior, pocket_sizes=None, rng=None, pocket_descriptor_vectors=None,
    stratified_bins=1,
):
    if not prior or prior.get("version") not in {2, 3}:
        raise ValueError("A version-2/3 training-only cardinality prior is required")
    rng = np.random if rng is None else rng
    if pocket_descriptor_vectors is not None:
        descriptors = list(pocket_descriptor_vectors)
    else:
        descriptors = None
    if pocket_sizes is None:
        if descriptors is None:
            raise ValueError("pocket_sizes or pocket_descriptor_vectors is required")
        pocket_sizes = [float(np.asarray(value)[0]) for value in descriptors]
    if descriptors is not None and len(descriptors) != len(pocket_sizes):
        raise ValueError("Pocket size/descriptor counts must match")

    bounds = prior.get("pocket_bounds", [])
    by_pocket = prior.get("atom_by_pocket", {})
    global_distribution = prior.get("atom_global")
    result = []
    for index, pocket_size in enumerate(pocket_sizes):
        if prior.get("version") == 3 and descriptors is not None:
            distribution = _neighbor_distribution(
                prior, descriptors[index], target="atom"
            )
        else:
            pbin = pocket_size_bin(float(pocket_size), bounds)
            distribution = by_pocket.get(str(pbin)) or global_distribution
        count = _stratified_choice(
            distribution, index % max(int(stratified_bins), 1),
            max(int(stratified_bins), 1), rng,
        )
        result.append(max(1, count))
    return result


def sample_pharmacophore_counts(
    prior, pocket_sizes, atom_counts, rng=None, pocket_descriptor_vectors=None,
):
    if not prior or prior.get("version") not in {2, 3}:
        raise ValueError("A version-2/3 training-only cardinality prior is required")
    if len(pocket_sizes) != len(atom_counts):
        raise ValueError("pocket_sizes and atom_counts must have the same length")
    descriptors = None if pocket_descriptor_vectors is None else list(pocket_descriptor_vectors)
    if descriptors is not None and len(descriptors) != len(atom_counts):
        raise ValueError("Pocket descriptor/atom count lengths must match")
    rng = np.random if rng is None else rng
    width = int(prior.get("atom_bucket_width", 4))
    bounds = prior.get("pocket_bounds", [])
    conditional = prior.get("conditional", {})
    by_pocket = prior.get("by_pocket", {})
    by_atom = prior.get("by_atom", {})
    global_distribution = prior.get("global")
    result = []
    for index, (pocket_size, atom_count) in enumerate(zip(pocket_sizes, atom_counts)):
        if prior.get("version") == 3 and descriptors is not None:
            distribution = _neighbor_distribution(
                prior, descriptors[index], target="phore", atom_count=atom_count
            )
        else:
            pbin = pocket_size_bin(float(pocket_size), bounds)
            abin = atom_count_bin(int(atom_count), width)
            distribution = (
                conditional.get(f"{pbin}:{abin}")
                or by_pocket.get(str(pbin))
                or by_atom.get(str(abin))
                or global_distribution
            )
        count = _choice(distribution, rng)
        result.append(max(0, min(count, int(atom_count))))
    return result
