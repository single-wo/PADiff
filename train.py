import argparse
import os
import random
import shutil
import sys
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
sys.path.append(".")
torch.multiprocessing.set_sharing_strategy("file_system")
from models.model import PADiff, ligand_atom_class_layout
import utils.transforms as transforms
from utils.dataset import get_dataset
from utils.misc import get_logger, get_new_log_dir, load_config, seed_all
from utils.cardinality_prior import load_or_build_cardinality_prior
from utils.train_utils import (
    build_optimizer,
    build_scheduler,
    infinite_iterator,
    model_loss_from_batch,
)


def compute_multiclass_auroc(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weighted, count = 0.0, 0
    for class_id in np.unique(y_true):
        binary = (y_true == class_id).astype(np.int64)
        if binary.min() == binary.max() or class_id >= y_pred.shape[1]:
            continue
        score = np.nan_to_num(y_pred[:, class_id], nan=0.0, posinf=1e10, neginf=-1e10)
        n_class = int(binary.sum())
        weighted += roc_auc_score(binary, score) * n_class
        count += n_class
    return weighted / max(count, 1)


def _plain_config(value):

    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def _scalar_value(metrics, key):

    if key not in metrics:
        return None
    value = metrics[key]
    if torch.is_tensor(value):
        value = value.detach().item()
    return float(value)


def format_train_console_metrics(loss_dict):

    groups = [
        ("loss", ("loss",)),
        ("ligand(pos/node/edge)", (
            "loss_ligand_pos", "loss_ligand_node", "loss_ligand_edge",
        )),
        ("geometry(bond/internal/valence/pocket)", (
            "loss_ligand_bond_length", "loss_ligand_internal_clash",
            "loss_ligand_distance_valence",
            "loss_pocket_ligand_clash",
        )),
        ("phore(pos/type/assign)", (
            "loss_phore_pos", "loss_phore_type",
            "loss_phore_atom_assignment",
        )),
    ]
    parts = []
    for label, keys in groups:
        values = [_scalar_value(loss_dict, key) for key in keys]
        values = [value for value in values if value is not None]
        if not values:
            continue
        if len(values) == 1:
            parts.append("%s=%.4f" % (label, values[0]))
        else:
            parts.append("%s=%s" % (
                label, "/".join("%.4f" % value for value in values)
            ))
    return " | ".join(parts)



def format_validation_console_metrics(metrics):

    parts = ["loss=%.4f" % float(metrics["loss"])]
    parts.append("bond_f1=%.4f" % float(metrics.get("bond_f1", 0.0)))
    if "phore_accuracy" in metrics:
        parts.append("phore_acc=%.4f" % float(metrics["phore_accuracy"]))
    parts.append(
        "selection=%.4f" % float(metrics["checkpoint_selection_score"])
    )
    return " | ".join(parts)


def get_bond_classification_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred)
    predicted = y_pred.argmax(axis=-1).astype(np.int64)
    true_is_bond = (y_true > 0) & (y_true < 5)
    pred_is_bond = (predicted > 0) & (predicted < 5)
    tp = int((true_is_bond & pred_is_bond).sum())
    fp = int((~true_is_bond & pred_is_bond).sum())
    fn = int((true_is_bond & ~pred_is_bond).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "bond_precision": float(precision),
        "bond_recall": float(recall),
        "bond_f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "bond_type_accuracy_true_bonds": float(
            (predicted[true_is_bond] == y_true[true_is_bond]).mean()
        ) if true_is_bond.any() else 0.0,
        "pred_bond_density": float(pred_is_bond.mean()) if len(predicted) else 0.0,
        "true_bond_density": float(true_is_bond.mean()) if len(y_true) else 0.0,
        "pred_aromatic_fraction": float(
            (predicted[pred_is_bond] == 4).mean()
        ) if pred_is_bond.any() else 0.0,
        "true_aromatic_fraction": float(
            (y_true[true_is_bond] == 4).mean()
        ) if true_is_bond.any() else 0.0,
    }
    names = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}
    for class_id, name in names.items():
        true_class = y_true == class_id
        pred_class = predicted == class_id
        class_tp = int((true_class & pred_class).sum())
        class_fp = int((~true_class & pred_class).sum())
        class_fn = int((true_class & ~pred_class).sum())
        metrics["bond_%s_precision" % name] = float(
            class_tp / max(class_tp + class_fp, 1)
        )
        metrics["bond_%s_recall" % name] = float(
            class_tp / max(class_tp + class_fn, 1)
        )
    return metrics

def _clamp_unit(value):
    return min(max(float(value), 0.0), 1.0)


def checkpoint_component_scores(metrics, config):
    scores = {
        "bond": float(
            _clamp_unit(metrics.get("bond_f1", 0.0))
            + _clamp_unit(metrics.get("bond_type_accuracy_true_bonds", 0.0))
        ),
    }
    if bool(getattr(config.model, "use_pharmacophore", True)):
        scores["phore"] = float(
            0.5 * _clamp_unit(metrics.get("phore_accuracy", 0.0))
            + 0.25 * _clamp_unit(metrics.get("phore_auroc", 0.0))
        )
    return scores


def checkpoint_selection_score(metrics, config):
    return float(
        -0.25 * float(metrics.get("loss", float("inf")))
        + sum(checkpoint_component_scores(metrics, config).values())
    )


def _require_pocket_only_checkpoint(
    checkpoint, checkpoint_path, purpose, expected_model=None,
):
    checkpoint_config = checkpoint.get("config")
    checkpoint_model_config = getattr(checkpoint_config, "model", None)
    is_pocket_only = bool(
        getattr(checkpoint_model_config, "condition_on_pocket_only", False)
    )
    if not is_pocket_only:
        raise ValueError(
            f"Cannot {purpose} from {checkpoint_path!r}: the checkpoint was not "
            "trained with model.condition_on_pocket_only=true. Start a fresh run "
            "or use a checkpoint produced by this pocket-only training pipeline."
        )
    if expected_model is not None:
        checkpoint_use_phore = bool(getattr(
            checkpoint_model_config, "use_pharmacophore", True
        ))
        if checkpoint_use_phore != expected_model.use_pharmacophore:
            raise ValueError(
                "Cannot resume across the pharmacophore ablation boundary; "
                "start a fresh run for use_pharmacophore=%s."
                % expected_model.use_pharmacophore
            )
        checkpoint_num_phore = getattr(
            checkpoint_model_config, "num_phore_types", None
        )
        checkpoint_is_legacy = checkpoint_num_phore == 11
        if checkpoint_is_legacy != expected_model.legacy_phore_schema:
            raise ValueError(
                "Cannot resume an 11-class legacy checkpoint into the compact "
                "six-class model. Start a fresh run with the corrected schema."
            )

def train_step(args, config, model, train_iterator, optimizer, scaler, logger, writer, it):
    optimizer.zero_grad(set_to_none=True)
    batch = next(train_iterator).to(args.device)
    amp_enabled = bool(config.train.use_amp) and args.device.startswith("cuda")
    amp_dtype_name = str(getattr(config.train, "amp_dtype", "bfloat16")).lower()
    if amp_dtype_name not in {"bfloat16", "float16"}:
        raise ValueError("train.amp_dtype must be 'bfloat16' or 'float16'")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"
    with torch.autocast(
        device_type=autocast_device, dtype=amp_dtype, enabled=amp_enabled
    ):
        loss_dict, _ = model_loss_from_batch(model, batch, config, training=True)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss at iteration {it}: {loss_dict}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    max_grad_norm = float(config.train.max_grad_norm)
    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_value = float(grad_norm.detach().cpu())
    if not np.isfinite(grad_norm_value):
        if scaler.is_enabled():
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            logger.warning(
                "[AMP] Iter %d skipped due to non-finite scaled gradients | "
                "grad_norm: %s | scale: %.1f -> %.1f",
                it, grad_norm_value, scale_before, scale_after,
            )
            writer.add_scalar("train/amp_skipped_step", 1, it)
            writer.add_scalar("train/amp_scale", scale_after, it)
            writer.flush()
            return False
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"Non-finite gradient norm at iteration {it}: {grad_norm_value}"
        )
    grad_was_clipped = grad_norm_value > max_grad_norm
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    scaler.step(optimizer)
    scaler.update()

    if it % config.train.train_report_iter == 0:
        lr_text = ",".join("%.3e" % lr for lr in learning_rates)
        loss_text = format_train_console_metrics(loss_dict)
        logger.info(
            "[Train] Iter %d | lr=%s | grad=%.4f%s | %s",
            it, lr_text, grad_norm_value,
            " (clipped)" if grad_was_clipped else "", loss_text,
        )
        for key, value in loss_dict.items():
            writer.add_scalar("train/%s" % key, value.item(), it)
        writer.add_scalar("train/lr", learning_rates[0], it)
        writer.add_scalar("train/grad_norm", grad_norm_value, it)
        writer.flush()
    return True



def get_validation_protocol(config, num_timesteps):
    num_eval_timesteps = int(getattr(config.train, "val_num_timesteps", 10))
    if num_eval_timesteps <= 0:
        raise ValueError("train.val_num_timesteps must be a positive integer")
    num_eval_timesteps = min(num_eval_timesteps, int(num_timesteps))
    timesteps = [
        min(
            int(num_timesteps) - 1,
            int((index + 0.5) * int(num_timesteps) / num_eval_timesteps),
        )
        for index in range(num_eval_timesteps)
    ]
    return {
        "name": "fixed_stratified_timesteps",
        "num_timesteps": num_eval_timesteps,
        "timesteps": timesteps,
        "seed": int(getattr(config.train, "val_seed", config.train.seed)),
        "aggregation": "num_graphs_weighted",
        "use_pharmacophore": bool(getattr(
            config.model, "use_pharmacophore", True
        )),
    }


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])



def build_checkpoint(
    config, model, optimizer, scheduler, iteration, metrics,
    best_loss, best_iteration, validation_protocol, cardinality_prior,
    best_selection_score=None,
):
    return {
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": int(iteration),
        "metrics": metrics,
        "best_loss": float(best_loss),
        "best_iteration": int(best_iteration),
        "best_selection_score": (
            None if best_selection_score is None else float(best_selection_score)
        ),
        "checkpoint_selection": "loss_plus_bond_and_pharmacophore_quality",
        "validation_protocol": validation_protocol,
        "cardinality_prior": cardinality_prior,
        "generation_metadata": {
            "version": 4,
            "conditioning": "protein_pocket_only",
            "stages": 1,
            "joint_outputs": (
                ["pharmacophore_type_position_vector"]
                if model.use_pharmacophore else []
            ) + ["ligand_atom_type_position", "ligand_bond_type"],
            "use_pharmacophore": bool(model.use_pharmacophore),
            "pharmacophore_schema": (
                "legacy_11_raw" if model.legacy_phore_schema else "active_6_compact"
            ),
            "uses_reference_ligand_condition": False,
            "uses_reference_pharmacophore_condition": False,
            "uses_reference_anchor_condition": False,
            "cardinality_source": "training_split_pocket_prior",
            "atom_capacity_mode": "pocket_prior_exact_atom_count",
            "pharmacophore_count_source": (
                "training_split_conditional_prior"
                if model.use_pharmacophore else "disabled"
            ),
            "diffusion_mask_class_index": getattr(model, "mask_atom_class", None),
        },
    }



def validate(
    args, config, model, val_loader, scheduler, logger, writer, it,
    step_scheduler=True,
):
    protocol = get_validation_protocol(config, model.num_timesteps)
    sums, total_graph_evaluations = {}, 0
    pred_atom, true_atom = [], []
    pred_bond, true_bond = [], []
    pred_phore, true_phore = [], []

    amp_enabled = bool(config.train.use_amp) and args.device.startswith("cuda")
    amp_dtype_name = str(getattr(config.train, "amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"

    rng_state = _capture_rng_state()
    model.eval()
    try:
        seed_all(protocol["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(protocol["seed"])
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                batch_graphs = int(batch.num_graphs)
                for timestep in protocol["timesteps"]:
                    time_step = torch.full(
                        (batch_graphs,), timestep,
                        dtype=torch.long, device=args.device,
                    )
                    with torch.autocast(
                        device_type=autocast_device,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        loss_dict, preds = model_loss_from_batch(
                            model, batch, config, training=False,
                            time_step=time_step,
                        )
                    for key, value in loss_dict.items():
                        sums[key] = sums.get(key, 0.0) + value.item() * batch_graphs
                    total_graph_evaluations += batch_graphs
                    pred_atom.append(preds["pred_ligand_node"].float().cpu().numpy())
                    true_atom.append(batch.ligand_atom_feat_full.cpu().numpy())
                    pred_bond.append(
                        preds["pred_ligand_halfedge"].float().cpu().numpy()
                    )
                    true_bond.append(batch.ligand_halfedge_type.cpu().numpy())
                    if model.use_pharmacophore and batch.phore_type.numel() > 0:
                        pred_phore.append(
                            preds["pred_phore_node"].float().cpu().numpy()
                        )
                        true_phore.append(
                            model.phore_types_to_model(batch.phore_type).cpu().numpy()
                        )
    finally:
        _restore_rng_state(rng_state)
        model.train()

    denominator = max(total_graph_evaluations, 1)
    average = {key: value / denominator for key, value in sums.items()}
    atom_targets = np.concatenate(true_atom)
    atom_logits = np.concatenate(pred_atom)
    average["atom_auroc"] = compute_multiclass_auroc(atom_targets, atom_logits)
    bond_targets = np.concatenate(true_bond)
    bond_logits = np.concatenate(pred_bond)
    average["bond_auroc"] = compute_multiclass_auroc(bond_targets, bond_logits)
    average.update(get_bond_classification_metrics(bond_targets, bond_logits))
    if pred_phore:
        phore_targets = np.concatenate(true_phore)
        phore_logits = np.concatenate(pred_phore)
        average["phore_auroc"] = compute_multiclass_auroc(
            phore_targets, phore_logits
        )
        average["phore_accuracy"] = float(
            (phore_logits.argmax(-1) == phore_targets).mean()
        ) if phore_targets.size else 0.0
    average["checkpoint_selection_score"] = checkpoint_selection_score(
        average, config
    )
    if step_scheduler:
        if config.train.scheduler.type == "plateau":
            scheduler.step(average["loss"])
        elif config.train.scheduler.type == "warmup_plateau":
            scheduler.step_ReduceLROnPlateau(average["loss"])
        else:
            scheduler.step()

    validation_lrs = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
    logger.info(
        "[Validate] iter=%d | lr=%s | %s",
        it, ",".join("%.3e" % lr for lr in validation_lrs),
        format_validation_console_metrics(average),
    )
    for key, value in average.items():
        writer.add_scalar("val/%s" % key, value, it)
    writer.flush()
    return average


def main(args):
    config = load_config(args.config)
    if args.no_pharmacophore:
        config.model.use_pharmacophore = False
    config.model.name = (
        "padiff" if bool(config.model.use_pharmacophore)
        else "padiff_ligand_only"
    )
    config.train.resume = bool(args.resume) and not args.fresh
    if args.resume:
        config.train.resume_ckpt = args.resume
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    seed_all(config.train.seed)

    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger("train", log_dir)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    logger.info(
        "Config=%s | model=%s | device=%s | run_dir=%s",
        args.config, config.model.name, args.device, log_dir,
    )
    with open(os.path.join(log_dir, os.path.basename(args.config)), "w") as handle:
        yaml.safe_dump(_plain_config(config), handle, sort_keys=False)

    featurizer = transforms.FeatureComplex(
        config.data.transform.ligand_atom_mode,
        sample=config.data.transform.sample,
    )
    transform_list = [featurizer]
    if config.data.transform.random_rot:
        transform_list.append(transforms.RandomRotation())
    dataset, subsets = get_dataset(
        config=config.data, transform=Compose(transform_list)
    )
    train_set = subsets["train"]
    val_set = subsets["test"]
    logger.info(
        "Dataset size: train=%d, validation=%d",
        len(train_set), len(val_set),
    )
    if not bool(getattr(config.model, "condition_on_pocket_only", False)):
        raise ValueError(
            "This training entry point now requires model.condition_on_pocket_only=true."
        )
    prior_cache_path = getattr(
        config.train, "cardinality_prior_cache",
        config.data.path + ".cardinality_prior_v2.pkl",
    )
    cardinality_prior = load_or_build_cardinality_prior(
        train_set=train_set,
        cache_path=prior_cache_path,
        data_path=config.data.path,
        split_path=config.data.split,
        logger=logger,
    )
    logger.info(
        "Ready pocket-only atom/pharmacophore cardinality prior: %d conditional bins",
        len(cardinality_prior["conditional"]),
    )
    loader_kwargs = dict(
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        follow_batch=featurizer.follow_batch,
        exclude_keys=featurizer.exclude_keys,
        pin_memory=config.train.pin_memory,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_set, shuffle=False, **loader_kwargs)
        if val_set is not None else None
    )
    train_iterator = infinite_iterator(train_loader)

    atom_layout = ligand_atom_class_layout(
        featurizer.atom_feat_dim, config.model
    )
    ligand_node_types = atom_layout["ligand_node_types"]
    model = PADiff(
        config=config.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=ligand_node_types,
        real_atom_type_count=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim,
    ).to(args.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: pharmacophore=%s | ligand classes=%d | phore classes=%d | parameters=%.4f M",
        "on" if model.use_pharmacophore else "off",
        ligand_node_types, model.num_phore_types, n_parameters / 1e6,
    )

    optimizer = build_optimizer(config.train.optimizer, model)
    scheduler = build_scheduler(config.train.scheduler, optimizer)
    amp_enabled = bool(config.train.use_amp) and args.device.startswith("cuda")
    amp_dtype_name = str(getattr(config.train, "amp_dtype", "bfloat16")).lower()
    if amp_dtype_name not in {"bfloat16", "float16"}:
        raise ValueError("train.amp_dtype must be 'bfloat16' or 'float16'")
    scaler_enabled = amp_enabled and amp_dtype_name == "float16"
    scaler_kwargs = dict(
        enabled=scaler_enabled,
        init_scale=float(getattr(config.train, "amp_init_scale", 256.0)),
        growth_interval=int(getattr(config.train, "amp_growth_interval", 2000)),
    )
    amp_grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if amp_grad_scaler is not None:
        scaler = amp_grad_scaler("cuda", **scaler_kwargs)
    else:
        scaler = torch.cuda.amp.GradScaler(**scaler_kwargs)
    logger.info(
        "Mixed precision: enabled=%s dtype=%s grad_scaler=%s",
        amp_enabled, amp_dtype_name, scaler.is_enabled(),
    )
    resume_step = 0
    best_loss = float("inf")
    best_selection_score = float("-inf")
    best_iteration = 0
    best_ckpt_path = None
    validation_protocol = get_validation_protocol(config, model.num_timesteps)
    logger.info(
        "Validation=%s | timesteps=%s | seed=%d",
        validation_protocol["name"], validation_protocol["timesteps"],
        validation_protocol["seed"],
    )
    if config.train.resume:
        checkpoint = torch.load(
            config.train.resume_ckpt, map_location=args.device, weights_only=False
        )
        _require_pocket_only_checkpoint(
            checkpoint, config.train.resume_ckpt, purpose="resume training",
            expected_model=model,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_step = int(checkpoint.get("iteration", 0))
        resume_metrics = checkpoint.get("metrics", {})
        saved_protocol = checkpoint.get("validation_protocol")
        protocol_matches = saved_protocol == validation_protocol

        if protocol_matches:
            best_loss = float(checkpoint.get(
                "best_loss", resume_metrics.get("loss", float("inf"))
            ))
            best_iteration = int(checkpoint.get(
                "best_iteration", resume_step if np.isfinite(best_loss) else 0
            ))
            saved_best_score = checkpoint.get("best_selection_score")
            if saved_best_score is None:
                saved_best_score = checkpoint_selection_score(resume_metrics, config)
            best_selection_score = float(saved_best_score)
            if np.isfinite(best_loss) and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

        if not protocol_matches or not np.isfinite(best_loss):
            if not protocol_matches:
                logger.info(
                    "[Resume] Checkpoint validation protocol is missing or changed; "
                    "re-evaluating iteration %d with the current fixed protocol.",
                    resume_step,
                )
            else:
                logger.info(
                    "[Resume] Checkpoint has no finite validation loss; "
                    "re-evaluating iteration %d with the current fixed protocol.",
                    resume_step,
                )
            resume_metrics = validate(
                args, config, model, val_loader, scheduler, logger, writer,
                resume_step, step_scheduler=False,
            )
            best_loss = float(resume_metrics["loss"])
            best_selection_score = float(
                resume_metrics["checkpoint_selection_score"]
            )
            best_iteration = resume_step

        logger.info(
            "[Resume] Loaded %s | iteration=%d | best validation loss=%s | "
            "best iteration=%d",
            config.train.resume_ckpt, resume_step,
            "%.6f" % best_loss if np.isfinite(best_loss) else "unknown",
            best_iteration,
        )
        if np.isfinite(best_loss):
            best_ckpt_path = os.path.join(checkpoint_dir, "%d.pt" % best_iteration)
            source_best_ckpt_path = os.path.join(
                os.path.dirname(os.path.abspath(config.train.resume_ckpt)),
                "%d.pt" % best_iteration,
            )
            if resume_step == best_iteration:
                torch.save(build_checkpoint(
                    config, model, optimizer, scheduler, resume_step,
                    resume_metrics, best_loss, best_iteration,
                    validation_protocol, cardinality_prior,
                    best_selection_score,
                ), best_ckpt_path)
                logger.info(
                    "[Checkpoint] Initialized best %d.pt from resumed model | "
                    "best validation loss: %.6f",
                    best_iteration, best_loss,
                )
            elif os.path.isfile(source_best_ckpt_path):
                shutil.copyfile(source_best_ckpt_path, best_ckpt_path)
                logger.info(
                    "[Checkpoint] Copied historical best %d.pt | "
                    "best validation loss: %.6f | resumed iteration: %d",
                    best_iteration, best_loss, resume_step,
                )
            else:
                best_ckpt_path = None
                logger.warning(
                    "[Checkpoint] Historical best iteration is %d (validation loss %.6f), "
                    "but %s is unavailable; keeping the metric threshold without "
                    "writing incorrectly labelled model weights.",
                    best_iteration, best_loss, source_best_ckpt_path,
                )

    model.train()
    for it in range(resume_step + 1, config.train.max_iters + 1):
        train_step(args, config, model, train_iterator, optimizer, scaler, logger, writer, it)
        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            metrics = validate(args, config, model, val_loader, scheduler, logger, writer, it)
            current_loss = float(metrics["loss"])
            current_selection_score = float(metrics["checkpoint_selection_score"])
            overall_improved = current_selection_score > best_selection_score
            previous_best_ckpt_path = best_ckpt_path
            if overall_improved:
                best_loss = current_loss
                best_selection_score = current_selection_score
                best_iteration = it

            checkpoint_payload = build_checkpoint(
                config, model, optimizer, scheduler, it, metrics,
                best_loss, best_iteration, validation_protocol, cardinality_prior,
                best_selection_score,
            )

            if overall_improved:
                new_best_ckpt_path = os.path.join(checkpoint_dir, "%d.pt" % it)
                torch.save(checkpoint_payload, new_best_ckpt_path)
                if (
                    previous_best_ckpt_path is not None
                    and previous_best_ckpt_path != new_best_ckpt_path
                    and os.path.exists(previous_best_ckpt_path)
                ):
                    os.remove(previous_best_ckpt_path)
                best_ckpt_path = new_best_ckpt_path
                best_alias_path = os.path.join(checkpoint_dir, "best.pt")
                best_alias_tmp_path = best_alias_path + ".tmp"
                shutil.copyfile(new_best_ckpt_path, best_alias_tmp_path)
                os.replace(best_alias_tmp_path, best_alias_path)
                logger.info(
                    "[Checkpoint] Updated %d.pt and best.pt | loss: %.6f | "
                    "selection score: %.6f | best iteration: %d",
                    best_iteration, current_loss, best_selection_score, best_iteration,
                )
            latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
            latest_tmp_path = latest_ckpt_path + ".tmp"
            torch.save(checkpoint_payload, latest_tmp_path)
            os.replace(latest_tmp_path, latest_ckpt_path)
            writer.add_scalar("val/best_loss", best_loss, it)
            writer.add_scalar(
                "val/best_checkpoint_selection_score", best_selection_score, it
            )
            writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="data/logs")
    parser.add_argument(
        "--no-pharmacophore", action="store_true",
        help="Train the ligand-only ablation with all pharmacophore nodes/heads/losses disabled.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="CHECKPOINT",
        help="Resume from a compatible checkpoint; fresh training is the default.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start fresh even if --resume was supplied.",
    )
    main(parser.parse_args())
