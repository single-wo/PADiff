import warnings
import numpy as np
import torch
from .warmup import GradualWarmupScheduler

class ExponentialLRWithMinimum(torch.optim.lr_scheduler.ExponentialLR):

    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1):
        self.min_lr = min_lr
        super().__init__(optimizer, gamma, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "Use get_last_lr() to read the scheduler's current rate.",
                UserWarning,
            )
        if self.last_epoch == 0:
            return self.base_lrs
        return [
            max(group["lr"] * self.gamma, self.min_lr)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


def infinite_iterator(iterable):

    while True:
        yield from iterable


def empty_anchor_context(batch, config):

    if not bool(getattr(config.model, "condition_on_pocket_only", False)):
        raise ValueError("Pocket-only training cannot consume ligand-derived anchors")
    return (
        batch.protein_element.new_empty((0,), dtype=torch.long),
        batch.protein_pos.new_empty((0, 3)),
        batch.protein_element_batch.new_empty((0,), dtype=torch.long),
        batch.protein_pos.new_empty((0,)),
        batch.protein_pos.new_empty((0, 3)),
    )


def model_loss_from_batch(model, batch, config, training=False, time_step=None):

    protein_pos = batch.protein_pos
    ligand_pos = batch.ligand_pos
    pharmacophore_pos = batch.phore_pos
    if training:
        protein_pos = protein_pos + torch.randn_like(protein_pos) * float(
            config.train.pos_noise_std
        )
        ligand_pos = ligand_pos + torch.randn_like(ligand_pos) * float(
            getattr(
                config.train,
                "ligand_pos_noise_std",
                config.train.pos_noise_std,
            )
        )
        if bool(getattr(config.model, "use_pharmacophore", True)):
            pharmacophore_pos = (
                pharmacophore_pos
                + torch.randn_like(pharmacophore_pos)
                * float(getattr(
                    config.train,
                    "phore_pos_noise_std",
                    config.train.pos_noise_std,
                ))
            )

    anchor_type, anchor_pos, anchor_batch, anchor_confidence, anchor_vec = (
        empty_anchor_context(batch, config)
    )
    return model.get_loss(
        protein_node=batch.protein_atom_feat.float(),
        protein_pos=protein_pos,
        protein_batch=batch.protein_element_batch,
        anchor_type=anchor_type,
        anchor_pos=anchor_pos,
        anchor_batch=anchor_batch,
        anchor_confidence=anchor_confidence,
        anchor_vec=anchor_vec,
        phore_type=batch.phore_type,
        phore_pos=pharmacophore_pos,
        phore_batch=batch.phore_type_batch,
        phore_vec=batch.phore_vec,
        ligand_node=batch.ligand_atom_feat_full,
        ligand_element=batch.ligand_element,
        ligand_pos=ligand_pos,
        ligand_batch=batch.ligand_element_batch,
        halfedge_type=batch.ligand_halfedge_type,
        halfedge_index=batch.ligand_halfedge_index,
        halfedge_batch=batch.ligand_halfedge_type_batch,
        ph2atom_edge_index=getattr(batch, "ph2atom_edge_index", None),
        ligand_atom_feature=getattr(batch, "ligand_atom_feature", None),
        num_mol=batch.num_graphs,
        time_step=time_step,
    )


def build_optimizer(config, model):
    if config.type == "adam":
        optimizer_type = torch.optim.Adam
    elif config.type == "adamw":
        optimizer_type = torch.optim.AdamW
    else:
        raise NotImplementedError("Unsupported optimizer: %s" % config.type)
    return optimizer_type(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )


def build_scheduler(config, optimizer):
    if config.type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
    if config.type == "warmup_plateau":
        return GradualWarmupScheduler(
            optimizer,
            multiplier=config.multiplier,
            total_epoch=config.total_epoch,
            after_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr,
            ),
        )
    if config.type in {"expmin", "expmin_milestone"}:
        gamma = config.factor
        if config.type == "expmin_milestone":
            gamma = np.exp(np.log(config.factor) / config.milestone)
        return ExponentialLRWithMinimum(
            optimizer, gamma=gamma, min_lr=config.min_lr
        )
    raise NotImplementedError("Unsupported scheduler: %s" % config.type)
