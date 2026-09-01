import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.diffusion import (
    categorical_kl,
    extract,
    index_to_log_onehot,
    log_categorical,
    log_sample_categorical,
    to_torch_const,
)


class ContinuousTransition(nn.Module):

    def __init__(self, betas, num_classes=None, scaling=1.):
        super().__init__()
        self.num_classes = num_classes
        self.scaling = scaling
        alphas  = 1. - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_prev = np.concatenate([[1.], alphas_bar[:-1]])

        self.betas = to_torch_const(betas)
        self.alphas = to_torch_const(alphas)
        self.alphas_bar = to_torch_const(alphas_bar)
        self.alphas_bar_prev = to_torch_const(alphas_bar_prev)
        self.sqrt_recip_alphas_bar = to_torch_const(np.sqrt(1.0 / alphas_bar))
        self.sqrt_recipm1_alphas_bar = to_torch_const(np.sqrt(1.0 / alphas_bar - 1))

        self.coef_x0 = to_torch_const(np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar))
        self.coef_xt = to_torch_const(np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar))
        self.std = to_torch_const(np.sqrt((1 - alphas_bar_prev) * betas / (1 - alphas_bar)))

    def add_noise(self, x, time_step, batch):
        if self.num_classes is not None:  
            x = F.one_hot(x, self.num_classes).float()
        x = x / self.scaling
        a_bar = self.alphas_bar.index_select(0, time_step)
        a_bar = a_bar.index_select(0, batch).unsqueeze(-1)
        noise = torch.zeros_like(x).to(x)
        noise.normal_()
        pert = a_bar.sqrt() * x + (1 - a_bar).sqrt() * noise
        if self.num_classes is None: 
            return pert
        else:
            return pert, x
            

    def get_prev_from_recon(self, x_t, x_recon, t, batch):
        coef_x0 = extract(self.coef_x0, t, batch)
        coef_xt = extract(self.coef_xt, t, batch)

        time_zero = (t[batch] == 0).unsqueeze(-1)
        mu = coef_x0 * x_recon + coef_xt * x_t
        sigma = extract(self.std, t, batch)
        x_prev = mu + sigma * torch.randn_like(mu)
        x_prev = torch.where(time_zero, mu, x_prev)
        return x_prev

    def sample_init(self, shape):
        if self.num_classes is None:
            return torch.randn(shape).to(self.betas.device)
        else:
            return torch.randn([shape, self.num_classes]).to(self.betas.device)


class GeneralCategoricalTransition(nn.Module):
    def __init__(self, betas, num_classes, init_prob=None):
        super().__init__()
        self.eps = 1e-30
        self.num_classes = num_classes
        if init_prob is None:
            self.init_prob = np.ones(num_classes) / num_classes
        elif init_prob == 'absorb':  
            init_prob = 0.01 * np.ones(num_classes)
            init_prob[0] = 1
            self.init_prob = init_prob / np.sum(init_prob)
        elif init_prob == 'tomask':  
            init_prob = 0.001 * np.ones(num_classes)
            init_prob[-1] = 1.
            self.init_prob = init_prob / np.sum(init_prob)
        elif init_prob == 'uniform':
            self.init_prob = np.ones(num_classes) / num_classes
        else:
            self.init_prob = init_prob / np.sum(init_prob)
        self.betas = (betas)
        self.num_timesteps = len(betas)

        q_one_step_mats = [self._get_transition_mat(t) for t in range(0, self.num_timesteps)]
        q_one_step_mats = np.stack(q_one_step_mats, axis=0)  

        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = np.tensordot(q_mat_t, q_one_step_mats[t], axes=[[1], [0]])
            q_mats.append(q_mat_t)
        q_mats = np.stack(q_mats, axis=0)
        
        transpopse_q_onestep_mats = np.transpose(q_one_step_mats, axes=[0, 2, 1])
        
        self.q_mats = to_torch_const(q_mats)
        self.transpopse_q_onestep_mats = to_torch_const(transpopse_q_onestep_mats)

    def _get_transition_mat(self, t):
        beta_t = self.betas[t]
        if self.init_prob is None:
            mat = np.full(shape=(self.num_classes, self.num_classes),
                        fill_value=beta_t/float(self.num_classes),
                        dtype=np.float64)
            diag_indices = np.diag_indices_from(mat)
            diag_val = 1. - beta_t * (self.num_classes-1.)/self.num_classes
            mat[diag_indices] = diag_val
        else:
            mat = np.repeat(np.expand_dims(self.init_prob, 0), self.num_classes, axis=0)
            mat = beta_t * mat
            mat_diag = np.eye(self.num_classes) * (1. - beta_t)
            mat = mat + mat_diag
        return mat

    def add_noise(self, v, time_step, batch):
        log_node_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_node_vt = self.q_vt_sample(log_node_v0, time_step, batch)
        v_perturbed = F.one_hot(v_perturbed, self.num_classes).float()
        return v_perturbed, log_node_vt, log_node_v0
    
    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    def q_vt_sample(self, log_v0, t, batch):
        log_q_vt_v0 = self.q_vt_pred(log_v0, t, batch)
        sample_class = log_sample_categorical(log_q_vt_v0)
        log_sample = index_to_log_onehot(sample_class, self.num_classes)
        return sample_class, log_sample

    def q_vt_pred(self, log_v0, t, batch):
        qt_mat = extract(self.q_mats, t, batch, ndim=1)
        q_vt = torch.einsum('...i,...ij->...j', log_v0.exp(), qt_mat)
        return torch.log(q_vt + self.eps).clamp_min(-32.)

    def q_v_posterior(self, log_v0, log_vt, t, batch, v0_prob):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  

        fact1 = extract(self.transpopse_q_onestep_mats, t, batch, ndim=1)
        fact1 = torch.einsum('bj,bjk->bk', torch.exp(log_vt), fact1)  
        
        if not v0_prob: 
            fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)
            class_v0 = log_v0.argmax(dim=-1)
            fact2 = fact2[torch.arange(len(class_v0)), class_v0]
        else: 
            fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)  
            fact2 = torch.einsum('bj,bjk->bk', torch.exp(log_v0), fact2)  
        
        ndim = log_v0.ndim
        if ndim == 2:
            t_expand = t[batch].unsqueeze(-1)
        elif ndim == 3:
            t_expand = t[batch].unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')
        
        out = torch.log(fact1 + self.eps).clamp_min(-32.) + torch.log(fact2 + self.eps).clamp_min(-32.)
        out = out - torch.logsumexp(out, dim=-1, keepdim=True)
        out_t0 = log_v0
        out = torch.where(t_expand == 0, out_t0, out)
        return out

    def compute_v_Lt(self, log_v_post_true, log_v_post_pred, log_v0, t, batch):
        kl_v = categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = - log_categorical(log_v0, log_v_post_pred)

        ndim = log_v_post_true.ndim
        if ndim == 2:
            mask = (t == 0).float()[batch]
        elif ndim == 3:
            mask = (t == 0).float()[batch].unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')
        loss_v = mask * decoder_nll_v + (1 - mask) * kl_v
        return loss_v
        
    def sample_init(self, n):
        if n == 0:
            empty_type = torch.empty(
                0, dtype=torch.long, device=self.q_mats.device
            )
            empty_state = torch.empty(
                (0, self.num_classes), dtype=self.q_mats.dtype,
                device=self.q_mats.device,
            )
            return empty_type, empty_state, empty_state.clone()
        init_log_atom_vt = torch.log(
            torch.from_numpy(self.init_prob)+self.eps).clamp_min(-32.).to(self.q_mats.device)
        init_log_atom_vt = init_log_atom_vt.unsqueeze(0).repeat(n, 1)
        init_types = log_sample_categorical(init_log_atom_vt)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt
