"""Sparse MoE FFN with shared expert and bias-balanced sigmoid router (§4, §7).

Router: ``s = sigmoid(W h)`` in FP32. Selection uses the top-k of ``s + b``
where ``b`` is a load-balancing bias (a buffer, not a gradient-trained
parameter, checkpointed with the model). Output weights use the *original*
affinities normalised over the selected experts, so the bias never scales
expert outputs. Node-limited routing restricts each token to experts on at most
``max_nodes_per_token`` expert-parallel nodes. Computation is dropless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import Modality
from ..config import MoEConfig
from ..kernels.moe import IMPLEMENTATIONS, TokenDropError, expert_counts
from .layers import SwiGLU

NUM_MODALITIES = len(Modality)


@dataclass
class RouterStats:
    counts: torch.Tensor  # [E] selected-slot counts over valid tokens
    modality_counts: torch.Tensor  # [M, E]
    tokens: int
    aux_loss: torch.Tensor  # global-batch balancing loss (scalar)
    z_loss: torch.Tensor
    entropy: torch.Tensor  # mean entropy of normalised affinities

    def detach(self) -> "RouterStats":
        return RouterStats(self.counts, self.modality_counts, self.tokens, self.aux_loss.detach(), self.z_loss.detach(), self.entropy.detach())


class Router(nn.Module):
    def __init__(self, d_model: int, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.routed_experts, d_model))
        self.register_buffer("bias", torch.zeros(cfg.routed_experts, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: [N, d] -> (idx [N,k], weights [N,k] fp32, affinity [N,E] fp32, logits [N,E] fp32)."""
        cfg = self.cfg
        logits = h.float() @ self.weight.float().t()
        s = torch.sigmoid(logits)
        sel = s + self.bias
        n_nodes = cfg.routed_experts // cfg.experts_per_node
        if n_nodes > cfg.max_nodes_per_token:
            per_node = sel.view(-1, n_nodes, cfg.experts_per_node)
            m = min(cfg.experts_per_node, max(1, math.ceil(cfg.top_k / cfg.max_nodes_per_token)))
            node_score = per_node.topk(m, dim=-1).values.sum(-1)  # [N, nodes]
            keep_nodes = node_score.topk(cfg.max_nodes_per_token, dim=-1).indices
            node_mask = torch.zeros_like(node_score, dtype=torch.bool).scatter_(1, keep_nodes, True)
            sel = sel.masked_fill(~node_mask.repeat_interleave(cfg.experts_per_node, dim=1), float("-inf"))
        idx = sel.topk(cfg.top_k, dim=-1).indices
        w = s.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
        return idx, w, s, logits

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor) -> None:
        """Auxiliary-loss-free balancing: raise the bias of under-loaded experts (DeepSeek-V3)."""
        c = counts.float().to(self.bias.device)
        if c.sum() <= 0:
            return
        self.bias.add_(self.cfg.bias_update_rate * torch.sign(c.mean() - c))


class MoELayer(nn.Module):
    def __init__(self, d_model: int, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        E, h = cfg.routed_experts, cfg.expert_hidden
        self.router = Router(d_model, cfg)
        self.w_gate = nn.Parameter(torch.empty(E, d_model, h))
        self.w_up = nn.Parameter(torch.empty(E, d_model, h))
        self.w_down = nn.Parameter(torch.empty(E, h, d_model))
        self.shared = nn.ModuleList([SwiGLU(d_model, h) for _ in range(cfg.shared_experts)])
        self.impl = "grouped"

    def reset_parameters(self, std: float, out_std: float) -> None:
        for p, s in ((self.w_gate, std), (self.w_up, std), (self.w_down, out_std), (self.router.weight, std)):
            nn.init.trunc_normal_(p, std=s, a=-3 * s, b=3 * s)

    def forward(
        self, x: torch.Tensor, modality_ids: Optional[torch.Tensor] = None, valid: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, RouterStats]:
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        N = flat.shape[0]
        idx, w, s, logits = self.router(flat)
        out = IMPLEMENTATIONS[self.impl](flat, idx, w, self.w_gate, self.w_up, self.w_down)
        for sh in self.shared:
            out = out + sh(flat)
        # ---- statistics and balancing losses (valid tokens only)
        E = self.cfg.routed_experts
        vmask = torch.ones(N, dtype=torch.bool, device=x.device) if valid is None else valid.reshape(-1)
        idx_v = idx[vmask]
        counts = expert_counts(idx_v, E)
        if int(counts.sum()) != idx_v.numel():
            raise TokenDropError("router statistics disagree with dispatched slots")
        mod = torch.zeros(N, dtype=torch.long, device=x.device) if modality_ids is None else modality_ids.reshape(-1).long()
        mod_v = mod[vmask].unsqueeze(-1).expand_as(idx_v)
        modality_counts = torch.zeros(NUM_MODALITIES * E, dtype=torch.long, device=x.device)
        modality_counts.index_add_(0, (mod_v * E + idx_v).reshape(-1), torch.ones(idx_v.numel(), dtype=torch.long, device=x.device))
        n_valid = int(vmask.sum())
        if n_valid:
            s_v = s[vmask]
            p = s_v / s_v.sum(-1, keepdim=True).clamp(min=1e-9)
            f = counts.float() * (E / (self.cfg.top_k * n_valid))
            aux = (f * p.mean(0)).sum()
            z = torch.logsumexp(logits[vmask], dim=-1).pow(2).mean()
            ent = -(p * p.clamp(min=1e-9).log()).sum(-1).mean()
        else:
            aux = z = ent = flat.new_zeros((), dtype=torch.float32)
        stats = RouterStats(counts, modality_counts.view(NUM_MODALITIES, E), n_valid, aux, z, ent)
        return out.view(B, T, d), stats
