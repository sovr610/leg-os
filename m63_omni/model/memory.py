"""Titans-style bounded session memory at attention anchors (blueprint §6).

Each anchor owns:

* a learned initialisation ``M0`` (two-layer MLP ``256 -> 512 -> 256``), copied
  into a fresh :class:`MemoryState` when a session starts;
* shared projections between the 4,096-d residual stream and the 256-d memory
  space (query, key, value, output) and learned gates (momentum ``eta``, inner
  step ``theta``, forgetting ``alpha``), each bounded by a sigmoid times a cap.

Reads: ``read_tokens_per_anchor`` slot queries (learned offsets plus a
projection of the *previous* segment's summary) are passed through the current
memory MLP and projected back to residual width. They depend only on completed
state, so a segment can never read information written from its own future.

Updates (only at segment boundaries, never during token-by-token generation)::

    g_t = clip(grad_M l(M; k_t, v_t))
    s_t = eta_t s_{t-1} - theta_t g_t
    M_t = (1 - alpha_t) M_{t-1} + s_t                    # decay_target = "zero" (blueprint)
    M_t = M0 + (1 - alpha_t) (M_{t-1} - M0) + s_t        # decay_target = "init" (ablation)

With ``"zero"`` forgetting erodes the learned initial memory over long
sessions; ``"init"`` forgets *toward* ``M0`` instead. Which is better is an
empirical question for the proxy tournament (``configs/tournament/memory_decay_to_init.yaml``).

Backbone weights never update at inference; the session owns only ``M`` and the
momentum ``s`` (~3 MiB at BF16 for three anchors in the reference shape).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MemoryConfig
from ..kernels.memory import assoc_grad_closed_form, assoc_loss, clip_by_norm, mlp_forward
from .layers import RMSNorm


class MemoryFrozenError(RuntimeError):
    pass


@dataclass
class MemoryState:
    W1: torch.Tensor  # [B, h, d_mem]
    W2: torch.Tensor  # [B, d_mem, h]
    S1: torch.Tensor  # momentum for W1
    S2: torch.Tensor  # momentum for W2
    summary: torch.Tensor  # [B, d_model] summary of the last completed segment
    updates: int = 0
    frozen: bool = False
    meta: dict = field(default_factory=dict)  # model/tokenizer fingerprint, parent event IDs

    def clone(self) -> "MemoryState":
        return MemoryState(self.W1.clone(), self.W2.clone(), self.S1.clone(), self.S2.clone(), self.summary.clone(),
                           self.updates, self.frozen, dict(self.meta))

    def detach(self) -> "MemoryState":
        return MemoryState(self.W1.detach(), self.W2.detach(), self.S1.detach(), self.S2.detach(), self.summary.detach(),
                           self.updates, self.frozen, dict(self.meta))

    def to(self, dtype: torch.dtype) -> "MemoryState":
        return MemoryState(self.W1.to(dtype), self.W2.to(dtype), self.S1.to(dtype), self.S2.to(dtype), self.summary,
                           self.updates, self.frozen, dict(self.meta))

    def weight_and_momentum_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.W1, self.W2, self.S1, self.S2))


class NeuralMemoryAnchor(nn.Module):
    def __init__(self, d_model: int, cfg: MemoryConfig, eps: float = 1e-6) -> None:
        super().__init__()
        self.cfg = cfg
        d_mem, d_hidden, _ = cfg.mlp_shape
        self.d_mem, self.d_hidden, self.d_model = d_mem, d_hidden, d_model
        self.W1_0 = nn.Parameter(torch.empty(d_hidden, d_mem))
        self.W2_0 = nn.Parameter(torch.empty(d_mem, d_hidden))
        self.q_proj = nn.Linear(d_model, d_mem, bias=False)
        self.k_proj = nn.Linear(d_model, d_mem, bias=False)
        self.v_proj = nn.Linear(d_model, d_mem, bias=False)
        self.o_proj = nn.Linear(d_mem, d_model, bias=False)
        self.slots = nn.Parameter(torch.empty(cfg.read_tokens_per_anchor, d_mem))
        self.gate = nn.Linear(d_model, 3, bias=True)  # eta, theta, alpha logits
        self.in_norm = RMSNorm(d_model, eps)
        self.out_norm = RMSNorm(d_model, eps)

    def reset_parameters(self, std: float) -> None:
        nn.init.normal_(self.W1_0, std=self.d_mem**-0.5)
        nn.init.normal_(self.W2_0, std=self.d_hidden**-0.5)
        nn.init.normal_(self.slots, std=1.0)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.trunc_normal_(lin.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():  # start with moderate momentum, small step, little forgetting
            self.gate.bias.copy_(torch.tensor([0.0, -1.0, -3.0]))

    # ------------------------------------------------------------------ state
    def init_state(self, batch: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> MemoryState:
        device = device or self.W1_0.device
        W1 = self.W1_0.detach().to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone()
        W2 = self.W2_0.detach().to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone()
        return MemoryState(W1, W2, torch.zeros_like(W1), torch.zeros_like(W2),
                           torch.zeros(batch, self.d_model, device=device, dtype=dtype))

    def initial_state_differentiable(self, batch: int) -> MemoryState:
        """Like :meth:`init_state` but keeps the graph to ``M0`` (outer training learns ``M0``)."""
        W1 = self.W1_0.unsqueeze(0).expand(batch, -1, -1)
        W2 = self.W2_0.unsqueeze(0).expand(batch, -1, -1)
        return MemoryState(W1, W2, torch.zeros_like(W1), torch.zeros_like(W2),
                           W1.new_zeros(batch, self.d_model))

    # ------------------------------------------------------------------- read
    def read(self, state: MemoryState) -> torch.Tensor:
        """Returns memory tokens ``[B, read_tokens, d_model]`` from completed state only."""
        cdt = self.q_proj.weight.dtype
        summary = state.summary.to(cdt)
        q = self.slots.unsqueeze(0) + self.q_proj(self.in_norm(summary)).unsqueeze(1)
        q = F.normalize(q.float(), dim=-1)
        r = mlp_forward(state.W1.float(), state.W2.float(), q)
        return self.out_norm(self.o_proj(r.to(cdt)))

    # ----------------------------------------------------------------- update
    def keys_values(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hn = self.in_norm(h)
        k = F.normalize(self.k_proj(hn).float(), dim=-1)
        v = F.normalize(self.v_proj(hn).float(), dim=-1) * (self.d_mem**0.5)  # unit-RMS targets (no collapse)
        return k, v

    def update(
        self, state: MemoryState, h: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> tuple[MemoryState, dict[str, torch.Tensor]]:
        """One bounded update from a segment of anchor hidden states ``h [B, T, d]``.

        The caller detaches ``h`` (§6: each anchor updates from its own detached
        segment). Gradients still flow to projections, gates, and ``M0`` so that
        truncated backpropagation through a few segments trains the memory.
        """
        if state.frozen:
            raise MemoryFrozenError("neural memory is read-only during generation")
        cfg = self.cfg
        k, v = self.keys_values(h)
        W1, W2 = state.W1.float(), state.W2.float()
        loss, g1, g2 = assoc_grad_closed_form(W1, W2, k, v, valid)
        g1, g2 = clip_by_norm(g1, cfg.grad_clip), clip_by_norm(g2, cfg.grad_clip)
        if valid is None:
            summary = h.mean(dim=1)
        else:
            vm = valid.to(h.dtype).unsqueeze(-1)
            summary = (h * vm).sum(1) / vm.sum(1).clamp(min=1.0)
        gl = self.gate(self.in_norm(summary)).float()
        eta = cfg.max_eta * torch.sigmoid(gl[:, 0]).view(-1, 1, 1)
        theta = cfg.max_theta * torch.sigmoid(gl[:, 1]).view(-1, 1, 1)
        alpha = cfg.max_alpha * torch.sigmoid(gl[:, 2]).view(-1, 1, 1)
        S1 = clip_by_norm(eta * state.S1.float() - theta * g1, cfg.grad_clip)
        S2 = clip_by_norm(eta * state.S2.float() - theta * g2, cfg.grad_clip)
        if cfg.decay_target == "init":
            M1, M2 = self.W1_0.to(W1.dtype).unsqueeze(0), self.W2_0.to(W2.dtype).unsqueeze(0)
            new_W1 = M1 + (1 - alpha) * (W1 - M1) + S1
            new_W2 = M2 + (1 - alpha) * (W2 - M2) + S2
        else:
            new_W1 = (1 - alpha) * W1 + S1
            new_W2 = (1 - alpha) * W2 + S2
        dt = state.W1.dtype
        new_state = MemoryState(new_W1.to(dt), new_W2.to(dt), S1.to(dt), S2.to(dt), summary.to(state.summary.dtype),
                                state.updates + 1, False, dict(state.meta))
        info = {
            "surprise": loss,
            "grad_norm": torch.sqrt(g1.flatten(1).pow(2).sum(-1) + g2.flatten(1).pow(2).sum(-1)),
            "eta": eta.flatten(), "theta": theta.flatten(), "alpha": alpha.flatten(),
        }
        return new_state, info

    def retrieval_loss(self, state: MemoryState, h: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Associative loss of the *current* memory on a segment (L_memory, predictive form)."""
        k, v = self.keys_values(h)
        return assoc_loss(state.W1.float(), state.W2.float(), k, v, valid)
