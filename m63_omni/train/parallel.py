"""Distributed layout validation (blueprint §11).

The reference uses Megatron-Core for tensor (TP), pipeline (PP), context (CP),
data (DP), and expert (EP) parallelism. This module validates a proposed layout
against the model shape *before* a job is launched; it does not launch one.
Megatron's MoE "parallel folding" lets expert groups re-partition the
``TP x CP x DP`` ranks of each pipeline stage: ``EP x ETP x EDP == TP x CP x DP``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import M63Config


@dataclass(frozen=True)
class ParallelLayout:
    tp: int
    pp: int
    cp: int
    dp: int
    ep: int
    etp: int = 0  # expert tensor parallel; 0 means "same as tp"
    gpus_per_node: int = 8

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.cp * self.dp

    @property
    def expert_tp(self) -> int:
        return self.etp or self.tp

    @property
    def expert_dp(self) -> int:
        return (self.tp * self.cp * self.dp) // (self.ep * self.expert_tp)


SHORT_CONTEXT_256 = ParallelLayout(tp=4, pp=4, cp=1, dp=16, ep=8)  # EP8 x ETP4 x EDP2 = 64 ranks per stage
LONG_CONTEXT_256 = ParallelLayout(tp=4, pp=4, cp=4, dp=4, ep=4)


def validate_layout(cfg: M63Config, layout: ParallelLayout, world_size: int | None = None) -> list[str]:
    """Return a list of problems (empty when the layout is consistent)."""
    errs: list[str] = []
    if world_size is not None and layout.world_size != world_size:
        errs.append(f"TPxPPxCPxDP = {layout.world_size} != world size {world_size}")
    if (layout.tp * layout.cp * layout.dp) % (layout.ep * layout.expert_tp):
        errs.append("EP x ETP must divide TP x CP x DP (MoE parallel folding)")
    if cfg.moe.routed_experts % layout.ep:
        errs.append(f"routed_experts {cfg.moe.routed_experts} not divisible by EP {layout.ep}")
    a = cfg.attention
    if a.kv_heads % layout.tp and layout.tp % a.kv_heads:
        errs.append("TP must divide kv_heads (or vice versa)")
    if a.query_heads % layout.tp or cfg.gdn.heads % layout.tp:
        errs.append("TP must divide query and GDN heads")
    group = len(cfg.model.hybrid_pattern)
    if cfg.model.layers % layout.pp:
        errs.append("PP must divide the number of layers")
    elif (cfg.model.layers // layout.pp) % group:
        errs.append(f"pipeline stages must hold whole {group}-block hybrid groups")
    if cfg.training.seq_len % (2 * layout.cp):
        errs.append("seq_len must be divisible by 2 x CP (load-balanced causal context parallelism)")
    return errs


def nodes_per_expert_group(layout: ParallelLayout) -> int:
    ranks = layout.ep * layout.expert_tp
    return max(1, -(-ranks // layout.gpus_per_node))


def experts_per_node(cfg: M63Config, layout: ParallelLayout) -> int:
    """Experts resident per physical node for node-limited routing."""
    return cfg.moe.routed_experts // nodes_per_expert_group(layout)
