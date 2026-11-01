"""Release gates (blueprint §13, §16).

Non-negotiable correctness/security gates block release unconditionally.
Product gates are tunable thresholds proposed by the blueprint for this project
(not external benchmark facts) and finalised after the pilot establishes a
credible baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

NON_NEGOTIABLE = (
    "zero_token_drops",
    "zero_cross_tenant_reads",
    "zero_unauthorized_tool_executions",
    "reproducible_checkpoint_resume",
    "verified_deletion",
    "memory_off_path_functional",
)


@dataclass(frozen=True)
class MemoryPromotionGate:
    """Promote neural memory only with a real long-session gain at acceptable cost (§13)."""

    min_session_gain_points: float = 3.0  # on the 128K/256K session suite vs memory-off
    max_latency_increase: float = 0.10  # end-to-end, target cohort
    max_short_context_regression_points: float = 0.5

    def evaluate(self, *, session_on: float, session_off: float, latency_on_ms: float, latency_off_ms: float,
                 short_on: float, short_off: float, security_unauthorized_actions: int) -> list[str]:
        reasons = []
        if session_on - session_off < self.min_session_gain_points:
            reasons.append(f"session gain {session_on - session_off:.2f} < {self.min_session_gain_points}")
        if latency_off_ms > 0 and (latency_on_ms / latency_off_ms - 1.0) > self.max_latency_increase:
            reasons.append(f"latency +{100 * (latency_on_ms / latency_off_ms - 1):.1f}% > {100 * self.max_latency_increase:.0f}%")
        if short_off - short_on > self.max_short_context_regression_points:
            reasons.append(f"short-context regression {short_off - short_on:.2f} > {self.max_short_context_regression_points}")
        if security_unauthorized_actions:
            reasons.append(f"{security_unauthorized_actions} unauthorized actions in the security suite")
        return reasons


@dataclass
class GateReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    memory_promoted: Optional[bool] = None

    def to_markdown(self) -> str:
        head = "PASS" if self.passed else "BLOCKED"
        lines = [f"**Release gates: {head}**"]
        lines += [f"- {f}" for f in self.failures]
        if self.memory_promoted is not None:
            lines.append(f"- neural memory {'promoted' if self.memory_promoted else 'ships disabled (memory-off core)'}")
        return "\n".join(lines)


def evaluate_release(results: dict[str, bool], memory_metrics: Optional[dict] = None,
                     memory_gate: MemoryPromotionGate = MemoryPromotionGate()) -> GateReport:
    """``results`` maps every non-negotiable gate name to True (passed) / False.

    A missing gate counts as a failure. Memory failing its product gate does not
    block release: the memory-off core ships instead (§12, §15).
    """
    failures = [f"non-negotiable gate failed or missing: {g}" for g in NON_NEGOTIABLE if not results.get(g, False)]
    promoted = None
    if memory_metrics is not None:
        promoted = not memory_gate.evaluate(**memory_metrics)
    return GateReport(not failures, failures, promoted)
