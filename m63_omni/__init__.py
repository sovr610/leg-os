"""M63-Omni: reference implementation of the mid-tier hybrid multimodal orchestrator blueprint.

Subpackages follow the blueprint's repository layout (§11):

* ``model``    — blocks, encoders, memory, MTP, parameter manifest
* ``kernels``  — GDN, attention, grouped MoE, memory-update kernels (+ naive references)
* ``data``     — manifests, filters, alignment, packing, tokenizer
* ``train``    — objectives, schedules, training loop, checkpoints
* ``runtime``  — hybrid cache, sessions, branches, event log, capacity planning
* ``agent``    — action schemas, control plane, verifiers, escalation
* ``eval``     — harness, release gates, security and long-context suites
"""

from .config import M63Config, load_config, reference, tiny

__all__ = ["M63Config", "load_config", "reference", "tiny"]
__version__ = "0.1.0"
