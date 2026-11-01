"""Data engine for M63-Omni (blueprint §5, §8, §12 step 4, §13, Appendix B).

Modules:

* :mod:`.tokenizer` — byte-level BPE, the fixed special-token registry, fertility
* :mod:`.dedup`     — exact (SHA-256) and near (MinHash/LSH, embedding cosine) dedup
* :mod:`.decontam`  — n-gram/media decontamination and source-event holdout splits
* :mod:`.filters`   — quality heuristics, script-based LID stub, PII/safety filtering
* :mod:`.manifest`  — source rights, provenance ledger, immutable data manifests
* :mod:`.pipeline`  — the ten mandatory §8 transformations as ordered stages
* :mod:`.media`     — sampling policies, token budgets, AV chunk interleaving
* :mod:`.mixture`   — Stage A-D curriculum and compute-normalised weights
* :mod:`.packing`   — packed rows with boundary masks and recurrent reset flags
"""

from .decontam import (
    ContaminationReport,
    Decontaminator,
    HoldoutError,
    HoldoutSplit,
    NGramIndex,
    ngram_tokens,
    split_by_source_event,
)
from .dedup import (
    ExactDeduplicator,
    MediaMatch,
    MediaNearDupIndex,
    MinHasher,
    MinHashLSH,
    NearDuplicateIndex,
    bytes_hash,
    cosine_similarity,
    jaccard,
    normalize_text,
    shingles,
    text_hash,
)
from .filters import (
    FilterOutcome,
    LanguageGuess,
    PIIFinding,
    QualityConfig,
    QualityResult,
    RedactionResult,
    SafetyConfig,
    detect_script,
    identify_language,
    luhn_valid,
    quality_check,
    redact_pii,
    safety_privacy_filter,
)
from .manifest import (
    ConsentStatus,
    DataManifest,
    ManifestEntry,
    ManifestError,
    ProvenanceLedger,
    RightsStatus,
    SourceRecord,
    TakedownStatus,
    TransformRecord,
    batch_order,
)
from .media import (
    AV_MARKERS,
    HIGH_DETAIL,
    HIGH_FIDELITY_AUDIO,
    LONG_PROXY,
    POLICIES,
    AVChunk,
    PolicyDecision,
    SamplingPolicy,
    TokenBudgetError,
    TokenEstimate,
    estimate_session_tokens,
    image_tokens,
    interleave_av_tokens,
    plan_av_chunks,
    select_policy,
)
from .mixture import (
    MIXTURE_KEYS,
    STAGES,
    TOTAL_POSITIONS,
    MixtureError,
    StageMixture,
    compute_normalized_weights,
    compute_shares,
    get_stage,
    stage_at,
    stage_position_budget,
    validate_curriculum,
)
from .packing import IGNORE_INDEX, PackedBatch, SegmentInfo, TokenizedDoc, pack_documents, single_document_batch
from .pipeline import (
    MANDATORY_STAGES,
    Decision,
    Document,
    MediaMeta,
    Pipeline,
    PipelineError,
    PipelineResult,
    Stage,
    StageReport,
    TimedSegment,
    build_default_pipeline,
)
from .tokenizer import (
    MODALITY_SPAN_TOKENS,
    NAMED_SPECIAL_TOKENS,
    ROLE_TOKENS,
    SPECIAL_TOKENS,
    TRUST_TOKENS,
    ByteBPETokenizer,
    FertilityStats,
    TokenizerError,
    fertility_report,
    is_special_id,
    pretokenize,
    special_id,
    special_ids,
    special_ids_for_config,
    special_index,
    special_name,
    special_token_names,
    train_bpe,
)

__all__ = [
    # tokenizer
    "SPECIAL_TOKENS", "NAMED_SPECIAL_TOKENS", "ROLE_TOKENS", "MODALITY_SPAN_TOKENS", "TRUST_TOKENS",
    "special_id", "special_ids", "special_ids_for_config", "special_index", "special_name", "special_token_names",
    "is_special_id", "pretokenize", "ByteBPETokenizer", "train_bpe", "FertilityStats", "fertility_report",
    "TokenizerError",
    # dedup
    "normalize_text", "text_hash", "bytes_hash", "ExactDeduplicator", "shingles", "jaccard", "MinHasher",
    "MinHashLSH", "NearDuplicateIndex", "cosine_similarity", "MediaMatch", "MediaNearDupIndex",
    # decontam
    "ngram_tokens", "NGramIndex", "ContaminationReport", "Decontaminator", "HoldoutSplit", "HoldoutError",
    "split_by_source_event",
    # filters
    "QualityConfig", "QualityResult", "quality_check", "detect_script", "LanguageGuess", "identify_language",
    "luhn_valid", "PIIFinding", "RedactionResult", "redact_pii", "SafetyConfig", "FilterOutcome",
    "safety_privacy_filter",
    # manifest
    "RightsStatus", "ConsentStatus", "TakedownStatus", "SourceRecord", "ProvenanceLedger", "TransformRecord",
    "ManifestEntry", "DataManifest", "ManifestError", "batch_order",
    # media
    "SamplingPolicy", "LONG_PROXY", "HIGH_FIDELITY_AUDIO", "HIGH_DETAIL", "POLICIES", "AV_MARKERS", "TokenEstimate",
    "estimate_session_tokens", "PolicyDecision", "TokenBudgetError", "select_policy", "AVChunk", "plan_av_chunks",
    "image_tokens", "interleave_av_tokens",
    # mixture
    "MIXTURE_KEYS", "StageMixture", "STAGES", "TOTAL_POSITIONS", "get_stage", "stage_at",
    "compute_normalized_weights", "compute_shares", "stage_position_budget", "validate_curriculum", "MixtureError",
    # packing
    "IGNORE_INDEX", "TokenizedDoc", "SegmentInfo", "PackedBatch", "pack_documents", "single_document_batch",
    # pipeline
    "MANDATORY_STAGES", "TimedSegment", "MediaMeta", "Document", "Decision", "StageReport", "Stage", "Pipeline",
    "PipelineResult", "PipelineError", "build_default_pipeline",
]
