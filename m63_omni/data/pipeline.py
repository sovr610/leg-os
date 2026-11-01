"""The ten mandatory data transformations as an ordered stage pipeline (blueprint §8).

§8: "The data pipeline has ten mandatory transformations: inventory and
provenance; language/modality identification; exact and near deduplication;
quality filtering; safety and privacy filtering; benchmark decontamination;
media decoding and integrity checks; timestamp/alignment construction;
tokenization and packing; and immutable manifest generation."

Each transformation is a :class:`Stage`. Stages see one :class:`Document` at a
time (``process``) and may aggregate at the end (``finalize``). Every stage
produces a :class:`StageReport` with kept / dropped / modified counts and
reasons, and every drop or modification is written to the
:class:`~m63_omni.data.manifest.ProvenanceLedger`, so lineage is queryable.
The final stage emits an immutable :class:`~m63_omni.data.manifest.DataManifest`
whose transformation list records each stage's exact parameters.

Media is not decoded here: stages operate on *declared* metadata (duration,
fps or frame timestamps, sample rate, sizes, a corrupted flag, a content hash)
and caller-supplied embeddings, which is what the reference implementation
needs to exercise every rule in the Appendix B "Multimodal alignment" row
(variable fps, missing channels, bad sync, duplicates, timestamps).
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import numpy as np

from ..common import Modality, content_hash
from .decontam import Decontaminator
from .dedup import ExactDeduplicator, MediaNearDupIndex, NearDuplicateIndex, text_hash
from .filters import QualityConfig, SafetyConfig, identify_language, quality_check, safety_privacy_filter
from .manifest import DataManifest, ManifestEntry, ProvenanceLedger, TransformRecord
from .media import LONG_PROXY, AVChunk, SamplingPolicy, TokenBudgetError, image_tokens, plan_av_chunks, select_policy
from .packing import PackedBatch, TokenizedDoc, pack_documents
from .tokenizer import MODALITY_SPAN_TOKENS, ByteBPETokenizer

__all__ = [
    "MANDATORY_STAGES",
    "TimedSegment",
    "MediaMeta",
    "Document",
    "Decision",
    "StageReport",
    "PipelineContext",
    "Stage",
    "InventoryStage",
    "IdentificationStage",
    "DedupStage",
    "QualityStage",
    "SafetyPrivacyStage",
    "DecontaminationStage",
    "MediaIntegrityStage",
    "AlignmentStage",
    "TokenizePackStage",
    "ManifestStage",
    "Pipeline",
    "PipelineResult",
    "PipelineError",
    "build_default_pipeline",
]

#: The §8 transformations, in their mandatory order.
MANDATORY_STAGES: tuple[str, ...] = (
    "inventory_provenance",
    "language_modality_id",
    "dedup",
    "quality_filter",
    "safety_privacy",
    "decontamination",
    "media_integrity",
    "timestamp_alignment",
    "tokenize_pack",
    "manifest",
)

_TEXTUAL = frozenset({Modality.TEXT, Modality.TOOL})
_TIMED = frozenset({Modality.VIDEO, Modality.AUDIO})
_ENCODER = frozenset({int(Modality.IMAGE), int(Modality.VIDEO), int(Modality.AUDIO)})


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------- documents
@dataclass
class TimedSegment:
    """A caption or transcript span aligned to media time."""

    start_ms: int
    end_ms: int
    text: str
    kind: str = "transcript"


@dataclass
class MediaMeta:
    """Declared media metadata (what a decoder would report)."""

    duration_s: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    fps: float | None = None
    frame_times_ms: tuple[int, ...] | None = None  # variable-frame-rate timestamps
    sample_rate: int | None = None
    audio_channels: int | None = None
    width: int | None = None
    height: int | None = None
    video_duration_s: float | None = None
    audio_duration_s: float | None = None
    corrupted: bool = False
    content_hash: str | None = None  # declared hash of the raw bytes

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        if d["frame_times_ms"] is not None:
            d["frame_times_ms"] = list(d["frame_times_ms"])
        return d


@dataclass
class Document:
    """A unit of training data flowing through the pipeline."""

    doc_id: str
    source_id: str
    modality: Modality
    text: str | None = None
    media: MediaMeta | None = None
    source_event: str | None = None  # holdout grouping key (§8)
    segments: list[TimedSegment] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # (n, d) frame/clip/audio embeddings for near-dup
    kind: str = "prose"  # "prose" | "code" (quality rules)
    synthetic: bool = False  # synthetic caption / data (§8: labelled as such)
    grounding_score: float | None = None  # temporal/visual grounding of a synthetic caption
    # ---- filled in by stages
    language: str | None = None
    source_version: str | None = None
    raw_hash: str | None = None
    av_chunks: list[AVChunk] = field(default_factory=list)
    n_image_tokens: int = 0
    tokens: list[int] | None = None
    modality_ids: list[int] | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = Modality(self.modality)

    def all_text(self) -> str:
        parts = [self.text or ""] + [s.text for s in self.segments]
        return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------- stages
@dataclass(frozen=True)
class Decision:
    """A stage's verdict on one document."""

    action: str  # "keep" | "modify" | "drop"
    reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def keep(*flags: str) -> "Decision":
        return Decision("keep", (), tuple(flags))

    @staticmethod
    def modify(*reasons: str, flags: Sequence[str] = ()) -> "Decision":
        return Decision("modify", tuple(reasons), tuple(flags))

    @staticmethod
    def drop(reason: str, **detail: Any) -> "Decision":
        return Decision("drop", (reason,), (), detail)


@dataclass
class StageReport:
    """Counts and reasons for one stage."""

    stage: str
    n_in: int = 0
    kept: int = 0
    dropped: int = 0
    modified: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    modify_reasons: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    dropped_ids: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_out(self) -> int:
        return self.kept

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "n_in": self.n_in,
            "kept": self.kept,
            "dropped": self.dropped,
            "modified": self.modified,
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
            "modify_reasons": dict(sorted(self.modify_reasons.items())),
            "flags": dict(sorted(self.flags.items())),
        }


@dataclass
class PipelineContext:
    ledger: ProvenanceLedger
    seed: int
    transformations: tuple[TransformRecord, ...] = ()
    reports: list[StageReport] = field(default_factory=list)
    packed: PackedBatch | None = None
    manifest: DataManifest | None = None


class Stage:
    """Base class: override :meth:`process` (per document) and optionally
    :meth:`setup` (fresh per-run state) and :meth:`finalize` (aggregate)."""

    name: ClassVar[str] = "stage"
    version: ClassVar[str] = "1"

    def params(self) -> dict[str, Any]:
        return {}

    def transform_record(self) -> TransformRecord:
        return TransformRecord.create(self.name, self.params(), self.version)

    def setup(self, ctx: PipelineContext) -> None:
        pass

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        return Decision.keep()

    def finalize(self, docs: list[Document], ctx: PipelineContext) -> None:
        pass

    def run(self, docs: list[Document], ctx: PipelineContext) -> tuple[list[Document], StageReport]:
        self.setup(ctx)
        report = StageReport(self.name, n_in=len(docs))
        drop_r: Counter[str] = Counter()
        mod_r: Counter[str] = Counter()
        flags: Counter[str] = Counter()
        out: list[Document] = []
        for doc in docs:
            d = self.process(doc, ctx)
            for f in d.flags:
                flags[f] += 1
                ctx.ledger.record_decision(doc.doc_id, doc.source_id, self.name, "flagged", f)
            if d.action == "drop":
                reason = d.reasons[0] if d.reasons else "unspecified"
                report.dropped += 1
                drop_r[reason] += 1
                report.dropped_ids.append((doc.doc_id, reason))
                ctx.ledger.record_decision(doc.doc_id, doc.source_id, self.name, "dropped", reason, dict(d.detail))
                continue
            if d.action == "modify":
                report.modified += 1
                mod_r.update(d.reasons)
                ctx.ledger.record_decision(doc.doc_id, doc.source_id, self.name, "modified", ",".join(d.reasons))
            elif d.action != "keep":
                raise PipelineError(f"stage {self.name}: unknown action {d.action!r}")
            out.append(doc)
        report.kept = len(out)
        report.drop_reasons = dict(drop_r)
        report.modify_reasons = dict(mod_r)
        report.flags = dict(flags)
        self.finalize(out, ctx)
        return out, report


# 1 ----------------------------------------------------------------- inventory
class InventoryStage(Stage):
    """Inventory and provenance: every document must come from a registered
    source whose rights are cleared, with consent and no takedown (§8)."""

    name = "inventory_provenance"

    def setup(self, ctx: PipelineContext) -> None:
        self._seen: set[str] = set()

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        if doc.doc_id in self._seen:
            return Decision.drop("duplicate_doc_id")
        self._seen.add(doc.doc_id)
        if doc.source_id not in ctx.ledger:
            return Decision.drop("unknown_source")
        rec = ctx.ledger.get(doc.source_id)
        reasons = rec.ineligibility_reasons()
        if reasons:
            return Decision.drop(reasons[0], all_reasons=reasons)
        doc.source_version = rec.version
        doc.raw_hash = content_hash(
            {
                "text": doc.text,
                "segments": [s.__dict__ for s in doc.segments],
                "media": doc.media.content_hash if doc.media and doc.media.content_hash else (doc.media.to_dict() if doc.media else None),
            }
        )
        ctx.ledger.record_decision(doc.doc_id, doc.source_id, self.name, "admitted", "", {"source_version": rec.version})
        if rec.synthetic and not doc.synthetic:
            doc.synthetic = True
            return Decision.modify("labelled_synthetic")
        return Decision.keep()


# 2 ------------------------------------------------------------ identification
class IdentificationStage(Stage):
    """Language and modality identification. Declared modality must match the
    payload; MEMORY positions are never produced by the data loader."""

    name = "language_modality_id"

    def __init__(self, allowed_languages: Iterable[str] | None = None) -> None:
        self.allowed_languages = frozenset(allowed_languages) if allowed_languages is not None else None

    def params(self) -> dict[str, Any]:
        return {"allowed_languages": sorted(self.allowed_languages) if self.allowed_languages is not None else None,
                "lid": "script_stub_v1"}

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        m, media = doc.modality, doc.media
        if m is Modality.MEMORY:
            return Decision.drop("memory_modality_not_loadable")
        if m in _TEXTUAL:
            if not doc.text or not doc.text.strip():
                return Decision.drop("modality_mismatch", expected="text")
        elif m is Modality.IMAGE:
            if media is None or media.width is None or media.height is None:
                return Decision.drop("modality_mismatch", expected="image size")
        elif m is Modality.VIDEO:
            if media is None or not media.has_video:
                return Decision.drop("modality_mismatch", expected="video channel")
        elif m is Modality.AUDIO:
            if media is None or not media.has_audio:
                return Decision.drop("modality_mismatch", expected="audio channel")
            if media.has_video:
                return Decision.drop("modality_mismatch", expected="audio only; declare VIDEO for AV")
        text = doc.all_text()
        if text:
            doc.language = identify_language(text).language
            if self.allowed_languages is not None and doc.language not in self.allowed_languages:
                return Decision.drop("language_not_allowed", language=doc.language)
        return Decision.keep()


# 3 --------------------------------------------------------------------- dedup
class DedupStage(Stage):
    """Exact + near dedup. Text: normalised SHA-256 then MinHash/LSH. Media:
    declared content hash then embedding cosine (§8). First occurrence wins."""

    name = "dedup"

    def __init__(self, near_threshold: float = 0.8, num_perm: int = 128, ngram: int = 5,
                 media_threshold: float = 0.95, media_min_match_fraction: float = 0.5, seed: int = 0) -> None:
        self.near_threshold = near_threshold
        self.num_perm = num_perm
        self.ngram = ngram
        self.media_threshold = media_threshold
        self.media_min_match_fraction = media_min_match_fraction
        self.seed = seed

    def params(self) -> dict[str, Any]:
        return {"text_exact": "sha256(nfkc_casefold_ws)", "near_threshold": self.near_threshold, "num_perm": self.num_perm,
                "ngram": self.ngram, "media_exact": "declared_sha256", "media_threshold": self.media_threshold,
                "media_min_match_fraction": self.media_min_match_fraction, "seed": self.seed}

    def setup(self, ctx: PipelineContext) -> None:
        self._exact = ExactDeduplicator()
        self._near = NearDuplicateIndex(self.near_threshold, self.num_perm, self.ngram, "char", self.seed)
        self._media = MediaNearDupIndex(threshold=self.media_threshold, seed=self.seed)

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        if doc.modality in _TEXTUAL:
            assert doc.text is not None
            first = self._exact.add(doc.doc_id, text_hash(doc.text))
            if first is not None:
                return Decision.drop("exact_duplicate", original=first)
            hit = self._near.check_and_add(doc.doc_id, doc.text)
            if hit is not None:
                return Decision.drop("near_duplicate", original=hit[0], similarity=round(hit[1], 4))
            return Decision.keep()
        if doc.media is not None and doc.media.content_hash:
            first = self._exact.add(doc.doc_id, "media:" + doc.media.content_hash)
            if first is not None:
                return Decision.drop("exact_duplicate", original=first)
        if doc.embeddings is not None:
            match = self._media.check_and_add(doc.doc_id, doc.embeddings, min_match_fraction=self.media_min_match_fraction)
            if match is not None:
                return Decision.drop("near_duplicate", original=match.item_id, similarity=round(match.max_similarity, 4))
        return Decision.keep()


# 4 ------------------------------------------------------------------- quality
class QualityStage(Stage):
    """Heuristic quality filter for text; synthetic captions must be grounded (§8)."""

    name = "quality_filter"

    def __init__(self, config: QualityConfig | None = None, min_grounding: float = 0.5) -> None:
        self.config = config or QualityConfig()
        self.min_grounding = min_grounding

    def params(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "min_grounding": self.min_grounding}

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        if doc.modality in _TEXTUAL:
            assert doc.text is not None
            res = quality_check(doc.text, self.config, kind=doc.kind)
            if not res.passed:
                return Decision.drop(f"quality:{res.reasons[0]}", all_reasons=list(res.reasons))
            return Decision.keep()
        if doc.synthetic and (doc.text or doc.segments):
            if doc.grounding_score is None or doc.grounding_score < self.min_grounding:
                return Decision.drop("ungrounded_synthetic_caption", grounding_score=doc.grounding_score)
        return Decision.keep()


# 5 -------------------------------------------------------------------- safety
class SafetyPrivacyStage(Stage):
    """PII redaction and safety filtering over text, captions, and transcripts."""

    name = "safety_privacy"

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def params(self) -> dict[str, Any]:
        return self.config.to_dict()

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        reasons: list[str] = []
        flags: list[str] = []
        targets: list[TimedSegment | None] = []
        if doc.text:
            targets.append(None)
        targets.extend(doc.segments)
        for seg in targets:
            original = (doc.text or "") if seg is None else seg.text
            res = safety_privacy_filter(original, self.config)
            if not res.keep:
                return Decision.drop(res.reasons[0], fired=res.fired)
            if res.modified:
                if seg is None:
                    doc.text = res.text
                else:
                    seg.text = res.text
                reasons.extend(f"redacted_{f.kind}" for f in res.findings)
            if res.fired.get("special_token_literal"):
                flags.append("special_token_literal")
        if reasons:
            return Decision.modify(*sorted(set(reasons)), flags=sorted(set(flags)))
        return Decision.keep(*sorted(set(flags)))


# 6 ----------------------------------------------------------- decontamination
class DecontaminationStage(Stage):
    """Drop training items overlapping evaluation holdouts: held-out source
    events, n-gram overlap with captions/transcripts/text, near-dup frames (§8)."""

    name = "decontamination"

    def __init__(self, decontaminator: Decontaminator) -> None:
        self.decontaminator = decontaminator

    def params(self) -> dict[str, Any]:
        return self.decontaminator.params()

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        dec = self.decontaminator
        ev = dec.check_event(doc.source_event)
        if ev.contaminated:
            return Decision.drop("holdout_event", event=doc.source_event)
        text = doc.all_text()
        if text:
            rep = dec.check_text(text)
            if rep.contaminated:
                return Decision.drop("contaminated_text", holdout_ids=list(rep.holdout_ids), n_overlapping=rep.n_overlapping)
        if doc.embeddings is not None:
            rep = dec.check_media(doc.embeddings)
            if rep.contaminated:
                return Decision.drop("contaminated_media", holdout_ids=list(rep.holdout_ids),
                                     max_similarity=round(rep.max_similarity, 4))
        return Decision.keep()


# 7 ----------------------------------------------------------- media integrity
class MediaIntegrityStage(Stage):
    """Media decoding and integrity checks on declared metadata (§8, App. B)."""

    name = "media_integrity"

    def __init__(self, target_sample_rate: int = 16000, max_fps: float = 240.0, max_gap_s: float = 2.0,
                 max_duration_s: float | None = None) -> None:
        self.target_sample_rate = target_sample_rate
        self.max_fps = max_fps
        self.max_gap_s = max_gap_s
        self.max_duration_s = max_duration_s

    def params(self) -> dict[str, Any]:
        return {"target_sample_rate": self.target_sample_rate, "max_fps": self.max_fps,
                "max_gap_s": self.max_gap_s, "max_duration_s": self.max_duration_s}

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        if doc.modality in _TEXTUAL:
            return Decision.keep()
        m = doc.media
        if m is None:
            return Decision.drop("missing_media_metadata")
        if m.corrupted:
            return Decision.drop("corrupted")
        reasons: list[str] = []
        flags: list[str] = []
        if doc.modality is Modality.IMAGE:
            if not (m.width and m.height and m.width > 0 and m.height > 0):
                return Decision.drop("invalid_size")
            return Decision.keep()
        if not (math.isfinite(m.duration_s) and m.duration_s > 0):
            return Decision.drop("invalid_duration")
        if self.max_duration_s is not None and m.duration_s > self.max_duration_s:
            return Decision.drop("too_long")
        duration_ms = int(round(m.duration_s * 1000))
        if m.has_video:
            if (m.width is not None and m.width <= 0) or (m.height is not None and m.height <= 0):
                return Decision.drop("invalid_size")
            if m.frame_times_ms is not None:
                times = [int(t) for t in m.frame_times_ms]
                if not times:
                    return Decision.drop("no_frames")
                if any(b < a for a, b in zip(times, times[1:])):
                    return Decision.drop("non_monotonic_timestamps")
                if times[0] < 0:
                    return Decision.drop("timestamp_out_of_range")
                dedup = sorted(set(times))
                if len(dedup) != len(times):
                    reasons.append("duplicate_frames_removed")
                kept = [t for t in dedup if t < duration_ms]
                if len(kept) != len(dedup):
                    reasons.append("frames_beyond_duration_trimmed")
                if not kept:
                    return Decision.drop("no_frames")
                if any(b - a > self.max_gap_s * 1000 for a, b in zip(kept, kept[1:])):
                    flags.append("timestamp_gap")
                m.frame_times_ms = tuple(kept)
            else:
                if m.fps is None:
                    return Decision.drop("missing_fps")
                if not (0 < m.fps <= self.max_fps):
                    return Decision.drop("invalid_fps")
        if m.has_audio:
            if m.sample_rate is None or m.sample_rate <= 0:
                return Decision.drop("invalid_sample_rate")
            if m.sample_rate != self.target_sample_rate:
                doc.notes["original_sample_rate"] = m.sample_rate
                m.sample_rate = self.target_sample_rate
                reasons.append(f"resampled_to_{self.target_sample_rate}")
            if m.audio_channels is not None and m.audio_channels > 1:
                doc.notes["original_channels"] = m.audio_channels
                m.audio_channels = 1
                reasons.append("downmixed_to_mono")
        if reasons:
            return Decision.modify(*reasons, flags=flags)
        return Decision.keep(*flags)


# 8 ----------------------------------------------------------------- alignment
class AlignmentStage(Stage):
    """Timestamp/alignment construction: AV sync check, segment validation,
    explicit token-budget policy selection, and interleaved chunk planning (§5)."""

    name = "timestamp_alignment"

    def __init__(self, policy: SamplingPolicy = LONG_PROXY, chunk_seconds: float = 4.0, sync_tolerance_s: float = 0.5,
                 max_positions: int | None = None) -> None:
        self.policy = policy
        self.chunk_seconds = chunk_seconds
        self.sync_tolerance_s = sync_tolerance_s
        self.max_positions = max_positions

    def params(self) -> dict[str, Any]:
        return {"policy": self.policy.to_dict(), "chunk_seconds": self.chunk_seconds,
                "sync_tolerance_s": self.sync_tolerance_s, "max_positions": self.max_positions}

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        if doc.modality in _TEXTUAL:
            return Decision.keep()
        m = doc.media
        assert m is not None
        if doc.modality is Modality.IMAGE:
            doc.n_image_tokens = image_tokens(int(m.width or 0), int(m.height or 0), self.policy)
            return Decision.modify("image_tokens_planned")
        reasons: list[str] = []
        if m.has_video and m.has_audio and m.video_duration_s is not None and m.audio_duration_s is not None:
            if abs(m.video_duration_s - m.audio_duration_s) > self.sync_tolerance_s:
                return Decision.drop("av_desync", video_s=m.video_duration_s, audio_s=m.audio_duration_s)
        duration_ms = int(round(m.duration_s * 1000))
        tol_ms = int(self.sync_tolerance_s * 1000)
        for seg in doc.segments:
            if not (0 <= seg.start_ms < seg.end_ms <= duration_ms + tol_ms):
                return Decision.drop("segment_out_of_range", start_ms=seg.start_ms, end_ms=seg.end_ms)
        order = sorted(range(len(doc.segments)), key=lambda i: (doc.segments[i].start_ms, i))
        if order != list(range(len(doc.segments))):
            doc.segments = [doc.segments[i] for i in order]
            reasons.append("segments_sorted")
        if self.max_positions is not None:
            try:
                select_policy(m.duration_s, self.max_positions, (self.policy,), chunk_seconds=self.chunk_seconds,
                              text_allowance=0, has_video=m.has_video, has_audio=m.has_audio)
            except TokenBudgetError as exc:
                return Decision.drop("exceeds_token_budget", message=str(exc))
        doc.av_chunks = plan_av_chunks(
            doc.doc_id, m.duration_s, self.policy, self.chunk_seconds, m.has_video, m.has_audio,
            source_fps=m.fps if m.frame_times_ms is None else None,
            frame_times_ms=m.frame_times_ms,
        )
        if not doc.av_chunks:
            return Decision.drop("no_media_tokens")
        reasons.append("av_chunks_planned")
        return Decision.modify(*reasons)


# 9 ------------------------------------------------------------ tokenize/pack
class TokenizePackStage(Stage):
    """Tokenisation (injection-safe: raw text never parses special tokens) and
    packing into fixed-length rows with boundary masks and reset flags (§8)."""

    name = "tokenize_pack"

    def __init__(self, tokenizer: ByteBPETokenizer, seq_len: int, *, strategy: str = "first_fit",
                 mask_media_targets: bool = True) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.strategy = strategy
        self.mask_media_targets = mask_media_targets

    def params(self) -> dict[str, Any]:
        return {"tokenizer": self.tokenizer.fingerprint, "vocab_size": self.tokenizer.vocab_size,
                "special_tokens": self.tokenizer.special_tokens, "seq_len": self.seq_len, "strategy": self.strategy,
                "mask_media_targets": self.mask_media_targets, "allow_special_in_raw_text": False}

    def _text(self, text: str, mod: int, toks: list[int], mods: list[int]) -> None:
        ids = self.tokenizer.encode(text)
        toks.extend(ids)
        mods.extend([mod] * len(ids))

    def _sid(self, name: str) -> int:
        return self.tokenizer.special_id(name)

    def tokenize(self, doc: Document) -> tuple[list[int], list[int]]:
        t = self.tokenizer
        TEXT = int(Modality.TEXT)
        toks: list[int] = [t.bos_id]
        mods: list[int] = [TEXT]
        if doc.modality in _TEXTUAL:
            self._text(doc.text or "", int(doc.modality), toks, mods)
        elif doc.modality is Modality.IMAGE:
            open_, close = MODALITY_SPAN_TOKENS[Modality.IMAGE]
            toks += [self._sid(open_)] + [self._sid("<|media|>")] * doc.n_image_tokens + [self._sid(close)]
            mods += [TEXT] + [int(Modality.IMAGE)] * doc.n_image_tokens + [TEXT]
            if doc.text:
                self._text(doc.text, TEXT, toks, mods)
        else:
            open_, close = MODALITY_SPAN_TOKENS[doc.modality]
            toks.append(self._sid(open_))
            mods.append(TEXT)
            windows: dict[int, list[AVChunk]] = defaultdict(list)
            for c in doc.av_chunks:
                windows[c.window].append(c)
            segs = list(doc.segments)
            si = 0
            media_id = self._sid("<|media|>")
            markers = [self._sid("<|av_chunk|>"), self._sid("<|ts|>")]
            for w in sorted(windows):
                chunks = windows[w]
                end = chunks[0].end_ms
                toks += markers
                mods += [TEXT] * len(markers)
                for c in chunks:  # planner order: video then audio
                    toks += [media_id] * c.n_tokens
                    mods += [int(c.modality)] * c.n_tokens
                while si < len(segs) and segs[si].start_ms < end:
                    self._text(segs[si].text, TEXT, toks, mods)
                    si += 1
            for seg in segs[si:]:
                self._text(seg.text, TEXT, toks, mods)
            toks.append(self._sid(close))
            mods.append(TEXT)
            if doc.text:
                self._text(doc.text, TEXT, toks, mods)
        toks.append(t.eos_id)
        mods.append(TEXT)
        return toks, mods

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        doc.tokens, doc.modality_ids = self.tokenize(doc)
        return Decision.modify("tokenized")

    def finalize(self, docs: list[Document], ctx: PipelineContext) -> None:
        tokenized = [TokenizedDoc(d.doc_id, tuple(d.tokens or ()), tuple(d.modality_ids or ())) for d in docs]
        ctx.packed = pack_documents(
            tokenized,
            self.seq_len,
            pad_id=self.tokenizer.pad_id,
            seed=ctx.seed,
            strategy=self.strategy,
            ignore_target_modalities=_ENCODER if self.mask_media_targets else (),
        )


# 10 ---------------------------------------------------------------- manifest
class ManifestStage(Stage):
    """Immutable manifest generation (§8). Re-checks rights at emission time and
    lets :class:`DataManifest` refuse anything ineligible."""

    name = "manifest"

    def __init__(self, batch_size: int = 1, drop_last: bool = False) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last

    def params(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size, "drop_last": self.drop_last, "ordering": "sha256_permutation"}

    def process(self, doc: Document, ctx: PipelineContext) -> Decision:
        rec = ctx.ledger.get(doc.source_id)
        if not rec.eligible:
            return Decision.drop("rights_revoked_before_manifest", reasons=rec.ineligibility_reasons())
        if rec.version != doc.source_version:
            return Decision.drop("source_version_changed")
        return Decision.keep()

    def finalize(self, docs: list[Document], ctx: PipelineContext) -> None:
        entries = [
            ManifestEntry(
                doc_id=d.doc_id,
                source_id=d.source_id,
                source_version=d.source_version or "",
                content_hash=content_hash({"tokens": d.tokens, "modality_ids": d.modality_ids,
                                           "media": d.media.content_hash if d.media else None}),
                modality=int(d.modality),
                n_tokens=len(d.tokens or ()),
            )
            for d in docs
        ]
        used = sorted({d.source_id for d in docs})
        stats = {r.stage: r.to_dict() for r in ctx.reports}
        if ctx.packed is not None:
            stats["packing"] = {"rows": ctx.packed.batch_size, "seq_len": ctx.packed.seq_len,
                                "tokens": ctx.packed.n_tokens, "targets": ctx.packed.n_targets,
                                "continuation_chunks": len(ctx.packed.continuations),
                                "skipped_empty": list(ctx.packed.skipped_empty)}
        ctx.manifest = DataManifest.build(
            [ctx.ledger.get(s) for s in used],
            ctx.transformations,
            ctx.seed,
            entries,
            ordering={"batch_size": self.batch_size, "drop_last": self.drop_last},
            stats=stats,
        )


# -------------------------------------------------------------------- pipeline
@dataclass
class PipelineResult:
    documents: list[Document]
    reports: list[StageReport]
    manifest: DataManifest
    packed: PackedBatch | None
    ledger: ProvenanceLedger

    def report(self, stage: str) -> StageReport:
        for r in self.reports:
            if r.stage == stage:
                return r
        raise KeyError(stage)

    def summary(self) -> dict[str, Any]:
        return {"stages": [r.to_dict() for r in self.reports], "manifest_hash": self.manifest.manifest_hash,
                "documents": len(self.documents)}


class Pipeline:
    """An ordered, composable sequence of stages.

    With ``strict=True`` (default) the ten §8 transformations must all be
    present, exactly once, in the mandatory order; additional custom stages may
    be interleaved. The last stage must be the manifest stage.
    """

    def __init__(self, stages: Sequence[Stage], *, seed: int = 0, strict: bool = True) -> None:
        self.stages = list(stages)
        self.seed = int(seed)
        self.strict = strict
        self.validate()

    def validate(self) -> None:
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise PipelineError(f"duplicate stage names in {names}")
        if not self.strict:
            return
        missing = [n for n in MANDATORY_STAGES if n not in names]
        if missing:
            raise PipelineError(f"missing mandatory stages {missing}")
        positions = [names.index(n) for n in MANDATORY_STAGES]
        if positions != sorted(positions):
            raise PipelineError(f"mandatory stages out of order; required order is {list(MANDATORY_STAGES)}")
        if names[-1] != "manifest":
            raise PipelineError("the manifest stage must be last")

    def run(self, documents: Iterable[Document], ledger: ProvenanceLedger) -> PipelineResult:
        """Run every stage in order on deep copies of ``documents``."""
        docs = [copy.deepcopy(d) for d in documents]
        ctx = PipelineContext(ledger=ledger, seed=self.seed,
                              transformations=tuple(s.transform_record() for s in self.stages))
        for stage in self.stages:
            docs, report = stage.run(docs, ctx)
            ctx.reports.append(report)
        if ctx.manifest is None:
            raise PipelineError("pipeline finished without emitting a manifest")
        return PipelineResult(docs, list(ctx.reports), ctx.manifest, ctx.packed, ledger)


def build_default_pipeline(
    tokenizer: ByteBPETokenizer,
    decontaminator: Decontaminator | None = None,
    *,
    seq_len: int = 4096,
    seed: int = 0,
    quality: QualityConfig | None = None,
    safety: SafetyConfig | None = None,
    policy: SamplingPolicy = LONG_PROXY,
    chunk_seconds: float = 4.0,
    allowed_languages: Iterable[str] | None = None,
    near_threshold: float = 0.8,
    media_threshold: float = 0.95,
    max_positions: int | None = None,
    batch_size: int = 1,
) -> Pipeline:
    """The ten mandatory §8 transformations with reference defaults."""
    return Pipeline(
        [
            InventoryStage(),
            IdentificationStage(allowed_languages),
            DedupStage(near_threshold=near_threshold, media_threshold=media_threshold, seed=seed),
            QualityStage(quality),
            SafetyPrivacyStage(safety),
            DecontaminationStage(decontaminator if decontaminator is not None else Decontaminator()),
            MediaIntegrityStage(),
            AlignmentStage(policy, chunk_seconds, max_positions=max_positions),
            TokenizePackStage(tokenizer, seq_len),
            ManifestStage(batch_size=batch_size),
        ],
        seed=seed,
    )
