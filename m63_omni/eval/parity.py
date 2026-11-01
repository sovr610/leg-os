"""Instruction-following parity gate: the omni model vs a text-only control (docs/IMPROVEMENTS.md §1, §4).

Peer omni models regressed on instruction following after multimodal training
(IFEval −3.7 to −7.2 points), and instruction following and tool calling are
this orchestrator's core skills. The gate compares a *candidate* (the
multimodal model) with a *text-only control* — the same configuration with the
multimodal front end disabled, trained on the candidate's text/code stream —
on the same IFEval prompts and BFCL items, scored by the same scorers.

It is a paired **non-inferiority** test. For each metric the difference
``candidate − control`` (in points) gets a one-sided lower confidence bound by
bootstrap over items, and the verdict is:

* ``pass`` — the lower bound is above ``−margin``: the candidate is shown to be
  no more than ``margin`` points worse;
* ``fail`` — the point estimate is already below ``−margin``;
* ``inconclusive`` — neither: the evidence is too thin to decide (more prompts,
  or several sampled responses per prompt, are needed). Only ``pass`` passes.

Metrics: IFEval prompt-level strict (bootstrap over prompts), IFEval
instruction-level strict (cluster bootstrap over prompts), and BFCL overall
(unweighted mean of category accuracies, bootstrap stratified by category). In
addition no single BFCL category may drop by more than ``bfcl_category_points``
(point estimate), so an aggregate cannot hide, e.g., a collapse in declining
irrelevant calls. Both scorers must pass their self-tests first.

The margins are this project's proposal, to be confirmed after the pilot like
the other product gates. At the official set sizes (541 IFEval prompts, ~1,700
BFCL AST items) a 2-point margin needs low disagreement between the two models
to resolve; ``inconclusive`` says so instead of guessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import bfcl, ifeval

VERDICTS = ("pass", "inconclusive", "fail")


@dataclass(frozen=True)
class ParityMargins:
    ifeval_prompt_strict: float = 2.0  # points
    ifeval_instruction_strict: float = 2.0
    bfcl_overall: float = 2.0
    bfcl_category_points: float = 5.0  # largest allowed point-estimate drop in any single BFCL category
    confidence: float = 0.95  # one-sided


@dataclass
class ParityMetric:
    name: str
    candidate: float  # points
    control: float
    diff: float  # candidate - control
    lower: float  # one-sided lower confidence bound on diff
    margin: float
    n: int
    verdict: str = ""

    def __post_init__(self) -> None:
        if not self.verdict:
            if self.lower >= -self.margin:
                self.verdict = "pass"
            elif self.diff < -self.margin:
                self.verdict = "fail"
            else:
                self.verdict = "inconclusive"


@dataclass
class ParityReport:
    metrics: list[ParityMetric] = field(default_factory=list)
    category_diffs: dict[str, float] = field(default_factory=dict)  # BFCL per-category diff, points
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    margins: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.failures or any(m.verdict == "fail" for m in self.metrics):
            return "fail"
        if any(m.verdict == "inconclusive" for m in self.metrics):
            return "inconclusive"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "passed": self.passed, "metrics": [asdict(m) for m in self.metrics],
                "category_diffs": self.category_diffs, "failures": self.failures, "notes": self.notes,
                "margins": self.margins}

    @classmethod
    def from_dict(cls, d: dict) -> "ParityReport":
        return cls([ParityMetric(**m) for m in d["metrics"]], dict(d.get("category_diffs", {})),
                   list(d.get("failures", [])), list(d.get("notes", [])), dict(d.get("margins", {})))

    def to_markdown(self) -> str:
        lines = [f"# Instruction-following parity vs text-only control — {self.verdict.upper()}", "",
                 "| metric | n | candidate | control | diff | one-sided bound | margin | verdict |",
                 "|---|---:|---:|---:|---:|---:|---:|---|"]
        for m in self.metrics:
            lines.append(f"| {m.name} | {m.n} | {m.candidate:.1f} | {m.control:.1f} | {m.diff:+.1f} | {m.lower:+.1f} | "
                         f"-{m.margin:.1f} | {m.verdict} |")
        if self.category_diffs:
            lines += ["", "BFCL per category (candidate − control, points): "
                      + ", ".join(f"{c} {d:+.1f}" for c, d in sorted(self.category_diffs.items()))]
        lines += [f"- FAIL: {f}" for f in self.failures] + [f"- note: {n}" for n in self.notes]
        if self.verdict == "inconclusive":
            lines.append("- inconclusive: the evidence cannot yet show the drop is within the margin; add prompts or "
                         "sample several responses per prompt, then rerun")
        return "\n".join(lines)


# ------------------------------------------------------------------ bootstrap
def _lower_bound(boot: np.ndarray, confidence: float) -> float:
    return float(np.quantile(boot, 1.0 - confidence))


def _paired_mean(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    d = a - b
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return float(d.mean()), d[idx].mean(axis=1)


def _paired_ratio(num_a: np.ndarray, num_b: np.ndarray, den: np.ndarray, n_boot: int,
                  rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """(sum a - sum b) / sum den with prompts (clusters) resampled together."""
    idx = rng.integers(0, den.size, size=(n_boot, den.size))
    boot = (num_a[idx].sum(1) - num_b[idx].sum(1)) / den[idx].sum(1)
    return float((num_a.sum() - num_b.sum()) / den.sum()), boot


def _stratified_mean_of_means(groups: dict[str, tuple[np.ndarray, np.ndarray]], n_boot: int,
                              rng: np.random.Generator) -> tuple[float, np.ndarray]:
    point, boot = [], np.zeros(n_boot)
    for a, b in groups.values():
        p, bs = _paired_mean(a, b, n_boot, rng)
        point.append(p)
        boot += bs
    return float(np.mean(point)), boot / len(groups)


# ----------------------------------------------------------------------- gate
class InstructionParityGate:
    def __init__(self, margins: ParityMargins = ParityMargins(), *, n_boot: int = 4000, seed: int = 0,
                 run_self_tests: bool = True) -> None:
        self.margins, self.n_boot, self.seed, self.run_self_tests = margins, n_boot, seed, run_self_tests

    def evaluate(
        self,
        *,
        ifeval_candidate: Optional[Sequence[ifeval.IFEvalResult]] = None,
        ifeval_control: Optional[Sequence[ifeval.IFEvalResult]] = None,
        bfcl_candidate: Optional[Sequence[bfcl.BFCLResult]] = None,
        bfcl_control: Optional[Sequence[bfcl.BFCLResult]] = None,
    ) -> ParityReport:
        """Score both benchmarks. Missing evidence for either one fails the gate."""
        mg = self.margins
        rep = ParityReport(margins=asdict(mg))
        if self.run_self_tests:
            ifeval.self_test()
            bfcl.self_test()
        rng = np.random.default_rng(self.seed)
        if ifeval_candidate and ifeval_control:
            self._ifeval(rep, ifeval_candidate, ifeval_control, rng)
        else:
            rep.failures.append("no IFEval evidence for both candidate and control")
        if bfcl_candidate and bfcl_control:
            self._bfcl(rep, bfcl_candidate, bfcl_control, rng)
        else:
            rep.failures.append("no BFCL evidence for both candidate and control")
        return rep

    def _ifeval(self, rep: ParityReport, cand, ctrl, rng) -> None:
        mg = self.margins
        c, k = ifeval.by_key(cand), ifeval.by_key(ctrl)
        if set(c) != set(k):
            raise ValueError(f"IFEval prompts differ between candidate and control ({len(set(c) ^ set(k))} unmatched); "
                             "the comparison must be paired")
        keys = sorted(c)
        for key in keys:
            if c[key].instruction_ids != k[key].instruction_ids:
                raise ValueError(f"IFEval prompt {key} has different instructions in the two arms")
        a = np.array([c[x].prompt_strict for x in keys], dtype=np.float64)
        b = np.array([k[x].prompt_strict for x in keys], dtype=np.float64)
        diff, boot = _paired_mean(a, b, self.n_boot, rng)
        rep.metrics.append(ParityMetric("IFEval prompt-level strict", 100 * a.mean(), 100 * b.mean(), 100 * diff,
                                        100 * _lower_bound(boot, mg.confidence), mg.ifeval_prompt_strict, len(keys)))
        na = np.array([sum(c[x].strict) for x in keys], dtype=np.float64)
        nb = np.array([sum(k[x].strict) for x in keys], dtype=np.float64)
        den = np.array([len(c[x].strict) for x in keys], dtype=np.float64)
        diff, boot = _paired_ratio(na, nb, den, self.n_boot, rng)
        rep.metrics.append(ParityMetric("IFEval instruction-level strict", 100 * na.sum() / den.sum(),
                                        100 * nb.sum() / den.sum(), 100 * diff, 100 * _lower_bound(boot, mg.confidence),
                                        mg.ifeval_instruction_strict, int(den.sum())))
        miss = sum(r.missing_response for r in cand)
        if miss:
            rep.notes.append(f"{miss} IFEval prompt(s) had no candidate response and count as failures")

    def _bfcl(self, rep: ParityReport, cand, ctrl, rng) -> None:
        mg = self.margins
        c = {r.id: r for r in cand}
        k = {r.id: r for r in ctrl}
        if len(c) != len(cand) or len(k) != len(ctrl):
            raise ValueError("duplicate BFCL item IDs")
        if set(c) != set(k):
            raise ValueError(f"BFCL items differ between candidate and control ({len(set(c) ^ set(k))} unmatched); "
                             "the comparison must be paired")
        groups: dict[str, tuple[list[float], list[float]]] = {}
        for i in sorted(c):
            if c[i].category != k[i].category:
                raise ValueError(f"BFCL item {i} has different categories in the two arms")
            ga, gb = groups.setdefault(c[i].category, ([], []))
            ga.append(float(c[i].correct))
            gb.append(float(k[i].correct))
        arrays = {cat: (np.array(a), np.array(b)) for cat, (a, b) in groups.items()}
        diff, boot = _stratified_mean_of_means(arrays, self.n_boot, rng)
        cand_overall = float(np.mean([a.mean() for a, _ in arrays.values()]))
        ctrl_overall = float(np.mean([b.mean() for _, b in arrays.values()]))
        rep.metrics.append(ParityMetric("BFCL overall (mean of categories)", 100 * cand_overall, 100 * ctrl_overall,
                                        100 * diff, 100 * _lower_bound(boot, mg.confidence), mg.bfcl_overall, len(c)))
        for cat, (a, b) in sorted(arrays.items()):
            d = 100 * float(a.mean() - b.mean())
            rep.category_diffs[cat] = d
            if d < -mg.bfcl_category_points:
                rep.failures.append(f"BFCL {cat} dropped {-d:.1f} points (limit {mg.bfcl_category_points:.1f})")
        missing = set(bfcl.CATEGORIES) - set(arrays)
        if missing:
            rep.notes.append(f"BFCL categories not evaluated: {sorted(missing)}")


def run_parity(
    *,
    ifeval_data: str | Path,
    ifeval_candidate: str | Path,
    ifeval_control: str | Path,
    bfcl_questions: Sequence[str | Path],
    bfcl_answers: Sequence[Optional[str | Path]],
    bfcl_candidate: str | Path,
    bfcl_control: str | Path,
    margins: ParityMargins = ParityMargins(),
    seed: int = 0,
) -> ParityReport:
    """Score official-format response files for both arms and run the gate."""
    items, skipped = ifeval.load_ifeval(ifeval_data, skip_unsupported=True)
    b_items = [it for q, a in zip(bfcl_questions, bfcl_answers) for it in bfcl.load_bfcl(q, a)]
    rep = InstructionParityGate(margins, seed=seed).evaluate(
        ifeval_candidate=ifeval.score_ifeval(items, ifeval.load_responses(ifeval_candidate)),
        ifeval_control=ifeval.score_ifeval(items, ifeval.load_responses(ifeval_control)),
        bfcl_candidate=bfcl.score_bfcl(b_items, bfcl.load_outputs(bfcl_candidate)),
        bfcl_control=bfcl.score_bfcl(b_items, bfcl.load_outputs(bfcl_control)),
    )
    if skipped:
        rep.notes.append(f"{len(skipped)} IFEval prompt(s) skipped in both arms (unsupported instruction types)")
    return rep


def write_report(rep: ParityReport, json_path: Optional[str] = None, md_path: Optional[str] = None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(rep.to_dict(), indent=2) + "\n")
    if md_path:
        Path(md_path).write_text(rep.to_markdown() + "\n")


def load_report(path: str | Path) -> ParityReport:
    return ParityReport.from_dict(json.loads(Path(path).read_text()))
