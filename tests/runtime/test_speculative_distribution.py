"""Sampled speculative decoding must preserve the target distribution exactly (§14).

Chi-square goodness-of-fit / homogeneity tests at alpha = 0.001 with fixed seeds
(deterministic, not flaky). A negative control shows the test detects a biased
verifier, so passing is meaningful.
"""

import math

import pytest
import torch

from m63_omni.config import tiny
from m63_omni.model.m63 import M63Omni
from m63_omni.runtime import speculative


def chi2_critical(df: int, z: float = 3.0902) -> float:
    """Wilson–Hilferty approximation of the chi-square quantile (z = 3.09 -> alpha 0.001)."""
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + z * math.sqrt(a)) ** 3


def chi2_gof(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, int]:
    n = counts.sum()
    exp = probs * n
    keep = exp >= 5
    stat = float(((counts[keep] - exp[keep]) ** 2 / exp[keep]).sum())
    return stat, int(keep.sum()) - 1


def _dists(V=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(3, V, generator=g) * 1.5, -1)
    q = torch.softmax(torch.randn(2, V, generator=g) * 1.5, -1)
    return p, q, g


def _run(verify, N=20000, V=6):
    p, q, g = _dists(V)
    first = torch.zeros(V)
    second = torch.zeros(V)
    for _ in range(N):
        d = [int(torch.multinomial(q[j], 1, generator=g)) for j in range(2)]
        n_acc, repl = verify(p.log(), d, [q[0], q[1]], 1.0, g)
        emitted = d[:n_acc] + [int(repl)]
        first[emitted[0]] += 1
        if len(emitted) > 1:
            second[emitted[1]] += 1
    return p, first, second


def test_verify_preserves_target_distribution():
    p, first, second = _run(speculative._verify)
    stat, df = chi2_gof(first, p[0])
    assert stat < chi2_critical(df), (stat, df)
    stat2, df2 = chi2_gof(second, p[1])  # conditional on the first draft being accepted
    assert stat2 < chi2_critical(df2), (stat2, df2)


def test_negative_control_detects_a_biased_verifier():
    def always_accept(target_logits, drafts, q_probs, temperature, gen):
        return len(drafts), target_logits[len(drafts)].argmax()

    p, first, _ = _run(always_accept, N=5000)
    stat, df = chi2_gof(first, p[0])
    assert stat > chi2_critical(df)  # emits draws from q, not p — the test must notice


@pytest.fixture
def single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # tiny matrices: threading overhead dominates
    yield
    torch.set_num_threads(n)


@pytest.mark.slow
def test_end_to_end_sampled_speculative_matches_target_only(single_thread):
    """Distribution of the 2nd and 3rd generated tokens: target-only vs MTP speculative (homogeneity)."""
    torch.manual_seed(0)
    cfg = tiny(model={"vocab_size": 80, "special_tokens": 64}, multimodal={"enabled": False},
               mtp={"layers": 2, "draft_tokens": 2})
    m = M63Omni(cfg).eval()
    with torch.no_grad():  # sharpen the output distribution so the test has power over 80 classes
        m.embed.weight.mul_(8.0)
    prompt = torch.randint(0, 16, (1, 6), generator=torch.Generator().manual_seed(1))
    N, V = 2500, cfg.model.vocab_size
    counts = torch.zeros(2, 2, V)
    for arm, draft in enumerate((0, 2)):
        g = torch.Generator().manual_seed(100 + arm)
        for _ in range(N):
            out, _, _ = speculative.generate(m, prompt, 3, draft_tokens=draft, temperature=1.0, generator=g)
            counts[arm, 0, out[1]] += 1
            counts[arm, 1, out[2]] += 1
    for pos in range(2):
        a, b = counts[0, pos], counts[1, pos]
        tot = a + b
        keep = tot >= 10
        a, b = torch.cat([a[keep], a[~keep].sum(0, keepdim=True)]), torch.cat([b[keep], b[~keep].sum(0, keepdim=True)])
        tot = a + b
        nonzero = tot > 0
        ea, eb = tot * (a.sum() / tot.sum()), tot * (b.sum() / tot.sum())
        stat = float((((a - ea) ** 2 / ea.clamp(min=1e-9)) + ((b - eb) ** 2 / eb.clamp(min=1e-9)))[nonzero].sum())
        df = int(nonzero.sum()) - 1
        assert stat < chi2_critical(df), (pos, stat, df)
