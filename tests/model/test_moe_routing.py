"""MoE router semantics (blueprint §4, §7, Appendix B "MoE routing")."""

import torch

from m63_omni.config import tiny
from m63_omni.model.moe import MoELayer, Router


def _router(**moe):
    cfg = tiny(moe=moe).moe
    torch.manual_seed(0)
    r = Router(16, cfg)
    torch.nn.init.normal_(r.weight, std=0.5)
    return r, cfg


def test_bias_changes_selection_not_output_weights():
    r, cfg = _router()
    h = torch.randn(64, 16)
    idx0, w0, s, _ = r(h)
    r.bias[3] = 10.0  # force expert 3 into every selection
    idx1, w1, s1, _ = r(h)
    assert (idx1 == 3).any(-1).all()
    expected = s1.gather(-1, idx1)
    torch.testing.assert_close(w1, expected / expected.sum(-1, keepdim=True))
    torch.testing.assert_close(w1.sum(-1), torch.ones(64))


def test_node_limited_routing():
    r, cfg = _router(routed_experts=8, experts_per_node=2, max_nodes_per_token=1, top_k=2)
    idx, *_ = r(torch.randn(128, 16))
    nodes = idx // cfg.experts_per_node
    assert (nodes == nodes[:, :1]).all()  # both experts on a single node


def test_node_limit_is_skipped_exactly_when_it_cannot_bind():
    """top-k <= max_nodes (the reference top-2 / 2 nodes) never changes a selection; top-4 / 2 nodes does."""
    h = torch.randn(512, 16)
    r, _ = _router(routed_experts=16, experts_per_node=2, max_nodes_per_token=2, top_k=2)
    assert not r.node_limited
    sel = torch.sigmoid(h @ r.weight.t()) + r.bias
    assert torch.equal(r(h)[0], sel.topk(2, dim=-1).indices)
    r4, cfg4 = _router(routed_experts=16, experts_per_node=2, max_nodes_per_token=2, top_k=4)
    assert r4.node_limited
    idx, *_ = r4(h)
    nodes = idx // cfg4.experts_per_node
    assert max(len(set(row.tolist())) for row in nodes) <= 2
    unrestricted = (torch.sigmoid(h @ r4.weight.t()) + r4.bias).topk(4, dim=-1).indices // cfg4.experts_per_node
    assert any(len(set(row.tolist())) > 2 for row in unrestricted)  # the limit binds for some tokens


def test_bias_update_favours_underloaded_experts():
    r, cfg = _router()
    counts = torch.tensor([100, 0, 50, 50, 50, 50, 50, 50])
    r.update_bias(counts)
    assert r.bias[1] > 0 and r.bias[0] < 0 and r.bias[2] == 0


def test_layer_stats_are_dropless_and_exclude_padding():
    cfg = tiny().moe
    torch.manual_seed(0)
    layer = MoELayer(16, cfg)
    layer.reset_parameters(0.1, 0.1)
    x = torch.randn(2, 10, 16)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 7:] = False
    mods = torch.zeros(2, 10, dtype=torch.long)
    mods[0, :4] = 2  # video
    _, st = layer(x, mods, valid)
    assert int(st.counts.sum()) == 17 * cfg.top_k
    assert int(st.modality_counts[2].sum()) == 4 * cfg.top_k
    assert st.tokens == 17 and torch.isfinite(st.aux_loss)
