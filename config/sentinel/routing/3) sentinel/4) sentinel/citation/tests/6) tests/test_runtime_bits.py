from sentinel.routing.gating import RouterCfg, make_router
from sentinel.early_exit import ExitCfg, make_exit_policy
from sentinel.citation.guard import CiteGuard

def test_router_topk_and_probs():
    router = make_router(RouterCfg(moe_top_k=4, tau=0.92))
    topk, probs = router([0.1, 1.2, -0.4, 0.7, 0.3])
    assert len(topk) == 4
    assert abs(sum(probs) - 1.0) < 1e-6

def test_exit_policy_thresholds():
    p = make_exit_policy(ExitCfg(enabled=True, p_conf=0.9))
    assert p([0.1, 0.9, 0.0]) is True
    p2 = make_exit_policy(ExitCfg(enabled=False))
    assert p2([0.99]) is False

def test_citation_guard_minimum():
    g = CiteGuard(min_cites=3)
    ok, missing = g.ensure_min(['a','b'])
    assert ok is False and missing == 1
    ok2, missing2 = g.ensure_min(['a','b','c'])
    assert ok2 is True and missing2 == 0
