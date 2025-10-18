# Minimal, dependency-free router with temperature + top-k
import math
from dataclasses import dataclass

@dataclass
class RouterCfg:
    moe_top_k: int = 4
    tau: float = 0.92

def _softmax(x, tau):
    # numerically stable softmax with temperature
    m = max(x)
    exps = [math.exp((v - m)/tau) for v in x]
    s = sum(exps)
    return [e/s for e in exps]

def make_router(cfg: RouterCfg):
    def route(logits):
        probs = _softmax(list(logits), cfg.tau)
        topk = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:cfg.moe_top_k]
        return topk, probs
    return route
