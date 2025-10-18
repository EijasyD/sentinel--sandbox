from dataclasses import dataclass

@dataclass
class ExitCfg:
    enabled: bool = True
    p_conf: float = 0.92  # confidence threshold

def make_exit_policy(cfg: ExitCfg):
    def should_exit(probs):
        if not cfg.enabled:
            return False
        return (max(probs) if probs else 0.0) >= cfg.p_conf
    return should_exit
