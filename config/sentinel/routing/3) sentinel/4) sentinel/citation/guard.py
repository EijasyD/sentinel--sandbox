class CiteGuard:
    def __init__(self, min_cites=3, coverage_target=0.995):
        self.min_cites = min_cites
        self.coverage_target = coverage_target

    def ensure_min(self, cites):
        count = len(cites or [])
        ok = count >= self.min_cites
        return ok, max(0, self.min_cites - count)
