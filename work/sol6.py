"""Solution state for 6_shortage (also generic): models, sources, item assignment."""
import json, collections

TYPES = "ntic"


class State:
    def __init__(self, inst, sol):
        self.inst = inst
        M, D = inst["models"], inst["datasets"]
        self.M, self.D = M, D
        self.src = {}            # source -> target
        for s, t in sol["modelMappings"]:
            self.src[s] = t
        self.tgt = collections.defaultdict(list)
        for s, t in self.src.items():
            self.tgt[t].append(s)
        self.item_bin = {}       # dataset id -> (model, type)
        for d, m in sol["dataMappings"]:
            self.item_bin[d] = (m, D[d]["dataType"])
        self.bin_items = collections.defaultdict(set)
        for d, b in self.item_bin.items():
            self.bin_items[b].add(d)
        used = set(m for _, m in sol["dataMappings"]) | set(self.src) | set(self.tgt)
        self.used = used

    def covered(self):
        return set((t, next(iter(self.M[s]["dataRequirements"]))) for s, t in self.src.items())

    def is_full(self, m):
        if m in self.src or self.tgt.get(m):
            return False
        for t in self.M[m]["dataRequirements"]:
            for d in self.bin_items.get((m, t), ()):
                if self.D[d]["isCopyrighted"]:
                    return False
        return True

    def to_sol(self):
        dm = [[d, b[0]] for d, b in sorted(self.item_bin.items())]
        mm = [[s, t] for s, t in sorted(self.src.items())]
        return {"dataMappings": dm, "modelMappings": mm}

    def score(self):
        import test_solution as ts
        sol = json.dumps(self.to_sol())
        sc, ok, msg = ts.getSolutionScore(sol, json.dumps(self.inst), show_details=False)
        return sc, ok, msg
