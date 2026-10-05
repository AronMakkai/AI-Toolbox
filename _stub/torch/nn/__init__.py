class Module:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None
    def to(self, *a, **k): return self
    def eval(self): return self
    def load_state_dict(self, *a, **k): pass
