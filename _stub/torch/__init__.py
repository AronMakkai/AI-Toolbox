"""Minimal stub so ai_toolbox.py can be imported for its Tk/GUI/logic
code without a real torch install (CUDA/inference code paths are
never exercised by these tests -- only imported at module load)."""

class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def device_count(): return 0
    @staticmethod
    def empty_cache(): pass

cuda = _Cuda()

class _Mps:
    @staticmethod
    def is_available(): return False

class backends:
    mps = _Mps()

def device(*a, **k): return "cpu"

class no_grad:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __call__(self, fn): return fn

def load(*a, **k): return {}
def save(*a, **k): pass
def from_numpy(x): return x
def tensor(x, *a, **k): return x
def manual_seed(*a, **k): pass

__version__ = "0.0.0-stub"
