"""Disk-caching utilities backed by joblib.Memory."""
import joblib
from thesis.config import Config


def make_memory(cfg: Config) -> joblib.Memory:
    """Return a joblib.Memory instance pointing at cfg.paths.cache."""
    cfg.paths.ensure_dirs()
    return joblib.Memory(cfg.paths.cache, verbose=0)
