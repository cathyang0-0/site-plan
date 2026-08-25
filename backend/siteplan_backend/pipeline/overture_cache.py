"""
Tiny disk cache for Overture fetch results, keyed by (kind, bbox).

Why: the Overture GeoParquet endpoint has multi-hour slow spells (observed
2026-08-08/09: every fetch timing out at 90 s), and iterating on one site
re-fetches identical data constantly. Caching successful results makes a
previously-fetched site immune to outages and instant to regenerate.

No TTL: Overture re-releases roughly monthly and site geometry churns slowly.
Delete the cache dir (backend/.overture_cache) to force a refresh, or point
SITEPLAN_OVERTURE_CACHE somewhere else.
"""
import os
import pickle
from pathlib import Path

CACHE_DIR = Path(os.environ.get(
    "SITEPLAN_OVERTURE_CACHE",
    str(Path(__file__).resolve().parents[2] / ".overture_cache")))


def _path(kind: str, bbox: tuple) -> Path:
    key = "_".join(f"{float(v):.6f}" for v in bbox)
    return Path(CACHE_DIR) / f"{kind}_{key}.pkl"


def get(kind: str, bbox: tuple):
    """Cached object for this fetch, or None. A corrupt/unreadable entry is
    treated as a miss, never an error."""
    p = _path(kind, bbox)
    if not p.exists():
        return None
    try:
        return pickle.loads(p.read_bytes())
    except Exception:
        return None


def put(kind: str, bbox: tuple, obj) -> None:
    """Store a successful fetch. Write-then-rename so a crash mid-write can't
    leave a half-written entry that later reads as corrupt."""
    p = _path(kind, bbox)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(p)
