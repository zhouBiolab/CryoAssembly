
import hashlib
import os
import re
import sqlite3
import subprocess
import time

USALIGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "USalign")

TM_CACHE_VERSION = 1
TM_USALIGN_ARGS = ("-TMscore", "7", "-ter", "0")
_USALIGN_TIMEOUT_S = 300


class USalignError(RuntimeError):
    pass


def default_tm_cache_path():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"),
                                                            ".cache")
    return os.path.join(base, "protassem", "tm.sqlite3")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

_FILE_FINGERPRINTS = {}


def structure_fingerprint(path):
    absolute = os.path.abspath(str(path))
    stat = os.stat(absolute)
    cache_key = (absolute, stat.st_size, stat.st_mtime_ns)
    cached = _FILE_FINGERPRINTS.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(absolute, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = "%d:%s" % (stat.st_size, digest.hexdigest())
    _FILE_FINGERPRINTS.clear()
    _FILE_FINGERPRINTS[cache_key] = value
    return value


def usalign_fingerprint(usalign_path=None):
    path = os.path.abspath(str(usalign_path or USALIGN_PATH))
    stat = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "%s:%d:%s" % (path, stat.st_size, digest.hexdigest()[:32])


def pair_key(pdb1, pdb2, usalign_path=None):
    first, second = sorted((structure_fingerprint(pdb1), structure_fingerprint(pdb2)))
    return "|".join([str(TM_CACHE_VERSION), first, second,
                     usalign_fingerprint(usalign_path), " ".join(TM_USALIGN_ARGS)])


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def usalign_pair(pdb1, pdb2, usalign_path=None):
    path = str(usalign_path or USALIGN_PATH)
    command = [path, str(pdb1), str(pdb2)] + list(TM_USALIGN_ARGS)
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=_USALIGN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise USalignError('USalign timed out (%ds): %s vs %s'
                           % (_USALIGN_TIMEOUT_S, pdb1, pdb2))
    except OSError as exc:
        raise USalignError('Cannot execute USalign (%s): %s' % (path, exc))
    if result.returncode != 0:
        raise USalignError('USalign exit code %d: %s vs %s\nstderr: %s'
                           % (result.returncode, pdb1, pdb2,
                              (result.stderr or "").strip()[:500]))
    return _parse_usalign(result.stdout, pdb1, pdb2)


def _parse_usalign(stdout, pdb1, pdb2):
    tm1 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1",
                    stdout)
    tm2 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2",
                    stdout)
    seqid = re.search(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)", stdout)
    if not (tm1 and tm2):
        raise USalignError('Cannot parse TM-score from USalign: %s vs %s\nstdout: %s'
                           % (pdb1, pdb2, (stdout or "").strip()[:500]))
    tm = min(float(tm1.group(1)), float(tm2.group(1)))
    return tm, (float(seqid.group(1)) if seqid else 0.0)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tm_scores (
    key TEXT PRIMARY KEY,
    tm REAL NOT NULL,
    seqid REAL NOT NULL,
    created_at REAL NOT NULL
)
"""


class TMStore:

    def __init__(self, path):
        self.path = str(path)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(_SCHEMA)
        self.connection.commit()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, key):
        row = self.connection.execute(
            "SELECT tm, seqid FROM tm_scores WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return {"tm": float(row[0]), "seqid": float(row[1])}

    def put(self, key, value):
        self.connection.execute(
            "INSERT OR REPLACE INTO tm_scores (key, tm, seqid, created_at) "
            "VALUES (?, ?, ?, ?)",
            (key, float(value["tm"]), float(value["seqid"]), time.time()))
        self.connection.commit()
        self.writes += 1

    def count(self):
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM tm_scores").fetchone()[0])

    def snapshot(self):
        return {"tm_cache": os.path.basename(self.path), "path": self.path,
                "hits": self.hits, "misses": self.misses, "writes": self.writes,
                "rows": self.count()}

    def close(self):
        try:
            self.connection.close()
        except Exception:
            pass


_STORE = None
_STORE_STATE = {"mode": "off", "path": None}
_MEMORY = {}


def configure_tm_cache(tm_cache="auto"):
    global _STORE
    if _STORE is not None:
        _STORE.close()
        _STORE = None
    _MEMORY.clear()
    mode = "auto" if tm_cache is None else str(tm_cache)
    if mode == "off":
        _STORE_STATE.update({"mode": "off", "path": None})
        return tm_cache_snapshot()
    path = default_tm_cache_path() if mode == "auto" else mode
    _STORE = TMStore(path)
    _STORE_STATE.update({"mode": mode, "path": path})
    return tm_cache_snapshot()


def tm_cache_enabled():
    return _STORE is not None


def tm_cache_snapshot():
    snapshot = {"tm_cache_mode": _STORE_STATE["mode"],
                "tm_cache_path": _STORE_STATE["path"],
                "memory_entries": len(_MEMORY)}
    if _STORE is not None:
        snapshot.update(_STORE.snapshot())
    return snapshot


def _cached_pair(pdb1, pdb2, usalign_path=None):
    key = pair_key(pdb1, pdb2, usalign_path)
    if _STORE is None:
        tm, seqid = usalign_pair(pdb1, pdb2, usalign_path)
        return {"tm": tm, "seqid": seqid}
    cached = _MEMORY.get(key)
    if cached is not None:
        return cached
    cached = _STORE.get(key)
    if cached is not None:
        _MEMORY[key] = cached
        return cached
    tm, seqid = usalign_pair(pdb1, pdb2, usalign_path)
    value = {"tm": tm, "seqid": seqid}
    _MEMORY[key] = value
    _STORE.put(key, value)
    return value


def calculate_tm_score(pdb1, pdb2, usalign_path=None):
    return _cached_pair(pdb1, pdb2, usalign_path)["tm"]


def calculate_seqid(pdb1, pdb2, usalign_path=None):
    return _cached_pair(pdb1, pdb2, usalign_path)["seqid"]


def _tm_worker(arg):
    p1, p2 = arg[0], arg[1]
    usalign_path = arg[2] if len(arg) > 2 else None
    try:
        tm, seqid = usalign_pair(p1, p2, usalign_path)
    except USalignError as exc:
        return {"ok": False, "p1": p1, "p2": p2, "error": str(exc)}
    return {"ok": True, "p1": p1, "p2": p2, "tm": tm, "seqid": seqid}


def prefill_tm_cache(pairs, context, usalign_path=None):
    todo, seen = [], set()
    for p1, p2 in pairs:
        key = pair_key(p1, p2, usalign_path)
        if _STORE is not None and (key in _MEMORY or _STORE.get(key) is not None):
            continue
        if key in seen:
            continue
        seen.add(key)
        todo.append((p1, p2, usalign_path))
    if not todo:
        return 0
    failures = []
    for result in context.map(_tm_worker, todo):
        if not result["ok"]:
            failures.append(result)
            continue
        key = pair_key(result["p1"], result["p2"], usalign_path)
        value = {"tm": result["tm"], "seqid": result["seqid"]}
        if _STORE is not None:
            _MEMORY[key] = value
            _STORE.put(key, value)
    if failures:
        first = failures[0]
        raise USalignError('%d/%d structure pairs failed; example: %s vs %s\n%s'
                           % (len(failures), len(todo), first["p1"], first["p2"],
                              first["error"]))
    return len(todo)
