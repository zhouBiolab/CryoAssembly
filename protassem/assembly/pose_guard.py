"""Per-round suppression of failed homologs with identical input poses."""

from dataclasses import dataclass
import hashlib
import logging
import math
import os
import re
import subprocess

from protassem.core.similarity import USALIGN_PATH

log = logging.getLogger(__name__)
POSE_ARGS = ("-se", "-TMscore", "7", "-ter", "0")


class PoseComparisonError(RuntimeError):
    pass


@dataclass(frozen=True)
class PoseComparison:
    tm1: float
    tm2: float
    rmsd: float
    length1: int
    length2: int
    aligned: int
    identical_sequence: bool

    @property
    def same(self):
        return (self.length1 == self.length2 == self.aligned
                and self.length1 > 0 and self.identical_sequence
                and min(self.tm1, self.tm2) >= 0.999 and self.rmsd <= 0.1)


class InitialPoseGuard:
    def __init__(self, usalign_path=USALIGN_PATH, timeout=30):
        self.usalign_path = str(usalign_path)
        self.timeout = timeout
        self._fingerprints = {}
        self._comparisons = {}
        self._scope = None
        self._failures = {}
        self._thresholds = {}

    def _fingerprint(self, filename):
        path = os.path.abspath(str(filename))
        stat = os.stat(path)
        key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._fingerprints:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for data in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(data)
            self._fingerprints[key] = (stat.st_size, digest.hexdigest())
        return self._fingerprints[key]

    def compare(self, first, second):
        try:
            key = (self._fingerprint(first), self._fingerprint(second),
                   self._fingerprint(self.usalign_path), POSE_ARGS)
            if key in self._comparisons:
                return self._comparisons[key]
            result = subprocess.run(
                [self.usalign_path, str(first), str(second)] + list(POSE_ARGS),
                capture_output=True, text=True, timeout=self.timeout)
            if result.returncode:
                raise PoseComparisonError("USalign exit code %d" % result.returncode)
            text = result.stdout
            lengths = [int(value) for value in re.findall(
                r"Length of Structure_[12]:\s*(\d+) residues", text)]
            aligned = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([\d.eE+-]+)", text)
            scores = [float(value) for value in re.findall(
                r"TM-score=\s*([\d.eE+-]+) \(normalized by length of Structure_[12]", text)]
            sequences = [line.strip() for line in text.splitlines()
                         if re.fullmatch(r"[A-Za-z-]+", line.strip())]
            if len(lengths) != 2 or len(scores) != 2 or not aligned or len(sequences) != 2:
                raise PoseComparisonError("Incomplete USalign pose comparison output")
            rmsd = float(aligned.group(2))
            if not all(math.isfinite(value) for value in scores + [rmsd]):
                raise PoseComparisonError("Nonfinite USalign pose scores")
            identical = (sequences[0] == sequences[1]
                         and '-' not in sequences[0]
                         and len(sequences[0]) == lengths[0]
                         and len(sequences[1]) == lengths[1])
            comparison = PoseComparison(scores[0], scores[1], rmsd,
                                        lengths[0], lengths[1], int(aligned.group(1)),
                                        identical)
            self._comparisons[key] = comparison
            return comparison
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise PoseComparisonError(str(exc)) from exc

    def scope(self, orch, round_num):
        paths = []
        for filename in (orch.current_density_mrc, orch.current_target_txt):
            stat = os.stat(str(filename))
            paths.append((os.path.abspath(str(filename)), stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns))
        scope = (round_num, getattr(orch, '_mask_iter', 0),
                 len(orch.accepted_domain_pdbs), len(orch.accepted_chains), tuple(paths))
        if scope != self._scope:
            self._scope = scope
            self._failures.clear()
            self._thresholds.clear()
        return scope

    def _failure_key(self, kind, group_id, threshold):
        group = (kind, group_id)
        threshold = float(threshold)
        if self._thresholds.get(group) != threshold:
            for key in list(self._failures):
                if key[:2] == group:
                    del self._failures[key]
            self._thresholds[group] = threshold
        return (kind, group_id, threshold)

    def find_failed(self, orch, round_num, kind, group_id, rec, threshold):
        if group_id is None:
            return None
        try:
            self.scope(orch, round_num)
            key = self._failure_key(kind, group_id, threshold)
            for failure in self._failures.get(key, []):
                if self._fingerprint(failure['pdb']) != failure['fingerprint']:
                    continue
                comparison = self.compare(failure['pdb'], rec['pdb_file'])
                if comparison.same:
                    label = self.label(kind, rec)
                    log.info('  [Same input pose] %s skipped; representative=%s round=%d '
                             'TM=(%.5f, %.5f) RMSD=%.3f; representative not accepted',
                             label, failure['label'], round_num,
                             comparison.tm1, comparison.tm2, comparison.rmsd)
                    return failure
        except (PoseComparisonError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
            log.warning('Initial-pose comparison unavailable for %s; fit normally: %s',
                        self.label(kind, rec), exc)
        return None

    def record_failed(self, orch, round_num, kind, group_id, rec, threshold,
                      cc, previous_scope):
        if group_id is None:
            return
        try:
            if not math.isfinite(float(cc)):
                return
            if self.scope(orch, round_num) != previous_scope:
                return
            key = self._failure_key(kind, group_id, threshold)
            self._failures.setdefault(key, []).append({
                'pdb': rec['pdb_file'], 'fingerprint': self._fingerprint(rec['pdb_file']),
                'label': self.label(kind, rec), 'cc': cc})
        except OSError as exc:
            log.warning('Cannot record failed input pose: %s', exc)

    @staticmethod
    def label(kind, rec):
        if kind == 'domain':
            return '%s_d%d' % (rec['chain_id'], rec['domain_num'])
        return rec['chain_id']
