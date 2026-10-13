
import json
import os

LEDGER_VERSION = 1
LEDGER_NAME = "candidates.jsonl"

CANDIDATE_STATES = ("ok", "filtered", "error")
END_STATUSES = ("ok", "error", "cancelled")

MASK_ERROR_REASON = "mask_evaluations_failed"


class CandidateLedgerWriter:

    def __init__(self, output_dir, request_id):
        if not request_id:
            raise ValueError('request_id must not be empty: ledger ownership must be identifiable')
        self.path = os.path.join(output_dir, LEDGER_NAME)
        self.request_id = str(request_id)
        self.count = 0
        self.finished = False

    def publish(self, candidate_id, state, name=None, overlap=None,
                source=None, error=None, reason=None):
        if state not in CANDIDATE_STATES:
            raise ValueError('Unknown candidate status: %r' % (state,))
        record = {"v": LEDGER_VERSION, "request_id": self.request_id,
                  "kind": "candidate", "id": int(candidate_id), "state": state}
        if name is not None:
            record["name"] = os.path.basename(str(name))
        if overlap is not None:
            record["overlap"] = float(overlap)
        if source:
            record["source"] = dict(source)
        if error is not None:
            record["error"] = str(error)
        if reason is not None:
            record["reason"] = str(reason)
        self._append(record)
        self.count = max(self.count, int(candidate_id) + 1)
        return record

    def finish(self, status, error=None):
        if status not in END_STATUSES:
            raise ValueError('Unknown end status: %r' % (status,))
        if status == "error" and not error:
            raise ValueError('status=error requires error text')
        if self.finished:
            return None
        record = {"v": LEDGER_VERSION, "request_id": self.request_id,
                  "kind": "end", "status": status, "count": self.count}
        if error is not None:
            record["error"] = str(error)
        self._append(record)
        self.finished = True
        return record

    def _append(self, record):
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class LedgerReader:

    def __init__(self, path, request_id=None):
        self.path = path
        self.request_id = str(request_id) if request_id is not None else None
        self.candidates = {}
        self.skipped = []
        self.end = None
        self._offset = 0
        self._partial = b""

    def poll(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset += len(chunk)
        if not chunk:
            return []
        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()
        records = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            record = self._parse(line.decode("utf-8"))
            self._apply(record)
            records.append(record)
        return records

    def pending_partial(self):
        return self._partial.decode("utf-8", "replace")

    def ready_ids(self, start, stop):
        return [cid for cid in range(start, stop) if cid in self.candidates]

    def expected_count(self):
        return self.end["count"] if self.end else None

    def _parse(self, line):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise RuntimeError('Invalid JSON ledger record (%s): %s' % (self.path, exc))
        if record.get("v") != LEDGER_VERSION:
            raise RuntimeError('Ledger protocol version mismatch: %r' % (record.get("v"),))
        if self.request_id is not None and record.get("request_id") != self.request_id:
            raise RuntimeError('Ledger ownership mismatch (expected %s, actual %s): refusing stale ledger'
                               % (self.request_id, record.get("request_id")))
        return record

    def _apply(self, record):
        kind = record.get("kind")
        if kind == "candidate":
            cid = int(record["id"])
            if record["state"] not in CANDIDATE_STATES:
                raise RuntimeError('Unknown candidate status: %r' % (record["state"],))
            if cid in self.candidates:
                raise RuntimeError('Candidate ID published twice: %d' % cid)
            self.candidates[cid] = record
            if record["state"] != "ok":
                self.skipped.append(record)
        elif kind == "end":
            if self.end is not None:
                raise RuntimeError('Duplicate request end record')
            self.end = record
        else:
            raise RuntimeError('Unknown ledger record type: %r' % (kind,))
