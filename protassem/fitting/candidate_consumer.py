
import time

from protassem.fitting.candidate_ledger import MASK_ERROR_REASON

POLL_INTERVAL_S = 2.5
END_GRACE_POLLS = 2


class CandidateConsumer:

    def __init__(self, reader, batch_size, on_batch, request_finished,
                 on_cancel=None, sleep=time.sleep, poll_interval=POLL_INTERVAL_S,
                 describe_request=None):
        self.reader = reader
        self.batch_size = max(1, int(batch_size))
        self.on_batch = on_batch
        self.request_finished = request_finished
        self.on_cancel = on_cancel or (lambda: None)
        self.sleep = sleep
        self.poll_interval = poll_interval
        self.describe_request = describe_request or (lambda: "")
        self.skipped_mask_errors = 0

    def _check_states(self, records):
        for record in records:
            if record["state"] != "error":
                continue
            if record.get("reason") == MASK_ERROR_REASON:
                self.skipped_mask_errors += 1
                continue
            raise RuntimeError('Candidate %d failed (state=error): %s'
                               % (record["id"], record.get("error") or 'No reason provided'))

    def _missing_end_error(self):
        detail = ""
        try:
            detail = str(self.describe_request() or "")
        except Exception:
            detail = ""
        message = 'Request ended without a ledger end record (incomplete request)'
        return RuntimeError("%s：%s" % (message, detail) if detail else message)

    def run(self):
        batches = []
        results = []
        next_id = 0
        early_stop = False
        missing_end_polls = 0

        while True:
            self.reader.poll()
            end = self.reader.end
            total = end["count"] if end else None
            progressed = False

            while True:
                window = list(range(next_id, next_id + self.batch_size))
                if total is not None:
                    window = [cid for cid in window if cid < total]
                if not window:
                    break
                if len(self.reader.ready_ids(window[0], window[-1] + 1)) < len(window):
                    break
                records = [self.reader.candidates[cid] for cid in window]
                self._check_states(records)
                batch_results, hit = self.on_batch(records)
                results.extend(batch_results)
                batches.append({"start": window[0], "stop": window[-1] + 1,
                                "skipped": sum(1 for r in records
                                               if r["state"] != "ok")})
                next_id = window[-1] + 1
                progressed = True
                if hit:
                    early_stop = True
                    break

            if early_stop:
                self.on_cancel()
                self._await_request_end()
                break
            if end is not None:
                if end["status"] == "error":
                    raise RuntimeError('Request failed (ledger end.status=error): %s'
                                       % end.get("error"))
                if next_id >= total:
                    break
                if not progressed:
                    raise RuntimeError('Incomplete ledger: stream ended but candidates are missing in [%d, %d)'
                                       % (next_id, next_id + self.batch_size))
                continue
            if self.request_finished():
                missing_end_polls += 1
                if missing_end_polls > END_GRACE_POLLS:
                    raise self._missing_end_error()
            else:
                missing_end_polls = 0
            self.sleep(self.poll_interval)

        return {"batches": batches, "results": results, "early_stop": early_stop,
                "skipped": list(self.reader.skipped), "consumed": next_id,
                "skipped_mask_errors": self.skipped_mask_errors}

    def _await_request_end(self):
        while True:
            self.reader.poll()
            if self.reader.end is not None:
                if self.reader.end["status"] == "error":
                    raise RuntimeError('Request failed (ledger end.status=error): %s'
                                       % self.reader.end.get("error"))
                if self.request_finished():
                    return
            elif self.request_finished():
                self.sleep(self.poll_interval)
                self.reader.poll()
                if self.reader.end is None:
                    raise self._missing_end_error()
                continue
            self.sleep(self.poll_interval)
