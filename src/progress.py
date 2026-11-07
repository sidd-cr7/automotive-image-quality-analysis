"""Lightweight per-request progress tracking.

The orchestrator calls begin()/end() at real stage boundaries; the frontend polls
GET /progress/{job_id}. Stages are only reported complete after end() is called.
No-ops when no job is active, so normal (non-UI) callers are unaffected.
"""

import contextvars
import threading
import time
from typing import Any, Dict, Optional

STAGES = ("detection", "inspection", "final")

_current_job: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("progress_job", default=None)
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_TTL_SECONDS = 900


def start(job_id: str) -> None:
    now = time.time()
    with _lock:
        for k in [k for k, v in _jobs.items() if now - v["updated"] > _TTL_SECONDS]:
            del _jobs[k]
        _jobs[job_id] = {"current": None, "completed": [], "done": False, "started": now, "updated": now}
    _current_job.set(job_id)


def _update(fn) -> None:
    job_id = _current_job.get()
    if not job_id:
        return
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            fn(job)
            job["updated"] = time.time()


def begin(stage: str) -> None:
    _update(lambda j: j.__setitem__("current", stage))


def end(stage: str) -> None:
    def _fn(j):
        if stage not in j["completed"]:
            j["completed"].append(stage)
        if j["current"] == stage:
            j["current"] = None
    _update(_fn)


def finish() -> None:
    _update(lambda j: j.update(done=True, current=None))


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return None if job is None else {
            "current": job["current"], "completed": list(job["completed"]),
            "done": job["done"], "elapsed_s": round(time.time() - job["started"], 1),
        }
