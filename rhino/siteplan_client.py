"""
HTTP client for the Site Plan Drafter backend — pure Python stdlib only, so it
runs unmodified inside Rhino 8's CPython (no pip installs) AND headless for
testing. The Rhino command (SitePlan_command.py) is a thin wrapper over this.

Usage (any Python):
    from siteplan_client import generate
    generate({"west": -76.5515, "south": 42.5305,
              "east": -76.5415, "north": 42.5385}, "/tmp/plan.dxf")

CLI smoke test:
    python siteplan_client.py WEST SOUTH EAST NORTH OUT.dxf [layer,layer,...]
"""
import json
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://localhost:8000"
START_HINT = ("cannot reach the Site Plan backend — start it with:\n"
              "  cd backend && python -m uvicorn app.main:app --port 8000")


class SitePlanError(RuntimeError):
    """Any client-visible failure: backend down, rejected request, failed job."""


def _request(method: str, url: str, body=None, timeout: float = 60):
    """One HTTP round-trip. Returns parsed JSON for JSON responses, raw bytes
    otherwise (the DXF download). Turns HTTP/socket errors into SitePlanError
    with the backend's own `detail` message when there is one."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if "json" in (resp.headers.get("Content-Type") or ""):
                return json.loads(raw)
            return raw
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", str(exc))
        except Exception:
            detail = str(exc)
        raise SitePlanError(f"backend rejected the request ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise SitePlanError(f"{START_HINT}\n({exc.reason})")


def submit_job(bbox: dict, layers=None, options=None, style=None,
               base: str = DEFAULT_BASE) -> str:
    """POST /api/jobs. Returns the job_id."""
    body = {"bbox": bbox}
    if layers is not None:
        body["layers"] = layers
    if options is not None:
        body["options"] = options
    if style is not None:
        body["style"] = style
    status = _request("POST", f"{base}/api/jobs", body)
    return status["job_id"]


def poll_job(job_id: str, base: str = DEFAULT_BASE, interval: float = 3.0,
             timeout: float = 1800, on_progress=None) -> dict:
    """Poll GET /api/jobs/{id} until the job finishes. Calls on_progress(status)
    each poll (return False from it to cancel). Raises SitePlanError on job
    failure or timeout; returns the final status on success."""
    deadline = time.time() + timeout
    while True:
        status = _request("GET", f"{base}/api/jobs/{job_id}")
        if on_progress is not None and on_progress(status) is False:
            raise SitePlanError("cancelled")
        if status["status"] == "complete":
            return status
        if status["status"] == "failed":
            raise SitePlanError(f"job failed: {status.get('error') or 'unknown error'}")
        if time.time() > deadline:
            raise SitePlanError(f"job still {status['status']} after {timeout:.0f}s")
        time.sleep(interval)


def export_dxf(job_id: str, out_path: str, style=None,
               base: str = DEFAULT_BASE) -> str:
    """POST /api/jobs/{id}/export and save the DXF to out_path."""
    raw = _request("POST", f"{base}/api/jobs/{job_id}/export", body=style,
                   timeout=300)
    if not isinstance(raw, bytes) or not raw:
        raise SitePlanError("export returned no file")
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path


def generate(bbox: dict, out_path: str, layers=None, options=None, style=None,
             base: str = DEFAULT_BASE, on_progress=None) -> str:
    """Convenience: submit -> poll -> export. Returns out_path."""
    job_id = submit_job(bbox, layers=layers, options=options, style=style, base=base)
    poll_job(job_id, base=base, on_progress=on_progress)
    return export_dxf(job_id, out_path, base=base)


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (6, 7):
        sys.exit(__doc__)
    w, s, e, n = (float(v) for v in sys.argv[1:5])
    layers = sys.argv[6].split(",") if len(sys.argv) == 7 else None

    def _print_progress(status):
        stages = status.get("progress") or {}
        done = [k for k, v in stages.items() if v == "done"]
        running = [k for k, v in stages.items() if v == "running"]
        print(f"  {status['status']}: done={done} running={running}")

    path = generate({"west": w, "south": s, "east": e, "north": n},
                    sys.argv[5], layers=layers, on_progress=_print_progress)
    print(f"saved {path}")
