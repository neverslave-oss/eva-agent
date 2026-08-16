# ADR-012: Async Pipeline Execution

**Status:** Approved — implement  
**Date:** 2026-05-11  
**Author:** Olly + Fabio  
**Related:** ADR-008 (Critic Replica Pipeline), ADR-010 (Evolution Pipeline), ADR-011 (Micro-Planner)

---

## Problem

`POST /replica/pipeline` blocks the HTTP connection until all stages complete. Under a single-threaded model server, a 2-stage pipeline takes 60–180 s; a 3-stage pipeline takes 4–6 minutes. This has three consequences:

1. **Harness timeouts** — Sim 6 and Sim 7 both failed entirely due to curl/urllib timeouts, not actual capability gaps
2. **Caller blocking** — Telegram bot, API clients, and the evolution loop all block their thread for the full duration
3. **No visibility** — the caller gets zero intermediate output; if the pipeline errors mid-run, the client sees only a timeout

---

## Decision

Add an async execution mode alongside the existing synchronous endpoint.

### New endpoints

```
POST /replica/pipeline/async          → { "job_id": "<uuid>", "status": "queued" }
GET  /replica/pipeline/{job_id}       → { "job_id", "status": queued|running|complete|error,
                                           "stage": "<current>", "results": {partial}, "error": null }
GET  /replica/pipeline/{job_id}/stream → SSE stream of stage completion events
DELETE /replica/pipeline/{job_id}      → cancel job
```

### Implementation

```python
# In api.py:
import asyncio
from fastapi import BackgroundTasks
from uuid import uuid4

_pipeline_jobs: dict[str, dict] = {}   # job_id → {status, stage, results, error}

@app.post("/replica/pipeline/async")
async def replica_pipeline_async(body: PipelineIn, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    _pipeline_jobs[job_id] = {
        "status": "queued", "stage": None,
        "results": {}, "error": None, "pipeline": body.pipeline
    }
    background_tasks.add_task(_run_pipeline_job, job_id, body)
    return {"job_id": job_id, "status": "queued"}

async def _run_pipeline_job(job_id: str, body: PipelineIn):
    job = _pipeline_jobs[job_id]
    job["status"] = "running"
    # ... same sequential stage logic as synchronous pipeline ...
    # Updates job["stage"] and job["results"] after each stage
    # Sets job["status"] = "complete" or "error" at end
```

### SSE stream format

```
data: {"event": "stage_start", "stage": "writer", "job_id": "..."}
data: {"event": "stage_complete", "stage": "writer", "result": "...", "elapsed_s": 45.2}
data: {"event": "stage_start", "stage": "critic", "job_id": "..."}
data: {"event": "stage_complete", "stage": "critic", "result": "...", "elapsed_s": 38.7}
data: {"event": "pipeline_complete", "job_id": "...", "total_elapsed_s": 84.1}
```

### Existing sync endpoint preserved

`POST /replica/pipeline` remains unchanged. Callers that can tolerate blocking (internal evolution hook via `rep.pipeline()`) continue using it. External callers (simulation harnesses, Telegram bot, future UI) use the async endpoint.

### Job store

In-memory dict is sufficient for the current scale (single process, jobs complete in minutes). Jobs are pruned after 1 hour:

```python
def _prune_old_jobs():
    cutoff = time.time() - 3600
    to_delete = [jid for jid, j in _pipeline_jobs.items() 
                 if j.get("completed_at", time.time()) < cutoff]
    for jid in to_delete:
        del _pipeline_jobs[jid]
```

---

## Simulation harness update (Sim 8)

Update `sim7/run_sim7.py` → `sim8/run_sim8.py` to use the async endpoint:

```python
def run_task_async(task: dict) -> dict:
    # POST to /replica/pipeline/async → get job_id
    # Poll GET /replica/pipeline/{job_id} every 10s
    # Timeout after 600s (10 min) — realistic for 3-stage pipelines
    # Return results dict
```

---

## Drafter role

The async model opens a new use for the drafter: **speculative stage pre-generation**.

When Stage 1 (writer) completes, the drafter can immediately begin Stage 2 (critic) using the writer output while the main model is momentarily free. The main model then only needs to verify/complete the drafter's partial output rather than start from scratch.

This is speculative decoding applied at the pipeline level rather than the token level:
- Drafter: runs Stage N+1 speculatively while Stage N result is being logged/written
- Main model: validates + extends the draft in Stage N+1 rather than full generation

Add `speculative_stage: bool = False` to `PipelineStage` to opt in per stage.

---

## Config

```yaml
pipeline:
  async_enabled: true
  job_ttl_s: 3600           # prune completed jobs after 1 hour
  speculative_stages: false  # drafter pre-runs next stage (experimental)
```

---

## Tests

- `test_async_pipeline_queues_job` — POST returns job_id with status=queued
- `test_async_pipeline_status_running` — GET shows running + current stage
- `test_async_pipeline_complete` — GET shows complete + full results
- `test_async_pipeline_prune` — old jobs removed after TTL
- `test_async_pipeline_cancel` — DELETE stops background task
- `test_sse_stream_events` — SSE endpoint yields correct event sequence
