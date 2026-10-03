"""api/routers/replicas.py — skills/routines/replicas and pipeline routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Pipeline job
state (_pipeline_jobs) lives here (only used by these functions). Route fns
resolve live api state through the _api() seam. Behavior-identical;
re-exported from api.__init__.
"""
import sys
import importlib
import json
import time
import uuid

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from ..schemas import PipelineIn, TaskIn, NamedReplicaIn, ReplicaMessageIn, PipelineJobStatus

router = APIRouter()

# job_id → job state dict (moved from api.__init__ — only used by pipeline routes)
_pipeline_jobs: dict[str, dict] = {}


def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


@router.get("/skills")
def list_skills():
    _m = _api()
    return [{"name": s["name"], "description": s["description"], "commands": s.get("commands", []), "is_core": s.get("is_core", False)} for s in _m.agent._skills]

@router.get("/routines")
def list_routines():
    _m = _api()
    return [{"name": r["name"], "description": r["description"], "trigger": r["trigger"]} for r in _m.agent._routines]

@router.post("/replica/spawn")
def spawn_replica(body: TaskIn):
    """Spawn a specialist replica if VRAM allows."""
    _m = _api()
    r = _m.rep.spawn(body.role, body.task, adapter_path=body.adapter_path)
    if r is None:
        return {"status": "rejected", "reason": "VRAM limit or replica cap reached"}
    return {"status": "spawned", "role": r.role, "name": r.name}

@router.get("/replica/active")
def active_replicas():
    _m = _api()
    return [{
        "name": r.name,
        "role": r.role,
        "task": r.task[:80],
        "done": r.done,
        "persistent": r.persistent,
        "adapter_path": r.adapter_path,
    } for r in _m.rep.active()]

@router.post("/replica/named")
def spawn_named_replica(body: NamedReplicaIn):
    """Spawn a named persistent replica with optional brief injection."""
    _m = _api()
    r = _m.rep.spawn_named(
        name=body.name,
        role=body.role,
        brief_path=body.brief_path,
        custom_prompt=body.custom_prompt,
        workspace=body.workspace,
        tools_enabled=body.tools_enabled,
        output_path=body.output_path,
        input_path=body.input_path,
        adapter_path=body.adapter_path,
        slot=body.slot,
    )
    if r is None:
        return {"status": "rejected", "reason": "VRAM limit or replica cap reached"}
    return {"status": "spawned", "name": r.name, "role": r.role,
            "persistent": r.persistent, "tools_enabled": r.tools_enabled,
            "adapter_path": r.adapter_path}

@router.post("/replica/pipeline")
def replica_pipeline(body: PipelineIn):
    """ADR-008 Change 2: Run a multi-stage writer→critic pipeline.
    _m = _api()

    Stages run sequentially. Each stage's output is passed as context
    to the next stage (via input_from). Results are written to output_path
    when specified. Replicas are cleaned up after the pipeline completes.
    """
    import logging
    logger = logging.getLogger(__name__)
    results: dict[str, str] = {}
    stage_outputs: dict[str, str] = {}

    for stage in body.stages:
        # Build message — inject previous stage output if input_from specified
        task_text = stage.task
        if stage.input_from and stage.input_from in stage_outputs:
            task_text = (
                f"{stage.task}\n\n"
                f"## Output from '{stage.input_from}' stage\n"
                f"{stage_outputs[stage.input_from]}"
            )
        # Also inject from output_path on disk if input_from stage wrote one
        input_path = None
        if stage.input_from:
            prev = next((s for s in body.stages if s.name == stage.input_from), None)
            if prev and prev.output_path:
                input_path = prev.output_path

        # Ensure no stale replica with this name
        _m.rep.stop(f"pipeline-{body.pipeline}-{stage.name}")

        r = _m.rep.spawn_named(
            name=f"pipeline-{body.pipeline}-{stage.name}",
            role=stage.role,
            custom_prompt=stage.brief,
            workspace=stage.workspace,
            tools_enabled=stage.tools,
            output_path=stage.output_path,
            input_path=input_path,
            adapter_path=stage.adapter_path,
        )
        if r is None:
            return JSONResponse(
                {"status": "error", "stage": stage.name,
                 "reason": "VRAM limit or replica cap reached"},
                status_code=503
            )

        reply = r.message(task_text)
        results[stage.name] = reply
        stage_outputs[stage.name] = reply
        logger.info(f"[pipeline:{body.pipeline}] stage '{stage.name}' done ({len(reply)} chars)")

        # Clean up stage replica immediately to free capacity for next stage
        _m.rep.stop(f"pipeline-{body.pipeline}-{stage.name}")

    return {"status": "complete", "pipeline": body.pipeline, "results": results}

def _prune_pipeline_jobs():
    """Remove completed jobs older than TTL."""
    _m = _api()
    ttl = _m._cfg.get("pipeline", {}).get("job_ttl_s", 3600)
    cutoff = time.time() - ttl
    to_delete = [
        jid for jid, j in _pipeline_jobs.items()
        if j.get("completed_at", time.time()) < cutoff
    ]
    for jid in to_delete:
        del _pipeline_jobs[jid]

async def _run_pipeline_job(job_id: str, body: PipelineIn):
    """Background task: execute pipeline stages, update job state."""
    _m = _api()
    import logging as _log
    _logger = _log.getLogger(__name__)
    job = _pipeline_jobs.get(job_id)
    if not job:
        return
    job["status"] = "running"
    results: dict[str, str] = {}
    stage_outputs: dict[str, str] = {}

    try:
        for stage in body.stages:
            if job.get("status") == "cancelled":

                break
            job["stage"] = stage.name

            # Build task text with input injection
            task_text = stage.task
            if stage.input_from and stage.input_from in stage_outputs:
                task_text = (
                    f"{stage.task}\n\n"
                    f"## Output from '{stage.input_from}' stage\n"
                    f"{stage_outputs[stage.input_from]}"
                )

            _m.rep.stop(f"pipeline-{body.pipeline}-{stage.name}")
            r = _m.rep.spawn_named(
                name=f"pipeline-{body.pipeline}-{stage.name}",
                role=stage.role,
                custom_prompt=stage.brief,
                workspace=stage.workspace,
                tools_enabled=stage.tools,
                output_path=stage.output_path,
                adapter_path=stage.adapter_path,
            )
            if r is None:
                raise RuntimeError(f"Stage '{stage.name}': could not spawn replica (VRAM/cap)")

            reply = r.message(task_text)
            results[stage.name] = reply
            stage_outputs[stage.name] = reply
            job["results"] = dict(results)
            _logger.info(f"[pipeline-async:{body.pipeline}:{job_id[:8]}] stage '{stage.name}' done")
            _m.rep.stop(f"pipeline-{body.pipeline}-{stage.name}")

        if job.get("status") != "cancelled":
            job["status"] = "complete"
        job["completed_at"] = time.time()

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["completed_at"] = time.time()
        _logger.error(f"[pipeline-async:{job_id[:8]}] error: {e}")

@router.post("/replica/pipeline/async")
async def replica_pipeline_async(body: PipelineIn, background_tasks: BackgroundTasks):
    """ADR-012: Async pipeline — returns job_id immediately, runs in background."""
    _m = _api()
    _prune_pipeline_jobs()
    job_id = str(uuid.uuid4())
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": None,
        "results": {},
        "error": None,
        "pipeline": body.pipeline,
        "created_at": time.time(),
        "completed_at": None,
    }
    background_tasks.add_task(_run_pipeline_job, job_id, body)
    return {"job_id": job_id, "status": "queued"}

@router.get("/replica/pipeline/{job_id}")
async def get_pipeline_job(job_id: str):
    """ADR-012: Get current job state."""
    _m = _api()
    job = _pipeline_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job

@router.get("/replica/pipeline/{job_id}/stream")
async def stream_pipeline_job(job_id: str):
    """ADR-012: SSE stream of stage completion events."""
    _m = _api()
    import asyncio

    async def event_generator():
        last_stage = None
        last_result_count = 0
        while True:
            job = _pipeline_jobs.get(job_id)
            if not job:
                yield f"data: {{\"event\": \"error\", \"message\": \"job not found\"}}\n\n"
                break

            current_stage = job.get("stage")
            results = job.get("results", {})

            # Emit stage_start when stage changes
            if current_stage and current_stage != last_stage:
                yield f"data: {json.dumps({'event': 'stage_start', 'stage': current_stage, 'job_id': job_id})}\n\n"
                last_stage = current_stage

            # Emit stage_complete for newly completed stages
            if len(results) > last_result_count:
                new_stages = list(results.keys())[last_result_count:]
                for sname in new_stages:
                    yield f"data: {json.dumps({'event': 'stage_complete', 'stage': sname, 'result': results[sname][:200], 'job_id': job_id})}\n\n"
                last_result_count = len(results)

            status = job.get("status")
            if status == "complete":
                total = (job.get("completed_at") or time.time()) - job.get("created_at", time.time())
                yield f"data: {json.dumps({'event': 'pipeline_complete', 'job_id': job_id, 'total_elapsed_s': round(total, 1)})}\n\n"
                break
            elif status in ("error", "cancelled"):
                yield f"data: {json.dumps({'event': 'pipeline_error', 'job_id': job_id, 'error': job.get('error', status)})}\n\n"
                break

            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.delete("/replica/pipeline/{job_id}")
async def cancel_pipeline_job(job_id: str):
    """ADR-012: Cancel a queued or running job."""
    _m = _api()
    job = _pipeline_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    job["status"] = "cancelled"
    job["completed_at"] = time.time()
    return {"job_id": job_id, "status": "cancelled"}

@router.post("/replica/{name}/message")
def message_replica(name: str, body: ReplicaMessageIn):
    """Send a message to a named persistent replica."""
    _m = _api()
    r = _m.rep.get(name)
    if r is None:
        return JSONResponse({"error": f"Replica '{name}' not found"}, status_code=404)
    if not r.persistent:
        return JSONResponse({"error": "Replica is not in persistent mode"}, status_code=400)
    reply = r.message(body.message)
    return {"name": name, "reply": reply}

@router.delete("/replica/{name}")
def stop_replica(name: str):
    """Stop and remove a named replica."""
    _m = _api()
    if _m.rep.stop(name):
        return {"status": "stopped", "name": name}
    return JSONResponse({"error": f"Replica '{name}' not found"}, status_code=404)

@router.get("/replica/{name}/status")
def replica_status(name: str):
    """Get status of a named replica."""
    _m = _api()
    r = _m.rep.get(name)
    if r is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "name": r.name,
        "role": r.role,
        "persistent": r.persistent,
        "done": r.done,
        "brief_path": r.brief_path,
        "history_turns": len(r.history),
        "adapter_path": r.adapter_path,
    }
