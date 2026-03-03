from __future__ import annotations

import asyncio
import tempfile
import subprocess
import sys
import time
import json
import re
import shutil
import socket
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services.pipeline import (
    build_scenes,
    configured_model_stack,
    generate_narration_audio_with_provider,
    generate_scene_images_with_provider,
    get_model_capabilities,
    probe_system_capabilities,
    refine_script_with_provider,
    render_video,
    write_manifest,
    write_refined_script,
    write_render_report,
)
from .services.training import LocalFineTuneTrainer

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSET_ROOT = BASE_DIR / "runtime" / "assets"
ASSET_ROOT.mkdir(parents=True, exist_ok=True)
STATE_FILE = BASE_DIR / "runtime" / "state.json"
EXPORT_ROOT = BASE_DIR / "runtime" / "exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_SCRIPT = "\n".join(
    [
        "Future of work needs private, low-latency content creation.",
        "Two teammates co-edit a product story in real time.",
        "AI generates visual scenes and narration from the script.",
        "The system exports a short demo package for final review.",
    ]
)

DEFAULT_WORKSPACE_FILE = "src/main.py"
DEFAULT_CODE = "\n".join(
    [
        "def main() -> None:",
        "    message = \"Slingshot collaborative coding workspace is ready.\"",
        "    print(message)",
        "",
        "",
        "if __name__ == \"__main__\":",
        "    main()",
    ]
)

app = FastAPI(title="Slingshot Prototype API", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSET_ROOT)), name="assets")


class ProjectCreateRequest(BaseModel):
    title: str = Field(default="Slingshot Demo Project", max_length=120)
    initial_script: str = Field(default=DEFAULT_SCRIPT, max_length=8000)


class ScriptUpdateRequest(BaseModel):
    script: str = Field(default="", max_length=8000)


class JobCreateRequest(BaseModel):
    kind: Literal["pipeline", "fine_tune"] = "pipeline"

class ExecutionRequest(BaseModel):
    language: str = Field(default="python")
    code: str = Field(..., description="Code to execute")
    timeout: int = Field(default=5, ge=1, le=60)

class ExecutionResponse(BaseModel):
    run_id: str
    stdout: str
    stderr: str
    exit_code: int
    time_ms: int
    status: str


class RunCleanupRequest(BaseModel):
    keep_latest: int = Field(default=3, ge=0, le=200)
    include_ungrouped: bool = False


class WorkspaceFolderCreateRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=400)


@app.post("/api/projects/{project_id}/workspace/folders")
async def create_workspace_folder(project_id: str, payload: WorkspaceFolderCreateRequest) -> dict[str, Any]:
    raw = payload.path
    try:
        folder_path = _normalize_workspace_path(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not folder_path:
        raise HTTPException(status_code=400, detail="Invalid folder path")

    keep_path = f"{folder_path}/.keep"
    now = utc_now()
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict):
            files = {}
            workspace["files"] = files
        if keep_path not in files:
            files[keep_path] = {
                "content": "",
                "updated_at": now,
                "language": "plaintext",
            }
        workspace["updated_at"] = now
        project["workspace"] = workspace
        _persist_state_unlocked()
        snapshot = _workspace_snapshot(project)

    if snapshot is not None:
        await _broadcast_workspace_state(project_id, snapshot)
    return {"folder": folder_path, "keep": keep_path}


@app.post("/api/migrate-workspace")
async def migrate_workspace() -> dict[str, Any]:
    migrated = 0
    now = utc_now()
    async with state_lock:
        for project_id, project in list(projects.items()):
            if project.get("workspace"):
                continue
            script = project.get("script", "")
            if not script:
                continue
            workspace = {
                "active_file": "src/main.py",
                "updated_at": now,
                "files": {
                    "src/main.py": {
                        "content": script,
                        "updated_at": now,
                        "language": "python",
                    }
                },
            }
            project["workspace"] = workspace
            _persist_state_unlocked()
            migrated += 1
    return {"migrated": migrated}


class WorkspaceFileCreateRequest(BaseModel):
    path: str = Field(min_length=1, max_length=260)
    content: str = Field(default="", max_length=200000)


class WorkspaceFileUpdateRequest(BaseModel):
    path: str = Field(min_length=1, max_length=260)
    content: str = Field(default="", max_length=200000)


class WorkspaceFileRenameRequest(BaseModel):
    old_path: str = Field(min_length=1, max_length=260)
    new_path: str = Field(min_length=1, max_length=260)


class WorkspaceActiveFileRequest(BaseModel):
    path: str = Field(min_length=1, max_length=260)


projects: dict[str, dict[str, Any]] = {}
jobs: dict[str, dict[str, Any]] = {}
rooms: dict[str, dict[str, WebSocket]] = {}
state_lock = asyncio.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def detect_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidate = str(sock.getsockname()[0])
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass

    try:
        host = socket.gethostname()
        addresses = socket.getaddrinfo(host, None, family=socket.AF_INET)
        for address in addresses:
            candidate = str(address[4][0])
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass

    return None


def _project_or_404(project_id: str) -> dict[str, Any]:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _job_or_404(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _normalize_workspace_path(path: str) -> str:
    candidate = path.replace("\\", "/").strip().strip("/")
    if not candidate:
        raise ValueError("Path is required")

    segments = candidate.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("Invalid path")

    if any(len(segment) > 120 for segment in segments):
        raise ValueError("Path segment too long")

    return candidate


def _default_workspace(initial_content: str) -> dict[str, Any]:
    now = utc_now()
    content = initial_content.strip() if initial_content.strip() else DEFAULT_CODE
    return {
        "active_file": DEFAULT_WORKSPACE_FILE,
        "updated_at": now,
        "files": {
            DEFAULT_WORKSPACE_FILE: {
                "content": content,
                "updated_at": now,
            }
        },
    }


def _normalize_workspace(raw_workspace: Any, script_fallback: str) -> dict[str, Any]:
    now = utc_now()
    fallback_content = script_fallback.strip() if script_fallback.strip() else DEFAULT_CODE

    if not isinstance(raw_workspace, dict):
        return _default_workspace(fallback_content)

    raw_files = raw_workspace.get("files")
    normalized_files: dict[str, dict[str, str]] = {}
    if isinstance(raw_files, dict):
        for raw_path, raw_item in raw_files.items():
            if not isinstance(raw_path, str):
                continue
            try:
                normalized_path = _normalize_workspace_path(raw_path)
            except ValueError:
                continue

            content = ""
            updated_at = now
            if isinstance(raw_item, dict):
                content = str(raw_item.get("content", ""))
                updated_at = str(raw_item.get("updated_at", now)) or now
            elif isinstance(raw_item, str):
                content = raw_item

            normalized_files[normalized_path] = {
                "content": content,
                "updated_at": updated_at,
            }

    if not normalized_files:
        normalized_files = {
            DEFAULT_WORKSPACE_FILE: {
                "content": fallback_content,
                "updated_at": now,
            }
        }

    requested_active_file = raw_workspace.get("active_file")
    active_file = ""
    if isinstance(requested_active_file, str):
        try:
            normalized_active = _normalize_workspace_path(requested_active_file)
            if normalized_active in normalized_files:
                active_file = normalized_active
        except ValueError:
            active_file = ""

    if not active_file:
        active_file = sorted(normalized_files.keys())[0]

    updated_at = str(raw_workspace.get("updated_at", now)) or now
    return {
        "active_file": active_file,
        "updated_at": updated_at,
        "files": normalized_files,
    }


def _sync_legacy_script_from_workspace_unlocked(project: dict[str, Any]) -> None:
    workspace = project.get("workspace")
    if not isinstance(workspace, dict):
        project["script"] = DEFAULT_CODE
        return

    files = workspace.get("files")
    active_file = str(workspace.get("active_file", ""))
    if not isinstance(files, dict) or not files:
        project["script"] = DEFAULT_CODE
        return

    active_entry = files.get(active_file)
    if not isinstance(active_entry, dict):
        first_entry = next(iter(files.values()))
        active_entry = first_entry if isinstance(first_entry, dict) else {"content": DEFAULT_CODE}
    project["script"] = str(active_entry.get("content", DEFAULT_CODE))


def _ensure_workspace_unlocked(project: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_workspace(project.get("workspace"), str(project.get("script", "")))
    project["workspace"] = normalized
    _sync_legacy_script_from_workspace_unlocked(project)
    return normalized


def _workspace_snapshot(project: dict[str, Any]) -> dict[str, Any]:
    workspace = _ensure_workspace_unlocked(project)
    files = workspace.get("files", {})
    normalized_files: dict[str, dict[str, str]] = {}
    if isinstance(files, dict):
        for file_path in sorted(files.keys()):
            item = files[file_path]
            if not isinstance(item, dict):
                continue
            normalized_files[str(file_path)] = {
                "content": str(item.get("content", "")),
                "updated_at": str(item.get("updated_at", utc_now())),
            }

    return {
        "active_file": str(workspace.get("active_file", "")),
        "updated_at": str(workspace.get("updated_at", utc_now())),
        "files": normalized_files,
    }


def _workspace_context_for_jobs(project: dict[str, Any]) -> str:
    workspace = _ensure_workspace_unlocked(project)
    files = workspace.get("files", {})
    if not isinstance(files, dict) or not files:
        return str(project.get("script", DEFAULT_CODE))

    chunks: list[str] = []
    for file_path in sorted(files.keys()):
        item = files[file_path]
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        chunks.append(f"FILE: {file_path}\n{content}")

    if not chunks:
        return str(project.get("script", DEFAULT_CODE))

    joined = "\n\n".join(chunks)
    return joined[:60000]


def _asset_url(path: Path) -> str:
    relative = path.relative_to(ASSET_ROOT)
    return f"/assets/{relative.as_posix()}"


def _asset_record(project_id: str, path: Path, kind: str, label: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "asset_id": str(uuid.uuid4()),
        "project_id": project_id,
        "kind": kind,
        "label": label,
        "url": _asset_url(path),
        "filename": path.name,
        "metadata": metadata or {},
        "created_at": utc_now(),
    }


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    raw_result = job.get("result", {})
    result_payload = raw_result if isinstance(raw_result, dict) else {}
    return {
        "id": str(job.get("id", "")),
        "project_id": str(job.get("project_id", "")),
        "kind": str(job.get("kind", "pipeline")),
        "status": str(job.get("status", "queued")),
        "progress": int(job.get("progress", 0)),
        "logs": list(job.get("logs", [])),
        "result": dict(result_payload),
        "error": job.get("error"),
        "created_at": str(job.get("created_at", "")),
        "updated_at": str(job.get("updated_at", "")),
    }


def _project_snapshot(project: dict[str, Any]) -> dict[str, Any]:
    workspace = _normalize_workspace(project.get("workspace"), str(project.get("script", "")))
    active_file = str(workspace.get("active_file", ""))
    active_entry = workspace.get("files", {}).get(active_file, {})
    active_content = str(active_entry.get("content", project.get("script", DEFAULT_CODE)))

    return {
        "id": str(project.get("id", "")),
        "title": str(project.get("title", "Slingshot Demo Project")),
        "script": active_content,
        "workspace": workspace,
        "assets": list(project.get("assets", [])),
        "active_clients": [],
        "created_at": str(project.get("created_at", utc_now())),
        "updated_at": str(project.get("updated_at", utc_now())),
    }


def _persist_state_unlocked() -> None:
    payload = {
        "saved_at": utc_now(),
        "projects": [_project_snapshot(project) for project in projects.values()],
        "jobs": [_job_snapshot(job) for job in jobs.values()],
    }
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _load_state_from_disk() -> None:
    if not STATE_FILE.exists():
        return

    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    loaded_projects: dict[str, dict[str, Any]] = {}
    for raw_project in payload.get("projects", []):
        if not isinstance(raw_project, dict):
            continue
        project = _project_snapshot(raw_project)
        project_id = project["id"]
        if not project_id:
            continue
        loaded_projects[project_id] = project

    loaded_jobs: dict[str, dict[str, Any]] = {}
    for raw_job in payload.get("jobs", []):
        if not isinstance(raw_job, dict):
            continue
        job = _job_snapshot(raw_job)
        job_id = job["id"]
        if not job_id:
            continue
        if job["status"] in {"queued", "running"}:
            now = utc_now()
            logs = list(job.get("logs", []))
            logs.append(f"[{now}] Marked failed after server restart.")
            job["logs"] = logs
            job["status"] = "failed"
            job["progress"] = 100
            job["error"] = "Server restarted while this job was active."
            job["updated_at"] = now
        loaded_jobs[job_id] = job

    projects.update(loaded_projects)
    jobs.update(loaded_jobs)


_load_state_from_disk()


def _latest_project_job_snapshot(project_id: str) -> dict[str, Any] | None:
    project_jobs = [job for job in jobs.values() if str(job.get("project_id", "")) == project_id]
    if not project_jobs:
        return None

    active_jobs = [job for job in project_jobs if str(job.get("status", "")) in {"queued", "running"}]
    if active_jobs:
        return _job_snapshot(max(active_jobs, key=lambda item: str(item.get("created_at", ""))))

    return _job_snapshot(max(project_jobs, key=lambda item: str(item.get("created_at", ""))))


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(job.get("id", "")),
        "kind": str(job.get("kind", "pipeline")),
        "status": str(job.get("status", "queued")),
        "progress": int(job.get("progress", 0)),
        "created_at": str(job.get("created_at", "")),
        "updated_at": str(job.get("updated_at", "")),
    }


def _asset_folder_name(asset: dict[str, Any]) -> str | None:
    url = str(asset.get("url", ""))
    if not url:
        return None

    clean_url = url.split("?", 1)[0]
    parts = [part for part in clean_url.split("/") if part]
    if len(parts) < 4 or parts[0] != "assets":
        return None
    return parts[2]


def _infer_run_id_from_folder(folder: str) -> str | None:
    match = re.match(r"^(run|finetune)_(\d{8})_(\d{6})_[0-9a-f]{6}$", folder)
    if not match:
        return None
    prefix, date_part, time_part = match.groups()
    return f"{prefix}_{date_part}_{time_part}"


def _run_id_from_asset(asset: dict[str, Any], folder: str) -> str | None:
    metadata = asset.get("metadata")
    if isinstance(metadata, dict):
        run_id = metadata.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return _infer_run_id_from_folder(folder)


def _run_kind_from_group(folder: str, assets_in_group: list[dict[str, Any]]) -> str:
    for asset in assets_in_group:
        metadata = asset.get("metadata")
        if isinstance(metadata, dict):
            job_kind = metadata.get("job_kind")
            if isinstance(job_kind, str) and job_kind:
                return job_kind

    if folder.startswith("finetune_"):
        return "fine_tune"
    if folder.startswith("run_"):
        return "pipeline"
    return "mixed"


def _build_run_records(project_id: str, project_assets: list[dict[str, Any]], project_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for asset in project_assets:
        folder = _asset_folder_name(asset)
        if folder is None:
            folder = "ungrouped"

        if folder not in groups:
            groups[folder] = {
                "folder": folder,
                "run_id": None,
                "kind": "mixed",
                "assets": [],
                "created_at": str(asset.get("created_at", "")),
                "updated_at": str(asset.get("created_at", "")),
                "job": None,
            }

        group = groups[folder]
        group["assets"].append(asset)
        created_at = str(asset.get("created_at", ""))
        if created_at and (not group["updated_at"] or created_at > str(group["updated_at"])):
            group["updated_at"] = created_at
        if created_at and (not group["created_at"] or created_at < str(group["created_at"])):
            group["created_at"] = created_at

        if group["run_id"] is None:
            run_id = _run_id_from_asset(asset, folder)
            if run_id:
                group["run_id"] = run_id

    jobs_by_run_id: dict[str, dict[str, Any]] = {}
    for job in project_jobs:
        result = job.get("result")
        if not isinstance(result, dict):
            continue
        run_id = result.get("run_id")
        if isinstance(run_id, str) and run_id:
            jobs_by_run_id[run_id] = job

    runs: list[dict[str, Any]] = []
    for group in groups.values():
        assets_in_group = list(group["assets"])
        run_id = group.get("run_id")
        group["kind"] = _run_kind_from_group(str(group["folder"]), assets_in_group)
        if isinstance(run_id, str) and run_id and run_id in jobs_by_run_id:
            group["job"] = _job_summary(jobs_by_run_id[run_id])

        runs.append(
            {
                "project_id": project_id,
                "folder": str(group["folder"]),
                "run_id": run_id,
                "kind": str(group["kind"]),
                "asset_count": len(assets_in_group),
                "created_at": str(group.get("created_at", "")),
                "updated_at": str(group.get("updated_at", "")),
                "job": group.get("job"),
                "assets": assets_in_group,
            }
        )

    runs.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return runs


def _safe_run_folder_name(run_folder: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", run_folder))


def _extension_language(name: str) -> str:
    n = name.lower()
    if n.endswith('.py'):
        return 'python'
    if n.endswith('.js') or n.endswith('.jsx') or n.endswith('.ts') or n.endswith('.tsx'):
        return 'javascript'
    if n.endswith('.json'):
        return 'json'
    if n.endswith('.md'):
        return 'markdown'
    if n.endswith('.yaml') or n.endswith('.yml'):
        return 'yaml'
    if n.endswith('.go'):
        return 'go'
    if n.endswith('.rs'):
        return 'rust'
    if n.endswith('.cpp') or n.endswith('.cc'):
        return 'cpp'
    return 'plaintext'


def _find_child(parent: dict[str, Any], name: str, type_filter: str | None = None):
    for child in parent.get('children', []) or []:
        if child.get('name') == name and (type_filter is None or child.get('type') == type_filter):
            return child
    return None


def _build_tree_from_paths(files: dict[str, Any]) -> dict[str, Any]:
    root = {"name": "", "path": "", "type": "folder", "children": []}
    if not isinstance(files, dict):
        return root

    for full_path in sorted(files.keys()):
        parts = [p for p in full_path.split('/') if p]
        current = root
        for i, part in enumerate(parts):
            is_file = i == len(parts) - 1
            path_acc = '/'.join(parts[: i+1])
            if is_file:
                # ensure file node under current
                file_node = _find_child(current, part, 'file')
                if not file_node:
                    lang = _extension_language(part)
                    entry = files[full_path]
                    content = entry.get('content', '') if isinstance(entry, dict) else str(entry)
                    file_node = {"type": "file", "name": part, "path": path_acc, "language": lang, "content": content, "updated_at": entry.get('updated_at','') if isinstance(entry, dict) else ''}
                    current.setdefault('children', []).append(file_node)
                # do not descend further for file
            else:
                folder_node = _find_child(current, part, 'folder')
                if not folder_node:
                    folder_node = {"type": "folder", "name": part, "path": path_acc, "children": []}
                    current.setdefault('children', []).append(folder_node)
                current = folder_node
    return root


@app.get("/api/projects/{project_id}/workspace/tree")
async def workspace_tree(project_id: str) -> dict[str, Any]:
    _project_or_404(project_id)
    project = projects[project_id]
    workspace = project.get("workspace", {})
    files = workspace.get("files", {})
    tree = _build_tree_from_paths(files)
    return {
        "root": tree,
        "active_file": workspace.get("active_file"),
        "updated_at": workspace.get("updated_at"),
    }


@app.get("/api/projects/{project_id}/workspace/files/{path:path}")
async def get_workspace_file(project_id: str, path: str) -> dict[str, Any]:
    _project_or_404(project_id)
    workspace = projects[project_id].get("workspace", {})
    files = workspace.get("files", {}) if isinstance(workspace, dict) else {}
    if not isinstance(files, dict) or path not in files:
        raise HTTPException(status_code=404, detail="File not found in workspace")
    entry = files[path]
    content = entry.get("content", "") if isinstance(entry, dict) else ""
    language = entry.get("language", _extension_language(path)) if isinstance(entry, dict) else _extension_language(path)
    updated_at = entry.get("updated_at", workspace.get("updated_at", utc_now())) if isinstance(entry, dict) else workspace.get("updated_at", utc_now())
    return {"path": path, "content": content, "language": language, "updated_at": updated_at}


@app.patch("/api/projects/{project_id}/workspace/files/move")
async def move_workspace_file(project_id: str, payload: WorkspaceFileRenameRequest) -> dict[str, Any]:
    try:
        old_path = _normalize_workspace_path(payload.old_path)
        new_path = _normalize_workspace_path(payload.new_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not old_path or not new_path:
        raise HTTPException(status_code=400, detail="Invalid paths")

    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict) or old_path not in files:
            raise HTTPException(status_code=404, detail="Source file not found")
        if new_path in files:
            raise HTTPException(status_code=409, detail="Destination file already exists")

        file_entry = files.pop(old_path)
        if isinstance(file_entry, dict):
            file_entry["updated_at"] = utc_now()
        else:
            file_entry = {"content": str(file_entry), "updated_at": utc_now(), "language": _extension_language(new_path)}
        file_entry["path"] = new_path
        files[new_path] = file_entry
        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        updated = _workspace_snapshot(project)

    if updated:
        await _broadcast_workspace_state(project_id, updated)
    return updated


def _run_export_path(project_id: str, run_folder: str) -> Path:
    return EXPORT_ROOT / f"{project_id}_{run_folder}.zip"


def _remove_run_assets_unlocked(project_id: str, run_folder: str) -> int:
    project = projects.get(project_id)
    if project is None:
        return 0

    existing_assets = list(project.get("assets", []))
    filtered_assets = [asset for asset in existing_assets if _asset_folder_name(asset) != run_folder]
    removed_count = len(existing_assets) - len(filtered_assets)

    if removed_count > 0:
        project["assets"] = filtered_assets
        project["updated_at"] = utc_now()
        _persist_state_unlocked()

    return removed_count


async def _delete_run_files(project_id: str, run_folder: str) -> None:
    if run_folder == "ungrouped":
        return

    run_dir = ASSET_ROOT / project_id / run_folder
    export_path = _run_export_path(project_id, run_folder)

    if run_dir.exists():
        await asyncio.to_thread(shutil.rmtree, run_dir)

    if export_path.exists():
        await asyncio.to_thread(export_path.unlink)


async def _broadcast_workspace_state(project_id: str, workspace: dict[str, Any], exclude_client: str | None = None) -> None:
    await _broadcast_to_room(
        project_id,
        {
            "type": "workspace_state",
            "project_id": project_id,
            "workspace": workspace,
            "updated_at": utc_now(),
        },
        exclude_client=exclude_client,
    )


async def _broadcast_workspace_event(project_id: str, payload: dict[str, Any], exclude_client: str | None = None) -> None:
    event_payload = dict(payload)
    event_payload["project_id"] = project_id
    event_payload["updated_at"] = utc_now()
    await _broadcast_to_room(project_id, event_payload, exclude_client=exclude_client)


async def _broadcast_job_update(project_id: str, job_payload: dict[str, Any]) -> None:
    await _broadcast_to_room(
        project_id,
        {
            "type": "job_update",
            "project_id": project_id,
            "job": job_payload,
            "updated_at": utc_now(),
        },
    )


async def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    project_id: str | None = None
    job_payload: dict[str, Any] | None = None

    async with state_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if message:
            job["logs"].append(f"[{utc_now()}] {message}")
        if error is not None:
            job["error"] = error
        if result is not None:
            job["result"] = result
        job["updated_at"] = utc_now()
        _persist_state_unlocked()
        project_id = str(job.get("project_id", ""))
        job_payload = _job_snapshot(job)

    if project_id and job_payload is not None:
        await _broadcast_job_update(project_id, job_payload)


async def _broadcast_to_room(project_id: str, payload: dict[str, Any], exclude_client: str | None = None) -> None:
    async with state_lock:
        room = rooms.get(project_id, {})
        recipients = [(client_id, ws) for client_id, ws in room.items() if client_id != exclude_client]

    stale_clients: list[str] = []
    for client_id, ws in recipients:
        try:
            await ws.send_json(payload)
        except Exception:
            stale_clients.append(client_id)

    if not stale_clients:
        return

    async with state_lock:
        room = rooms.get(project_id, {})
        for client_id in stale_clients:
            room.pop(client_id, None)
        if project_id in projects:
            projects[project_id]["active_clients"] = sorted(room.keys())


async def _broadcast_presence(project_id: str) -> None:
    async with state_lock:
        room = rooms.get(project_id, {})
        clients = sorted(room.keys())
        if project_id in projects:
            projects[project_id]["active_clients"] = clients
            projects[project_id]["updated_at"] = utc_now()

    await _broadcast_to_room(
        project_id,
        {
            "type": "presence",
            "project_id": project_id,
            "clients": clients,
            "updated_at": utc_now(),
        },
    )


async def _run_pipeline_job(job_id: str) -> None:
    started = time.perf_counter()
    system_metrics = await asyncio.to_thread(probe_system_capabilities)
    backend = str(system_metrics.get("gpu_backend", "none"))
    await _update_job(job_id, status="running", progress=4, message=f"Pipeline started (backend={backend}).")

    async with state_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        project = projects.get(job["project_id"])
        if project is None:
            await _update_job(job_id, status="failed", progress=100, error="Project not found", message="Pipeline aborted.")
            return
        project_id = project["id"]
        script = _workspace_context_for_jobs(project)

    stage_timings: dict[str, float] = {}
    model_runtime: dict[str, Any] = {}
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = ASSET_ROOT / project_id / f"{run_id}_{job_id[:6]}"

    try:
        stage_start = time.perf_counter()
        await _update_job(job_id, progress=12, message="Preparing workspace.")
        await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=True)
        stage_timings["prepare_workspace"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        await _update_job(job_id, progress=24, message="Refining script structure.")
        refined_script, script_runtime = await asyncio.to_thread(refine_script_with_provider, script)
        model_runtime["script"] = script_runtime
        await _update_job(
            job_id,
            progress=30,
            message=f"Script provider: {script_runtime.get('provider', 'mock')} ({script_runtime.get('mode', 'synthetic')}).",
        )
        refined_script_path = await asyncio.to_thread(write_refined_script, run_dir, refined_script)
        stage_timings["refine_script"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        await _update_job(job_id, progress=40, message="Building storyboard scenes.")
        scenes = await asyncio.to_thread(build_scenes, refined_script)
        stage_timings["build_scenes"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        await _update_job(job_id, progress=58, message="Generating scene visuals.")
        scene_images, image_runtime = await asyncio.to_thread(generate_scene_images_with_provider, run_dir, scenes)
        model_runtime["image"] = image_runtime
        await _update_job(
            job_id,
            progress=64,
            message=f"Image provider: {image_runtime.get('provider', 'mock')} ({image_runtime.get('mode', 'synthetic')}).",
        )
        stage_timings["generate_images"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        await _update_job(job_id, progress=74, message="Generating narration audio.")
        duration_s = max(sum(int(scene.get("duration_s", 5)) for scene in scenes), 6)
        narration_audio, tts_runtime = await asyncio.to_thread(
            generate_narration_audio_with_provider,
            run_dir,
            duration_s,
            refined_script,
        )
        model_runtime["tts"] = tts_runtime
        await _update_job(
            job_id,
            progress=80,
            message=f"TTS provider: {tts_runtime.get('provider', 'mock')} ({tts_runtime.get('mode', 'synthetic')}).",
        )
        stage_timings["generate_audio"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        await _update_job(job_id, progress=86, message="Rendering demo output.")
        video_file, render_mode, render_note = await asyncio.to_thread(render_video, run_dir, narration_audio, duration_s)
        stage_timings["render_video"] = round(time.perf_counter() - stage_start, 3)

        render_report: Path | None = None
        if video_file is None:
            render_report = await asyncio.to_thread(write_render_report, run_dir, render_note)

        total_duration_s = round(time.perf_counter() - started, 3)
        run_metrics = {
            "timings_s": stage_timings,
            "total_duration_s": total_duration_s,
            "system_metrics": system_metrics,
            "configured_model_stack": configured_model_stack(),
            "model_runtime": model_runtime,
        }

        stage_start = time.perf_counter()
        await _update_job(job_id, progress=94, message="Writing pipeline manifest.")
        manifest_path = await asyncio.to_thread(
            write_manifest,
            run_dir,
            project_id,
            refined_script,
            scenes,
            scene_images,
            narration_audio,
            render_mode,
            render_note,
            video_file,
            run_metrics,
        )
        stage_timings["write_manifest"] = round(time.perf_counter() - stage_start, 3)

        created_assets: list[dict[str, Any]] = []
        created_assets.append(
            _asset_record(
                project_id,
                refined_script_path,
                kind="script",
                label="Refined Script",
                metadata={"run_id": run_id},
            )
        )
        for scene in scenes:
            scene_index = int(scene["index"])
            image_path = scene_images[scene_index - 1]
            created_assets.append(
                _asset_record(
                    project_id,
                    image_path,
                    kind="image",
                    label=f"Scene {scene_index:02d}",
                    metadata={"caption": scene["caption"], "run_id": run_id},
                )
            )

        created_assets.append(
            _asset_record(
                project_id,
                narration_audio,
                kind="audio",
                label="Narration Audio",
                metadata={"duration_s": duration_s, "run_id": run_id},
            )
        )

        if video_file is not None:
            created_assets.append(
                _asset_record(
                    project_id,
                    video_file,
                    kind="video",
                    label="Demo Video",
                    metadata={"render_mode": render_mode, "run_id": run_id},
                )
            )

        if render_report is not None:
            created_assets.append(
                _asset_record(
                    project_id,
                    render_report,
                    kind="report",
                    label="Render Report",
                    metadata={"run_id": run_id},
                )
            )

        created_assets.append(
            _asset_record(
                project_id,
                manifest_path,
                kind="manifest",
                label="Pipeline Manifest",
                metadata={"run_id": run_id, "render_mode": render_mode},
            )
        )

        async with state_lock:
            if project_id in projects:
                projects[project_id]["assets"] = created_assets + projects[project_id].get("assets", [])
                projects[project_id]["updated_at"] = utc_now()
                _persist_state_unlocked()

        await _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Pipeline completed.",
            result={
                "run_id": run_id,
                "asset_count": len(created_assets),
                "render_mode": render_mode,
                "render_note": render_note,
                "timings_s": stage_timings,
                "total_duration_s": total_duration_s,
                "system_metrics": system_metrics,
                "configured_model_stack": configured_model_stack(),
                "model_runtime": model_runtime,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive path
        await _update_job(
            job_id,
            status="failed",
            progress=100,
            message="Pipeline failed.",
            error=str(exc),
        )


async def _run_fine_tune_job(job_id: str) -> None:
    system_metrics = await asyncio.to_thread(probe_system_capabilities)
    model_stack = configured_model_stack()
    backend = str(system_metrics.get("gpu_backend", "none"))
    await _update_job(job_id, status="running", progress=5, message=f"Fine-tune demo started (backend={backend}).")

    async with state_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        project = projects.get(job["project_id"])
        if project is None:
            await _update_job(job_id, status="failed", progress=100, error="Project not found", message="Fine-tune aborted.")
            return
        project_id = project["id"]
        script = _workspace_context_for_jobs(project)

    run_id = datetime.now(timezone.utc).strftime("finetune_%Y%m%d_%H%M%S")
    report_dir = ASSET_ROOT / project_id / f"{run_id}_{job_id[:6]}"
    report_path = report_dir / "finetune_report.json"

    training_trace: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []

    try:
        await asyncio.to_thread(report_dir.mkdir, parents=True, exist_ok=True)
        await _update_job(job_id, progress=12, message="Preparing local training dataset.")
        trainer = await asyncio.to_thread(LocalFineTuneTrainer, script, report_dir, 6, 0.16)
        dataset_meta = await asyncio.to_thread(trainer.prepare)

        await _update_job(
            job_id,
            progress=22,
            message=(
                "Dataset ready "
                f"(samples={dataset_meta.get('samples_total', 0)}, "
                f"train={dataset_meta.get('train_samples', 0)}, "
                f"val={dataset_meta.get('validation_samples', 0)})."
            ),
        )

        await _update_job(job_id, progress=28, message="Starting gradient-based local adapter training.")
        for epoch in range(1, trainer.epochs + 1):
            epoch_result = await asyncio.to_thread(trainer.train_epoch, epoch)
            checkpoint_path = epoch_result.get("checkpoint_path")
            if isinstance(checkpoint_path, Path):
                checkpoint_paths.append(checkpoint_path)

            trace_entry = {
                "epoch": int(epoch_result.get("epoch", epoch)),
                "train_loss": float(epoch_result.get("train_loss", 0.0)),
                "validation_loss": float(epoch_result.get("validation_loss", 0.0)),
                "weight_norm": float(epoch_result.get("weight_norm", 0.0)),
            }
            training_trace.append(trace_entry)

            progress = 28 + int((epoch / trainer.epochs) * 56)
            await _update_job(
                job_id,
                progress=progress,
                message=(
                    f"Epoch {epoch}/{trainer.epochs} complete "
                    f"(train={trace_entry['train_loss']:.4f}, "
                    f"val={trace_entry['validation_loss']:.4f}, "
                    f"norm={trace_entry['weight_norm']:.4f})."
                ),
            )

        await _update_job(job_id, progress=90, message="Writing training report and checkpoint metadata.")

        trainer_summary = await asyncio.to_thread(trainer.summary)
        final_train_loss = training_trace[-1]["train_loss"] if training_trace else 0.0
        final_validation_loss = training_trace[-1]["validation_loss"] if training_trace else 0.0

        report = {
            "project_id": project_id,
            "run_id": run_id,
            "generated_at": utc_now(),
            "system_metrics": system_metrics,
            "configured_model_stack": model_stack,
            "dataset": dataset_meta,
            "training": {
                **trainer_summary,
                "trace": training_trace,
                "checkpoint_files": [path.name for path in checkpoint_paths],
                "final_train_loss": final_train_loss,
                "final_validation_loss": final_validation_loss,
            },
        }
        await asyncio.to_thread(report_path.write_text, json.dumps(report, indent=2), "utf-8")

        created_assets: list[dict[str, Any]] = []
        created_assets.append(
            _asset_record(
                project_id,
                report_path,
                kind="report",
                label="Fine-tune Report",
                metadata={"run_id": run_id, "job_kind": "fine_tune"},
            )
        )

        for checkpoint_path in checkpoint_paths:
            created_assets.append(
                _asset_record(
                    project_id,
                    checkpoint_path,
                    kind="report",
                    label=f"Checkpoint {checkpoint_path.stem.split('_')[-1].upper()}",
                    metadata={"run_id": run_id, "job_kind": "fine_tune"},
                )
            )

        async with state_lock:
            if project_id in projects:
                projects[project_id]["assets"] = created_assets + projects[project_id].get("assets", [])
                projects[project_id]["updated_at"] = utc_now()
                _persist_state_unlocked()

        await _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Fine-tune job completed.",
            result={
                "run_id": run_id,
                "asset_count": len(created_assets),
                "training_trace": training_trace,
                "final_train_loss": final_train_loss,
                "final_validation_loss": final_validation_loss,
                "system_metrics": system_metrics,
                "configured_model_stack": model_stack,
                "dataset": dataset_meta,
                "checkpoint_count": len(checkpoint_paths),
            },
        )
    except Exception as exc:  # pragma: no cover - defensive path
        await _update_job(
            job_id,
            status="failed",
            progress=100,
            message="Fine-tune job failed.",
            error=str(exc),
        )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "timestamp": utc_now(), "projects": len(projects), "jobs": len(jobs)}


@app.get("/api/network")
async def network_info(request: Request) -> dict[str, Any]:
    return {
        "lan_ip": detect_lan_ip(),
        "request_host": request.url.hostname,
        "request_port": request.url.port,
    }


@app.get("/api/model-capabilities")
async def model_capabilities() -> dict[str, Any]:
    return await asyncio.to_thread(get_model_capabilities)


@app.post("/api/projects")
async def create_project(payload: ProjectCreateRequest) -> dict[str, Any]:
    project_id = uuid.uuid4().hex[:8]
    now = utc_now()
    workspace = _default_workspace(payload.initial_script)
    project = {
        "id": project_id,
        "title": payload.title.strip() or "Slingshot Demo Project",
        "script": payload.initial_script or DEFAULT_CODE,
        "workspace": workspace,
        "assets": [],
        "active_clients": [],
        "created_at": now,
        "updated_at": now,
    }

    async with state_lock:
        projects[project_id] = project
        _persist_state_unlocked()

    return project


@app.get("/api/projects")
async def list_projects() -> list[dict[str, Any]]:
    async with state_lock:
        records: list[dict[str, Any]] = []
        for project in projects.values():
            _ensure_workspace_unlocked(project)
            records.append(dict(project))
    return records


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        _ensure_workspace_unlocked(project)
        return dict(project)


@app.get("/api/projects/{project_id}/workspace")
async def get_workspace(project_id: str) -> dict[str, Any]:
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        snapshot = _workspace_snapshot(project)
    return snapshot


@app.post("/api/projects/{project_id}/workspace/files")
async def create_workspace_file(project_id: str, payload: WorkspaceFileCreateRequest) -> dict[str, Any]:
    try:
        file_path = _normalize_workspace_path(payload.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot: dict[str, Any] | None = None
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict):
            files = {}
            workspace["files"] = files

        if file_path in files:
            raise HTTPException(status_code=409, detail="File already exists")

        files[file_path] = {
            "content": payload.content,
            "updated_at": utc_now(),
        }
        workspace["active_file"] = file_path
        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        snapshot = _workspace_snapshot(project)

    if snapshot is not None:
        await _broadcast_workspace_state(project_id, snapshot)
    return snapshot or {}


@app.put("/api/projects/{project_id}/workspace/files/content")
async def update_workspace_file_content(project_id: str, payload: WorkspaceFileUpdateRequest) -> dict[str, Any]:
    try:
        file_path = _normalize_workspace_path(payload.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot: dict[str, Any] | None = None
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict) or file_path not in files:
            raise HTTPException(status_code=404, detail="File not found")

        files[file_path] = {
            "content": payload.content,
            "updated_at": utc_now(),
        }
        workspace["active_file"] = file_path
        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        snapshot = _workspace_snapshot(project)

    if snapshot is not None:
        await _broadcast_workspace_state(project_id, snapshot)
    return snapshot or {}


@app.patch("/api/projects/{project_id}/workspace/files/rename")
async def rename_workspace_file(project_id: str, payload: WorkspaceFileRenameRequest) -> dict[str, Any]:
    try:
        old_path = _normalize_workspace_path(payload.old_path)
        new_path = _normalize_workspace_path(payload.new_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot: dict[str, Any] | None = None
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict) or old_path not in files:
            raise HTTPException(status_code=404, detail="File not found")
        if new_path in files and new_path != old_path:
            raise HTTPException(status_code=409, detail="Destination file already exists")

        file_entry = files.pop(old_path)
        if not isinstance(file_entry, dict):
            file_entry = {"content": "", "updated_at": utc_now()}
        file_entry["updated_at"] = utc_now()
        files[new_path] = file_entry

        if str(workspace.get("active_file", "")) == old_path:
            workspace["active_file"] = new_path
        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        snapshot = _workspace_snapshot(project)

    if snapshot is not None:
        await _broadcast_workspace_state(project_id, snapshot)
    return snapshot or {}


@app.patch("/api/projects/{project_id}/workspace/active-file")
async def set_workspace_active_file(project_id: str, payload: WorkspaceActiveFileRequest) -> dict[str, Any]:
    try:
        file_path = _normalize_workspace_path(payload.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot: dict[str, Any] | None = None
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict) or file_path not in files:
            raise HTTPException(status_code=404, detail="File not found")

        workspace["active_file"] = file_path
        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        snapshot = _workspace_snapshot(project)

    if snapshot is not None:
        await _broadcast_workspace_state(project_id, snapshot)
    return snapshot or {}


@app.delete("/api/projects/{project_id}/workspace/files/{file_path:path}")
async def delete_workspace_file(project_id: str, file_path: str) -> dict[str, Any]:
    try:
        normalized_path = _normalize_workspace_path(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot: dict[str, Any] | None = None
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workspace = _ensure_workspace_unlocked(project)
        files = workspace.get("files", {})
        if not isinstance(files, dict) or normalized_path not in files:
            raise HTTPException(status_code=404, detail="File not found")

        files.pop(normalized_path, None)
        if not files:
            files[DEFAULT_WORKSPACE_FILE] = {
                "content": DEFAULT_CODE,
                "updated_at": utc_now(),
            }
            workspace["active_file"] = DEFAULT_WORKSPACE_FILE
        elif str(workspace.get("active_file", "")) == normalized_path:
            workspace["active_file"] = sorted(files.keys())[0]

        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        snapshot = _workspace_snapshot(project)

    if snapshot is not None:
        await _broadcast_workspace_state(project_id, snapshot)
    return snapshot or {}


@app.patch("/api/projects/{project_id}/script")
async def update_script(project_id: str, payload: ScriptUpdateRequest) -> dict[str, Any]:
    workspace_payload: dict[str, Any] | None = None
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        workspace = _ensure_workspace_unlocked(project)
        active_file = str(workspace.get("active_file", DEFAULT_WORKSPACE_FILE))
        files = workspace.get("files", {})
        if not isinstance(files, dict):
            files = {}
            workspace["files"] = files

        if active_file not in files:
            active_file = sorted(files.keys())[0] if files else DEFAULT_WORKSPACE_FILE
            workspace["active_file"] = active_file

        files[active_file] = {
            "content": payload.script,
            "updated_at": utc_now(),
        }
        workspace["updated_at"] = utc_now()
        project["workspace"] = workspace
        _sync_legacy_script_from_workspace_unlocked(project)
        project["updated_at"] = utc_now()
        _persist_state_unlocked()
        updated_at = project["updated_at"]
        workspace_payload = _workspace_snapshot(project)

    if workspace_payload is not None:
        await _broadcast_workspace_state(project_id, workspace_payload)
    return {"ok": True, "project_id": project_id, "updated_at": updated_at}


@app.get("/api/projects/{project_id}/assets")
async def list_assets(project_id: str) -> list[dict[str, Any]]:
    project = _project_or_404(project_id)
    return project.get("assets", [])


@app.get("/api/projects/{project_id}/runs")
async def list_project_runs(project_id: str) -> list[dict[str, Any]]:
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        project_assets = list(project.get("assets", []))
        project_jobs = [job for job in jobs.values() if str(job.get("project_id", "")) == project_id]

    return _build_run_records(project_id, project_assets, project_jobs)


@app.delete("/api/projects/{project_id}/runs/{run_folder}")
async def delete_project_run(project_id: str, run_folder: str) -> dict[str, Any]:
    if not _safe_run_folder_name(run_folder):
        raise HTTPException(status_code=400, detail="Invalid run folder name")

    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        project_assets = list(project.get("assets", []))
        project_jobs = [job for job in jobs.values() if str(job.get("project_id", "")) == project_id]

    run_records = _build_run_records(project_id, project_assets, project_jobs)
    target = next((run for run in run_records if str(run.get("folder", "")) == run_folder), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Run folder not found")

    job_info = target.get("job")
    if isinstance(job_info, dict) and str(job_info.get("status", "")) in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Cannot delete a run while its job is active")

    try:
        await _delete_run_files(project_id, run_folder)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete run files: {exc}") from exc

    async with state_lock:
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        removed_assets = _remove_run_assets_unlocked(project_id, run_folder)

    return {
        "ok": True,
        "project_id": project_id,
        "folder": run_folder,
        "run_id": target.get("run_id"),
        "kind": target.get("kind"),
        "removed_assets": removed_assets,
    }


@app.post("/api/projects/{project_id}/runs/cleanup")
async def cleanup_project_runs(project_id: str, payload: RunCleanupRequest) -> dict[str, Any]:
    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        project_assets = list(project.get("assets", []))
        project_jobs = [job for job in jobs.values() if str(job.get("project_id", "")) == project_id]

    run_records = _build_run_records(project_id, project_assets, project_jobs)
    filtered_runs: list[dict[str, Any]] = []
    for run in run_records:
        folder = str(run.get("folder", ""))
        if folder == "ungrouped" and not payload.include_ungrouped:
            continue
        filtered_runs.append(run)

    removable = filtered_runs[payload.keep_latest :]
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_assets_removed = 0

    for run in removable:
        folder = str(run.get("folder", ""))
        job_info = run.get("job")
        if isinstance(job_info, dict) and str(job_info.get("status", "")) in {"queued", "running"}:
            skipped.append(
                {
                    "folder": folder,
                    "reason": "active_job",
                }
            )
            continue

        try:
            await _delete_run_files(project_id, folder)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not delete run '{folder}': {exc}") from exc

        async with state_lock:
            if project_id not in projects:
                raise HTTPException(status_code=404, detail="Project not found")
            removed_assets = _remove_run_assets_unlocked(project_id, folder)

        total_assets_removed += removed_assets
        removed.append(
            {
                "folder": folder,
                "run_id": run.get("run_id"),
                "kind": run.get("kind"),
                "removed_assets": removed_assets,
            }
        )

    return {
        "ok": True,
        "project_id": project_id,
        "keep_latest": payload.keep_latest,
        "include_ungrouped": payload.include_ungrouped,
        "removed_run_count": len(removed),
        "removed_asset_count": total_assets_removed,
        "removed_runs": removed,
        "skipped_runs": skipped,
    }


@app.post("/api/execute", response_model=ExecutionResponse)
def execute_code(req: ExecutionRequest) -> ExecutionResponse:
    # Basic sandbox for Python execution (no external network or file writes beyond tmp)
    if req.language.lower() != 'python':
        return ExecutionResponse(run_id='n/a', stdout='', stderr='Language not supported', exit_code=1, time_ms=0, status='failed')

    run_id = uuid.uuid4().hex
    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'script.py'
        script_path.write_text(req.code, encoding='utf-8')
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=req.timeout,
            )
            stdout = (result.stdout or "").rstrip()
            stderr = (result.stderr or "").rstrip()
            exit_code = result.returncode
        except subprocess.TimeoutExpired as te:
            raw_stdout = te.stdout or ''
            stdout = raw_stdout.decode('utf-8', errors='ignore') if isinstance(raw_stdout, bytes) else str(raw_stdout)
            stderr = f'Timeout after {req.timeout}s'
            exit_code = -1
        except Exception as ex:
            stdout = ''
            stderr = str(ex)
            exit_code = -1
        elapsed = int((time.time() - start) * 1000)
    return ExecutionResponse(run_id=run_id, stdout=stdout, stderr=stderr, exit_code=exit_code, time_ms=elapsed, status='completed' if exit_code == 0 else 'failed')


@app.get("/api/projects/{project_id}/runs/{run_folder}/download")
async def download_project_run(project_id: str, run_folder: str) -> FileResponse:
    _project_or_404(project_id)
    if not _safe_run_folder_name(run_folder):
        raise HTTPException(status_code=400, detail="Invalid run folder name")

    run_dir = ASSET_ROOT / project_id / run_folder
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run folder not found")

    archive_path = _run_export_path(project_id, run_folder)
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(run_dir).as_posix()
                    archive.write(path, arcname=f"{run_folder}/{relative}")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create archive: {exc}") from exc

    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"{run_folder}.zip",
    )


@app.get("/api/projects/{project_id}/jobs")
async def list_project_jobs(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    project_jobs = [job for job in jobs.values() if job["project_id"] == project_id]
    project_jobs.sort(key=lambda item: item["created_at"], reverse=True)
    return project_jobs


@app.post("/api/projects/{project_id}/jobs")
async def create_job(project_id: str, payload: JobCreateRequest) -> dict[str, Any]:
    _project_or_404(project_id)
    job_id = uuid.uuid4().hex[:10]
    now = utc_now()
    job = {
        "id": job_id,
        "project_id": project_id,
        "kind": payload.kind,
        "status": "queued",
        "progress": 0,
        "logs": [f"[{now}] Job queued."],
        "result": {},
        "error": None,
        "created_at": now,
        "updated_at": now,
    }

    queued_payload: dict[str, Any] | None = None
    async with state_lock:
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")

        has_active_job = any(
            current_job.get("project_id") == project_id
            and str(current_job.get("status", "")) in {"queued", "running"}
            for current_job in jobs.values()
        )
        if has_active_job:
            raise HTTPException(status_code=409, detail="Another job is already running for this project")

        jobs[job_id] = job
        queued_payload = _job_snapshot(job)
        _persist_state_unlocked()

    if queued_payload is not None:
        await _broadcast_job_update(project_id, queued_payload)

    if payload.kind == "fine_tune":
        asyncio.create_task(_run_fine_tune_job(job_id))
    else:
        asyncio.create_task(_run_pipeline_job(job_id))
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    return _job_or_404(job_id)


@app.websocket("/ws/{project_id}/{client_id}")
async def project_socket(websocket: WebSocket, project_id: str, client_id: str) -> None:
    await websocket.accept()

    async with state_lock:
        project = projects.get(project_id)
        if project is None:
            await websocket.send_json({"type": "error", "message": "Project not found"})
            await websocket.close(code=4404)
            return

        room = rooms.setdefault(project_id, {})
        room[client_id] = websocket
        project["active_clients"] = sorted(room.keys())
        project["updated_at"] = utc_now()
        workspace_state = _workspace_snapshot(project)
        current_script = project.get("script", "")
        latest_job = _latest_project_job_snapshot(project_id)

    await websocket.send_json(
        {
            "type": "state",
            "project_id": project_id,
            "script": current_script,
            "updated_at": utc_now(),
        }
    )
    await websocket.send_json(
        {
            "type": "workspace_state",
            "project_id": project_id,
            "workspace": workspace_state,
            "updated_at": utc_now(),
        }
    )
    if latest_job is not None:
        await websocket.send_json(
            {
                "type": "job_update",
                "project_id": project_id,
                "job": latest_job,
                "updated_at": utc_now(),
            }
        )
    await _broadcast_presence(project_id)

    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type")

            if message_type in {"edit", "file_edit"}:
                timestamp = utc_now()
                raw_content = payload.get("content", "")
                content = str(raw_content)[:200000]
                target_path = ""

                async with state_lock:
                    project = projects.get(project_id)
                    if project is None:
                        break

                    workspace = _ensure_workspace_unlocked(project)
                    files = workspace.get("files", {})
                    if not isinstance(files, dict):
                        files = {}
                        workspace["files"] = files

                    if message_type == "edit":
                        active_file = str(workspace.get("active_file", ""))
                        if not active_file or active_file not in files:
                            active_file = sorted(files.keys())[0] if files else DEFAULT_WORKSPACE_FILE
                        target_path = active_file
                    else:
                        try:
                            target_path = _normalize_workspace_path(str(payload.get("path", "")))
                        except ValueError as exc:
                            await websocket.send_json({"type": "error", "message": str(exc)})
                            continue

                    if target_path not in files:
                        files[target_path] = {
                            "content": "",
                            "updated_at": timestamp,
                        }

                    files[target_path] = {
                        "content": content,
                        "updated_at": timestamp,
                    }
                    workspace["active_file"] = target_path
                    workspace["updated_at"] = timestamp
                    project["workspace"] = workspace
                    _sync_legacy_script_from_workspace_unlocked(project)
                    project["updated_at"] = timestamp
                    _persist_state_unlocked()

                await _broadcast_workspace_event(
                    project_id,
                    {
                        "type": "file_edit",
                        "editor": client_id,
                        "path": target_path,
                        "content": content,
                        "active_file": target_path,
                    },
                    exclude_client=client_id,
                )
            elif message_type == "file_create":
                timestamp = utc_now()
                try:
                    file_path = _normalize_workspace_path(str(payload.get("path", "")))
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                content = str(payload.get("content", ""))[:200000]
                async with state_lock:
                    project = projects.get(project_id)
                    if project is None:
                        break

                    workspace = _ensure_workspace_unlocked(project)
                    files = workspace.get("files", {})
                    if not isinstance(files, dict):
                        files = {}
                        workspace["files"] = files

                    if file_path in files:
                        await websocket.send_json({"type": "error", "message": "File already exists"})
                        continue

                    files[file_path] = {
                        "content": content,
                        "updated_at": timestamp,
                    }
                    workspace["active_file"] = file_path
                    workspace["updated_at"] = timestamp
                    project["workspace"] = workspace
                    _sync_legacy_script_from_workspace_unlocked(project)
                    project["updated_at"] = timestamp
                    _persist_state_unlocked()

                await _broadcast_workspace_event(
                    project_id,
                    {
                        "type": "file_create",
                        "editor": client_id,
                        "path": file_path,
                        "content": content,
                        "active_file": file_path,
                    },
                )
            elif message_type == "file_delete":
                timestamp = utc_now()
                try:
                    file_path = _normalize_workspace_path(str(payload.get("path", "")))
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                active_file_after = ""
                async with state_lock:
                    project = projects.get(project_id)
                    if project is None:
                        break

                    workspace = _ensure_workspace_unlocked(project)
                    files = workspace.get("files", {})
                    if not isinstance(files, dict) or file_path not in files:
                        await websocket.send_json({"type": "error", "message": "File not found"})
                        continue

                    files.pop(file_path, None)
                    if not files:
                        files[DEFAULT_WORKSPACE_FILE] = {
                            "content": DEFAULT_CODE,
                            "updated_at": timestamp,
                        }
                        workspace["active_file"] = DEFAULT_WORKSPACE_FILE
                    elif str(workspace.get("active_file", "")) == file_path:
                        workspace["active_file"] = sorted(files.keys())[0]

                    active_file_after = str(workspace.get("active_file", ""))
                    workspace["updated_at"] = timestamp
                    project["workspace"] = workspace
                    _sync_legacy_script_from_workspace_unlocked(project)
                    project["updated_at"] = timestamp
                    _persist_state_unlocked()

                await _broadcast_workspace_event(
                    project_id,
                    {
                        "type": "file_delete",
                        "editor": client_id,
                        "path": file_path,
                        "active_file": active_file_after,
                    },
                )
            elif message_type == "file_rename":
                timestamp = utc_now()
                try:
                    old_path = _normalize_workspace_path(str(payload.get("old_path", "")))
                    new_path = _normalize_workspace_path(str(payload.get("new_path", "")))
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                active_file_after = ""
                async with state_lock:
                    project = projects.get(project_id)
                    if project is None:
                        break

                    workspace = _ensure_workspace_unlocked(project)
                    files = workspace.get("files", {})
                    if not isinstance(files, dict) or old_path not in files:
                        await websocket.send_json({"type": "error", "message": "File not found"})
                        continue
                    if new_path in files and new_path != old_path:
                        await websocket.send_json({"type": "error", "message": "Destination file already exists"})
                        continue

                    entry = files.pop(old_path)
                    if not isinstance(entry, dict):
                        entry = {"content": "", "updated_at": timestamp}
                    entry["updated_at"] = timestamp
                    files[new_path] = entry

                    if str(workspace.get("active_file", "")) == old_path:
                        workspace["active_file"] = new_path
                    active_file_after = str(workspace.get("active_file", ""))
                    workspace["updated_at"] = timestamp
                    project["workspace"] = workspace
                    _sync_legacy_script_from_workspace_unlocked(project)
                    project["updated_at"] = timestamp
                    _persist_state_unlocked()

                await _broadcast_workspace_event(
                    project_id,
                    {
                        "type": "file_rename",
                        "editor": client_id,
                        "old_path": old_path,
                        "new_path": new_path,
                        "active_file": active_file_after,
                    },
                )
            elif message_type == "set_active_file":
                timestamp = utc_now()
                try:
                    file_path = _normalize_workspace_path(str(payload.get("path", "")))
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                async with state_lock:
                    project = projects.get(project_id)
                    if project is None:
                        break

                    workspace = _ensure_workspace_unlocked(project)
                    files = workspace.get("files", {})
                    if not isinstance(files, dict) or file_path not in files:
                        await websocket.send_json({"type": "error", "message": "File not found"})
                        continue

                    workspace["active_file"] = file_path
                    workspace["updated_at"] = timestamp
                    project["workspace"] = workspace
                    _sync_legacy_script_from_workspace_unlocked(project)
                    project["updated_at"] = timestamp
                    _persist_state_unlocked()

                await _broadcast_workspace_event(
                    project_id,
                    {
                        "type": "active_file",
                        "editor": client_id,
                        "path": file_path,
                    },
                )
            elif message_type == "request_state":
                async with state_lock:
                    project = projects.get(project_id, {})
                    script = project.get("script", "")
                    workspace = _workspace_snapshot(project) if isinstance(project, dict) else _default_workspace("")
                await websocket.send_json({"type": "state", "project_id": project_id, "script": script, "updated_at": utc_now()})
                await websocket.send_json(
                    {
                        "type": "workspace_state",
                        "project_id": project_id,
                        "workspace": workspace,
                        "updated_at": utc_now(),
                    }
                )
            elif message_type == "request_workspace":
                async with state_lock:
                    project = projects.get(project_id)
                    if project is None:
                        break
                    workspace = _workspace_snapshot(project)
                await websocket.send_json(
                    {
                        "type": "workspace_state",
                        "project_id": project_id,
                        "workspace": workspace,
                        "updated_at": utc_now(),
                    }
                )
            elif message_type == "ping":
                await websocket.send_json({"type": "pong", "updated_at": utc_now()})
    except WebSocketDisconnect:
        pass
    finally:
        async with state_lock:
            room = rooms.get(project_id, {})
            if room.get(client_id) is websocket:
                room.pop(client_id, None)
            if not room:
                rooms.pop(project_id, None)
            if project_id in projects:
                projects[project_id]["active_clients"] = sorted(room.keys())
                projects[project_id]["updated_at"] = utc_now()
        await _broadcast_presence(project_id)
