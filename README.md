# Slingshot Prototype (Execution Build)

This folder contains the first executable prototype for the **Distributed GPU Accelerated Multimodal AI Collaboration Studio** from your PPT.

Current build focuses on a reliable MVP flow:

- Real-time collaborative coding workspace with VS Code-style explorer
- Nested folder support with workspace tree API + folder creation
- Legacy script-to-workspace migration endpoint and UI action
- Separate code execution pipeline panel (stdout/stderr/exit/time)
- One-click pipeline job execution
- Storyboard image generation (SVG scenes)
- Narration audio generation (WAV)
- Optional MP4 rendering when `ffmpeg` is installed
- Job progress, logs, and generated asset browser
- Live job status sync to all connected collaborators
- Real local fine-tune worker with epoch checkpoints and loss metrics
- Stage timing + GPU telemetry snapshot in job output
- Optional real provider hooks (Ollama, Diffusers, pyttsx3) with safe fallback
- Generated assets grouped by run folder with per-run asset counts
- Per-run ZIP export from Generated Assets
- Run deletion and old-run cleanup controls
- QR code share link for quick mobile join
- React frontend (Vite build) preserving existing API/WebSocket contract

## Stack

- Backend: FastAPI + WebSocket
- Frontend: React + Vite
- Media pipeline: Python utilities + optional `ffmpeg`

## Run locally (serve React build from FastAPI)

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python -m uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

If you run from the parent folder (`AMD`), use:

```bash
python -m uvicorn app.main:app --reload --port 8000 --app-dir ./prototype
```

## Optional real model providers

By default, the app uses stable mock providers. You can switch specific stages to local models.

Install optional dependencies:

```bash
pip install -r requirements-ml.txt
```

Environment variables (examples):

```bash
# Script refinement via local Ollama
SLINGSHOT_SCRIPT_PROVIDER=ollama
SLINGSHOT_OLLAMA_URL=http://127.0.0.1:11434
SLINGSHOT_OLLAMA_MODEL=llama3.2:3b

# Image generation via local Diffusers model path
SLINGSHOT_IMAGE_PROVIDER=diffusers
SLINGSHOT_DIFFUSERS_MODEL=C:/models/stable-diffusion-xl-base-1.0
SLINGSHOT_DIFFUSERS_STEPS=16
SLINGSHOT_DIFFUSERS_GUIDANCE=6.5

# Narration via local pyttsx3 voice engine
SLINGSHOT_TTS_PROVIDER=pyttsx3
SLINGSHOT_TTS_RATE=175
```

The endpoint `/api/model-capabilities` reports configured providers and local availability.

## Open on laptop and mobile together

1. Start API with LAN binding:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Open on laptop with either:
   - `http://localhost:8000`
   - `http://<your-laptop-ip>:8000`
3. Use **Copy project link** or **Show QR code** in the app.
4. Open/scan the same `Share URL` on mobile.
5. Ensure both devices are on the same Wi-Fi network.

## Frontend development mode (hot reload)

Use two terminals:

```bash
# Terminal 1 (API)
uvicorn app.main:app --reload --port 8000

# Terminal 2 (React dev server)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` while developing.

## Demo flow

1. Open the app in two browser tabs.
2. Share the same project link (`?project=<id>`) between tabs.
3. Create/edit files and folders in one tab and watch live sync in the other tab.
4. Use **Run active file** in Collaborative Coding to execute Python code and inspect stdout/stderr in **Execution Output**.
5. Click **Run pipeline** or **Run fine-tune**.
6. Track live job logs, metrics, and generated assets on both devices.
7. For fine-tune jobs, review `checkpoint_epoch_*.json` and `finetune_report.json` assets.
8. In **Generated Assets**, each run appears as its own folder with asset count and ZIP export.
9. Use **Delete run** or **Cleanup old runs** to manage storage.

## Phase 1 APIs added

- `GET /api/projects/{project_id}/workspace/tree`
- `GET /api/projects/{project_id}/workspace/files/{path}`
- `POST /api/projects/{project_id}/workspace/folders`
- `PATCH /api/projects/{project_id}/workspace/files/move`
- `POST /api/migrate-workspace`
- `POST /api/execute`

## Notes

- If `ffmpeg` is missing, the pipeline still completes with storyboard assets and a render report.
- All generated files are stored in `runtime/assets/<project_id>/...`.
- Projects and jobs are persisted to `runtime/state.json` so restarts keep history.
