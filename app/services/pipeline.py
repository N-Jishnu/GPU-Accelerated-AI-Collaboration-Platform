from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

SCENE_COLORS = [
    ("#0b132b", "#1c2541"),
    ("#072f5f", "#1261a0"),
    ("#3a0ca3", "#4361ee"),
    ("#003049", "#1d3557"),
]

_DIFFUSERS_PIPE: Any | None = None
_DIFFUSERS_MODEL_ID = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configured_model_stack() -> dict[str, str]:
    return {
        "script": _provider_from_env("SLINGSHOT_SCRIPT_PROVIDER", "mock"),
        "image": _provider_from_env("SLINGSHOT_IMAGE_PROVIDER", "mock"),
        "tts": _provider_from_env("SLINGSHOT_TTS_PROVIDER", "mock"),
    }


def get_model_capabilities() -> dict[str, Any]:
    stack = configured_model_stack()
    ollama_reachable, ollama_summary = _ollama_healthcheck()
    diffusers_available = importlib.util.find_spec("diffusers") is not None
    pyttsx3_available = importlib.util.find_spec("pyttsx3") is not None

    return {
        "configured": stack,
        "providers": {
            "script": {
                "mock": True,
                "ollama": ollama_reachable,
                "ollama_summary": ollama_summary,
            },
            "image": {
                "mock": True,
                "diffusers": diffusers_available,
                "diffusers_model": os.environ.get("SLINGSHOT_DIFFUSERS_MODEL", ""),
            },
            "tts": {
                "mock": True,
                "pyttsx3": pyttsx3_available,
            },
        },
    }


def refine_script(script: str) -> str:
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not lines:
        lines = _sentence_split(script)

    if not lines:
        lines = [
            "Introduce the team and problem statement for future productivity.",
            "Show two collaborators editing the same script in real time.",
            "Generate visual scenes and narration locally with accelerated inference.",
            "Render the final output package and review project assets.",
        ]

    normalized: list[str] = []
    for index, line in enumerate(lines[:6], start=1):
        cleaned = " ".join(line.split())
        if not cleaned:
            continue
        if len(cleaned) > 180:
            cleaned = f"{cleaned[:177]}..."
        cleaned = cleaned[0].upper() + cleaned[1:]
        normalized.append(f"Scene {index}: {cleaned}")

    return "\n".join(normalized)


def refine_script_with_provider(script: str) -> tuple[str, dict[str, Any]]:
    provider = configured_model_stack()["script"]
    if provider == "ollama":
        refined, meta = _refine_script_with_ollama(script)
        if refined:
            return refined, meta
        fallback = refine_script(script)
        return fallback, {
            "provider": "mock",
            "mode": "fallback",
            "fallback_reason": meta.get("error", "Ollama unavailable"),
        }

    return refine_script(script), {"provider": "mock", "mode": "synthetic"}


def write_refined_script(output_dir: Path, refined_script: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "refined_script.txt"
    path.write_text(refined_script, encoding="utf-8")
    return path


def build_scenes(refined_script: str, max_scenes: int = 4) -> list[dict[str, Any]]:
    entries = [line.strip() for line in refined_script.splitlines() if line.strip()]
    if not entries:
        entries = ["Scene 1: Build a short collaborative AI demo narrative."]

    scenes: list[dict[str, Any]] = []
    for index, entry in enumerate(entries[:max_scenes], start=1):
        _, _, caption = entry.partition(":")
        caption = caption.strip() or entry
        title_words = caption.split()[:6]
        title = " ".join(title_words) or f"Scene {index}"
        if len(caption) > 180:
            caption = f"{caption[:177]}..."

        scenes.append(
            {
                "index": index,
                "title": title,
                "caption": caption,
                "duration_s": 5,
            }
        )

    return scenes


def generate_scene_images(output_dir: Path, scenes: list[dict[str, Any]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    for scene in scenes:
        scene_index = int(scene["index"])
        color_a, color_b = SCENE_COLORS[(scene_index - 1) % len(SCENE_COLORS)]
        file_path = output_dir / f"scene_{scene_index:02d}.svg"
        _write_scene_svg(
            file_path=file_path,
            scene_index=scene_index,
            title=str(scene["title"]),
            caption=str(scene["caption"]),
            color_a=color_a,
            color_b=color_b,
        )
        files.append(file_path)

    return files


def generate_scene_images_with_provider(output_dir: Path, scenes: list[dict[str, Any]]) -> tuple[list[Path], dict[str, Any]]:
    provider = configured_model_stack()["image"]
    if provider == "diffusers":
        images, meta = _generate_images_with_diffusers(output_dir, scenes)
        if images:
            return images, meta
        fallback = generate_scene_images(output_dir, scenes)
        return fallback, {
            "provider": "mock",
            "mode": "fallback",
            "fallback_reason": meta.get("error", "Diffusers unavailable"),
        }

    return generate_scene_images(output_dir, scenes), {"provider": "mock", "mode": "synthetic"}


def generate_narration_audio(output_dir: Path, total_duration_s: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "narration.wav"

    duration = max(total_duration_s, 6)
    sample_rate = 22050
    total_samples = duration * sample_rate

    audio = bytearray()
    for i in range(total_samples):
        t = i / sample_rate
        carrier = math.sin(2 * math.pi * 220.0 * t)
        modulation = 0.5 + 0.5 * math.sin(2 * math.pi * 0.75 * t)
        sample = int(32767 * 0.18 * carrier * modulation)
        audio.extend(sample.to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)

    return path


def generate_narration_audio_with_provider(
    output_dir: Path,
    total_duration_s: int,
    narration_text: str,
) -> tuple[Path, dict[str, Any]]:
    provider = configured_model_stack()["tts"]
    if provider == "pyttsx3":
        audio_path, meta = _generate_narration_with_pyttsx3(output_dir, narration_text, total_duration_s)
        if audio_path is not None:
            return audio_path, meta

        fallback = generate_narration_audio(output_dir, total_duration_s)
        return fallback, {
            "provider": "mock",
            "mode": "fallback",
            "fallback_reason": meta.get("error", "pyttsx3 unavailable"),
        }

    return generate_narration_audio(output_dir, total_duration_s), {"provider": "mock", "mode": "synthetic"}


def render_video(output_dir: Path, audio_file: Path, duration_s: int) -> tuple[Path | None, str, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return None, "mock", "ffmpeg not found. Install ffmpeg to enable MP4 rendering."

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "demo_output.mp4"
    duration = max(duration_s, 6)

    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x10253f:s=1280x720:d={duration}",
        "-i",
        str(audio_file),
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(video_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        summary = "ffmpeg failed to render video."
        if exc.stderr:
            lines = [line for line in exc.stderr.splitlines() if line.strip()]
            if lines:
                summary = f"ffmpeg failed: {lines[-1]}"
        return None, "mock", summary
    except subprocess.TimeoutExpired:
        return None, "mock", "ffmpeg timed out while rendering."

    return video_path, "ffmpeg", "Video rendered via ffmpeg."


def probe_system_capabilities() -> dict[str, Any]:
    ffmpeg_available = shutil.which("ffmpeg") is not None
    report: dict[str, Any] = {
        "ffmpeg_available": ffmpeg_available,
        "gpu_backend": "none",
        "gpu_tool": None,
        "gpu_summary": "No GPU telemetry tool detected.",
    }

    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        report["gpu_backend"] = "rocm"
        report["gpu_tool"] = "rocm-smi"
        summary = _run_telemetry_command([rocm_smi, "--showproductname", "--showuse"])
        if summary:
            report["gpu_summary"] = summary
        return report

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        report["gpu_backend"] = "cuda"
        report["gpu_tool"] = "nvidia-smi"
        summary = _run_telemetry_command(
            [
                nvidia_smi,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        )
        if summary:
            report["gpu_summary"] = summary

    return report


def write_render_report(output_dir: Path, note: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "render_report.txt"
    report.write_text(
        "\n".join(
            [
                "Slingshot Prototype Render Report",
                f"Generated at: {utc_now()}",
                f"Status: {note}",
                "Tip: Install ffmpeg and rerun pipeline for MP4 output.",
            ]
        ),
        encoding="utf-8",
    )
    return report


def write_manifest(
    output_dir: Path,
    project_id: str,
    refined_script: str,
    scenes: list[dict[str, Any]],
    image_files: list[Path],
    audio_file: Path,
    render_mode: str,
    render_note: str,
    video_file: Path | None,
    run_metrics: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    payload = {
        "project_id": project_id,
        "generated_at": utc_now(),
        "render_mode": render_mode,
        "render_note": render_note,
        "refined_script_preview": refined_script.splitlines(),
        "scene_count": len(scenes),
        "scenes": scenes,
        "assets": {
            "images": [path.name for path in image_files],
            "audio": audio_file.name,
            "video": video_file.name if video_file else None,
        },
        "run_metrics": run_metrics or {},
    }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _provider_from_env(variable: str, default: str) -> str:
    raw = os.environ.get(variable, default)
    return str(raw).strip().lower() or default


def _ollama_healthcheck() -> tuple[bool, str]:
    base_url = os.environ.get("SLINGSHOT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        payload = _http_json(
            f"{base_url}/api/tags",
            method="GET",
            body=None,
            timeout_s=2,
        )
        models = payload.get("models", []) if isinstance(payload, dict) else []
        model_count = len(models) if isinstance(models, list) else 0
        return True, f"reachable ({model_count} model(s))"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"unreachable ({exc})"


def _refine_script_with_ollama(script: str) -> tuple[str | None, dict[str, Any]]:
    base_url = os.environ.get("SLINGSHOT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("SLINGSHOT_OLLAMA_MODEL", "llama3.2:1b")

    prompt = "\n".join(
        [
            "Rewrite the script as 4 to 6 concise scene lines.",
            "Output one line per scene.",
            "Do not include markdown or explanations.",
            "Input:",
            script or "No script provided.",
        ]
    )

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        payload = _http_json(
            f"{base_url}/api/generate",
            method="POST",
            body=body,
            timeout_s=40,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, {
            "provider": "ollama",
            "mode": "error",
            "error": str(exc),
            "model": model,
        }

    candidate = ""
    if isinstance(payload, dict):
        candidate = str(payload.get("response", "")).strip()

    normalized = _normalize_scene_lines(candidate)
    if not normalized:
        return None, {
            "provider": "ollama",
            "mode": "error",
            "error": "Empty or invalid response from model",
            "model": model,
        }

    return normalized, {
        "provider": "ollama",
        "mode": "model",
        "model": model,
    }


def _generate_images_with_diffusers(
    output_dir: Path,
    scenes: list[dict[str, Any]],
) -> tuple[list[Path], dict[str, Any]]:
    model_id = os.environ.get("SLINGSHOT_DIFFUSERS_MODEL", "").strip()
    if not model_id:
        return [], {
            "provider": "diffusers",
            "mode": "error",
            "error": "SLINGSHOT_DIFFUSERS_MODEL is not configured",
        }

    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionPipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return [], {
            "provider": "diffusers",
            "mode": "error",
            "error": f"Missing dependency: {exc}",
        }

    global _DIFFUSERS_PIPE, _DIFFUSERS_MODEL_ID
    try:
        if _DIFFUSERS_PIPE is None or _DIFFUSERS_MODEL_ID != model_id:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            if device == "cuda":
                pipe = pipe.to("cuda")
            _DIFFUSERS_PIPE = pipe
            _DIFFUSERS_MODEL_ID = model_id

        steps = max(8, int(os.environ.get("SLINGSHOT_DIFFUSERS_STEPS", "16")))
        guidance = float(os.environ.get("SLINGSHOT_DIFFUSERS_GUIDANCE", "6.5"))
        files: list[Path] = []
        for scene in scenes:
            scene_index = int(scene["index"])
            prompt = f"cinematic storyboard frame, {scene['caption']}, high detail, no text"
            image = _DIFFUSERS_PIPE(  # type: ignore[operator]
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=1024,
                height=576,
            ).images[0]
            file_path = output_dir / f"scene_{scene_index:02d}.png"
            image.save(file_path)
            files.append(file_path)

        return files, {
            "provider": "diffusers",
            "mode": "model",
            "model": model_id,
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        return [], {
            "provider": "diffusers",
            "mode": "error",
            "error": str(exc),
            "model": model_id,
        }


def _generate_narration_with_pyttsx3(
    output_dir: Path,
    narration_text: str,
    total_duration_s: int,
) -> tuple[Path | None, dict[str, Any]]:
    try:
        import pyttsx3  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, {
            "provider": "pyttsx3",
            "mode": "error",
            "error": f"Missing dependency: {exc}",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "narration.wav"

    plain_text = " ".join(line.strip() for line in narration_text.splitlines() if line.strip())
    if not plain_text:
        plain_text = "Slingshot prototype narration generated locally."

    max_chars = max(300, total_duration_s * 55)
    speak_text = plain_text[:max_chars]

    try:
        engine = pyttsx3.init()
        rate = int(os.environ.get("SLINGSHOT_TTS_RATE", "175"))
        engine.setProperty("rate", rate)
        engine.save_to_file(speak_text, str(path))
        engine.runAndWait()
        engine.stop()
    except Exception as exc:  # pragma: no cover - runtime dependent
        return None, {
            "provider": "pyttsx3",
            "mode": "error",
            "error": str(exc),
        }

    if not path.exists() or path.stat().st_size < 1024:
        return None, {
            "provider": "pyttsx3",
            "mode": "error",
            "error": "Generated audio file is empty",
        }

    return path, {
        "provider": "pyttsx3",
        "mode": "model",
        "rate": int(os.environ.get("SLINGSHOT_TTS_RATE", "175")),
    }


def _http_json(url: str, method: str, body: dict[str, Any] | None, timeout_s: int) -> dict[str, Any]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response")
    return parsed


def _normalize_scene_lines(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        normalized = line
        if normalized.lower().startswith("scene"):
            _, _, candidate = normalized.partition(":")
            normalized = candidate.strip() or normalized
        normalized = " ".join(normalized.split())
        if normalized:
            cleaned.append(normalized)

    if not cleaned:
        return ""

    scene_lines = []
    for index, line in enumerate(cleaned[:6], start=1):
        scene_lines.append(f"Scene {index}: {line}")
    return "\n".join(scene_lines)


def _sentence_split(text: str) -> list[str]:
    collapsed = " ".join(text.replace("\n", " ").split())
    if not collapsed:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", collapsed) if segment.strip()]


def _run_telemetry_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=6)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return ""
    preview = lines[:4]
    return " | ".join(preview)


def _write_scene_svg(
    file_path: Path,
    scene_index: int,
    title: str,
    caption: str,
    color_a: str,
    color_b: str,
) -> None:
    wrapped_caption = textwrap.wrap(caption, width=56)[:4]
    caption_lines: list[str] = []
    for i, line in enumerate(wrapped_caption):
        y_pos = 304 + i * 48
        caption_lines.append(
            f'<text x="96" y="{y_pos}" fill="#f8fafc" font-size="34" font-family="Verdana">{escape(line)}</text>'
        )

    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">',
            "  <defs>",
            f'    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{color_a}"/><stop offset="100%" stop-color="{color_b}"/></linearGradient>',
            "  </defs>",
            '  <rect x="0" y="0" width="1280" height="720" fill="url(#bg)"/>',
            '  <circle cx="1130" cy="90" r="220" fill="#ffffff" opacity="0.12"/>',
            '  <circle cx="120" cy="620" r="180" fill="#ffffff" opacity="0.08"/>',
            f'  <text x="96" y="118" fill="#e2e8f0" font-size="34" font-family="Verdana">Scene {scene_index:02d}</text>',
            f'  <text x="96" y="220" fill="#ffffff" font-size="56" font-family="Verdana">{escape(title)}</text>',
            *caption_lines,
            '  <text x="96" y="676" fill="#cbd5e1" font-size="26" font-family="Verdana">Slingshot Local Prototype</text>',
            "</svg>",
        ]
    )

    file_path.write_text(svg, encoding="utf-8")
