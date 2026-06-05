#!/usr/bin/env python3
"""Validate rollout MP4 artifacts.

Every real SimToolReal rollout in this project must leave a video artifact.
This checker keeps that contract explicit: the file must exist, be readable,
have enough frames/duration, and contain non-blank visual signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    return streams[0] if streams else {}


def _visual_stats_with_ffmpeg(path: Path, probe: dict[str, Any], *, max_frames: int = 32) -> dict[str, Any]:
    width = int(probe.get("width") or 0)
    height = int(probe.get("height") or 0)
    if width <= 0 or height <= 0:
        return {
            "available": False,
            "reason": "ffprobe did not report valid dimensions",
            "sampled_frames": 0,
            "max_stddev": 0.0,
            "mean_rgb": None,
        }
    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {
            "available": False,
            "reason": f"ffmpeg raw-frame sampling failed: {exc}",
            "sampled_frames": 0,
            "max_stddev": 0.0,
            "mean_rgb": None,
        }
    raw = result.stdout
    sampled = len(raw) // frame_bytes
    max_stddev = 0.0
    mean_rgb = None
    for frame_idx in range(sampled):
        frame = raw[frame_idx * frame_bytes : (frame_idx + 1) * frame_bytes]
        if not frame:
            continue
        n = len(frame)
        mean = sum(frame) / n
        variance = sum((value - mean) ** 2 for value in frame) / n
        max_stddev = max(max_stddev, variance ** 0.5)
        channels = [frame[channel::3] for channel in range(3)]
        mean_rgb = [sum(channel) / len(channel) for channel in channels]
    return {
        "available": True,
        "backend": "ffmpeg_rawvideo",
        "sampled_frames": sampled,
        "max_stddev": max_stddev,
        "mean_rgb": mean_rgb,
    }


def _visual_stats(path: Path, probe: dict[str, Any], *, max_frames: int = 32) -> dict[str, Any]:
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError:
        return _visual_stats_with_ffmpeg(path, probe, max_frames=max_frames)

    reader = imageio.get_reader(str(path))
    sampled = 0
    max_stddev = 0.0
    mean_rgb = None
    try:
        for idx, frame in enumerate(reader):
            if idx >= max_frames:
                break
            rgb = np.asarray(frame)[..., :3]
            sampled += 1
            max_stddev = max(max_stddev, float(np.std(rgb)))
            mean_rgb = [float(x) for x in np.mean(rgb, axis=(0, 1))]
    finally:
        reader.close()
    return {
        "available": True,
        "backend": "imageio_numpy",
        "sampled_frames": sampled,
        "max_stddev": max_stddev,
        "mean_rgb": mean_rgb,
    }


def check_video_file(
    path: str | Path,
    *,
    min_size_bytes: int = 1024,
    min_duration_sec: float = 0.5,
    min_frames: int = 2,
    min_stddev: float = 1.0,
) -> dict[str, Any]:
    path = Path(path)
    failures: list[str] = []
    if not path.exists():
        return {
            "passed": False,
            "path": str(path),
            "failures": ["video file does not exist"],
        }
    size_bytes = path.stat().st_size
    if size_bytes < min_size_bytes:
        failures.append(f"video size {size_bytes} below {min_size_bytes} bytes")

    probe: dict[str, Any] = {}
    try:
        probe = _ffprobe(path)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
        failures.append(f"ffprobe failed: {exc}")

    duration = float(probe.get("duration") or 0.0)
    nb_frames_raw = probe.get("nb_frames")
    nb_frames = int(nb_frames_raw) if str(nb_frames_raw).isdigit() else 0
    if duration < min_duration_sec:
        failures.append(f"duration {duration:.3f}s below {min_duration_sec:.3f}s")
    if nb_frames and nb_frames < min_frames:
        failures.append(f"frame count {nb_frames} below {min_frames}")

    stats = _visual_stats(path, probe)
    if stats.get("available"):
        if stats["sampled_frames"] < min_frames:
            failures.append(f"sampled only {stats['sampled_frames']} frames")
        if stats["max_stddev"] < min_stddev:
            failures.append(
                f"max frame stddev {stats['max_stddev']:.3f} below {min_stddev:.3f}"
            )
    else:
        failures.append(str(stats.get("reason", "visual stats unavailable")))

    return {
        "passed": not failures,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": size_bytes,
        "ffprobe": probe,
        "visual_stats": stats,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    report = check_video_file(args.video)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
