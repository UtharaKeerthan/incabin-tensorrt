"""Download and unpack the MediaPipe models used in this project.

A MediaPipe ``.task`` file is a zip archive. It can contain .tflite models
and, in some versions, further nested .task archives. ``extract_task``
unpacks everything recursively so we can see every model inside.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

# Official MediaPipe model hosting. "latest" points at the newest release,
# so record the file hash (see file_info) in your README for reproducibility.
MODEL_URLS: Dict[str, str] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
}


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless the file already exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return dest
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    return dest


def extract_task(task_path: Path, out_dir: Path) -> List[Path]:
    """Recursively unzip a .task bundle. Returns every extracted file."""
    task_path, out_dir = Path(task_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(task_path) as z:
        for member in z.namelist():
            target = out_dir / member
            z.extract(member, out_dir)
            extracted.append(target)
    # Nested bundles: unpack them into a sub-folder named after the bundle.
    for p in list(extracted):
        if p.is_file() and zipfile.is_zipfile(p) and p.suffix in {".task", ".zip"}:
            extracted += extract_task(p, out_dir / p.stem)
    return extracted


def file_info(path: Path) -> Dict[str, object]:
    """Size and SHA-256 of a file (the hash proves which model version you used)."""
    import hashlib

    path = Path(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": path.name, "size_kB": round(path.stat().st_size / 1024, 1), "sha256": h[:16]}


def find_files(root: Path, suffix: str = ".tflite") -> List[Path]:
    return sorted(Path(root).rglob(f"*{suffix}"))
