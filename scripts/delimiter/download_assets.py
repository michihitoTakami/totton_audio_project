from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model(manifest: dict, model_id: str, sample_rate: int) -> dict:
    for item in manifest.get("models", []):
        if (
            item.get("id") == model_id
            and int(item.get("sample_rate", 0)) == sample_rate
        ):
            return item
    raise KeyError(f"Model {model_id} (sr={sample_rate}) not found in manifest")


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_artifact(artifact: dict, dest_dir: Path, force: bool) -> Path:
    filename = artifact["filename"]
    url = artifact["url"]
    expected_hash = artifact.get("sha256") or ""
    dest = dest_dir / filename

    if dest.exists() and expected_hash and not force:
        if sha256sum(dest) == expected_hash:
            print(f"[skip] {filename} (hash verified)")
            return dest

    print(f"[download] {filename} <- {url}")
    download(url, dest)

    if expected_hash:
        actual = sha256sum(dest)
        if actual != expected_hash:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_hash}, got {actual}"
            )
    return dest


def write_license(
    license_info: dict | None, model_root: Path, force: bool
) -> Path | None:
    if not license_info:
        return None
    url = license_info.get("source")
    dest_name = license_info.get("dest", "LICENSE.upstream")
    if not url:
        return None

    dest = model_root / dest_name
    if dest.exists() and not force:
        print(f"[skip] {dest_name} (already exists)")
        return dest

    print(f"[download] {dest_name} <- {url}")
    download(url, dest)
    return dest


def iter_artifacts(model: dict) -> Iterable[dict]:
    for item in model.get("artifacts", []):
        yield item


def parse_args() -> argparse.Namespace:
    root = project_root()
    p = argparse.ArgumentParser(
        description="Download De-limiter weights/assets with checksum verification (#1098)"
    )
    p.add_argument(
        "--model", default="jeonchangbin49-de-limiter", help="model id in manifest"
    )
    p.add_argument(
        "--sample-rate", type=int, default=44100, help="sample rate key in manifest"
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=root / "data/delimiter/weights",
        help="destination root for assets",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/delimiter/weights/manifest.json",
        help="manifest json path",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the existing file matches the expected hash",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    try:
        model = find_model(manifest, args.model, args.sample_rate)
    except KeyError as e:  # pragma: no cover - CLI surface
        print(str(e), file=sys.stderr)
        return 1

    dest_root = args.dest.resolve()
    model_root = dest_root / args.model
    sr_root = model_root / str(args.sample_rate)

    license_path = write_license(model.get("license"), model_root, args.force)
    if license_path:
        print(f"[ok] license -> {license_path}")

    for artifact in iter_artifacts(model):
        path = ensure_artifact(artifact, sr_root, args.force)
        print(f"[ok] {artifact['filename']} -> {path}")

    print(
        "Done. Place ONNX exports in the same sample-rate directory after running export_onnx.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
