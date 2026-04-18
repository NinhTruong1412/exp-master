#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1Ebc_dHoB4G9RiltOJJCII6A_LGEcugAU?usp=drive_link"
TARGET_DIRS = ("data", "processed_final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download large project data from Google Drive into the local repo."
    )
    parser.add_argument(
        "--drive-url",
        default=DEFAULT_DRIVE_URL,
        help="Google Drive folder URL that contains the project data.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root where data/ and processed_final/ should be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing local data/ and processed_final/ directories.",
    )
    return parser.parse_args()


def ensure_gdown_installed() -> None:
    if importlib.util.find_spec("gdown") is not None:
        return

    raise SystemExit(
        "Missing dependency: gdown\n"
        "Install it with:\n"
        f"  {sys.executable} -m pip install gdown"
    )


def run_gdown(drive_url: str, download_root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        drive_url,
        "--remaining-ok",
    ]
    try:
        subprocess.run(cmd, cwd=download_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Google Drive download failed with exit code {exc.returncode}.") from exc


def find_unique_dir(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_dir() and path.name == name]
    if not matches:
        raise SystemExit(
            f"Downloaded Drive folder does not contain a '{name}/' directory.\n"
            "Make sure the shared Drive folder includes both 'data/' and 'processed_final/' at some level."
        )
    if len(matches) > 1:
        formatted = "\n".join(f"  - {path}" for path in matches)
        raise SystemExit(
            f"Found multiple '{name}/' directories in the downloaded Drive content:\n{formatted}\n"
            "Clean up the Drive folder layout so there is only one."
        )
    return matches[0]


def replace_tree(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        has_files = any(dst.iterdir())
        if has_files and not force:
            raise SystemExit(
                f"Destination already exists and is not empty: {dst}\n"
                "Re-run with --force to replace it."
            )
        shutil.rmtree(dst)

    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    ensure_gdown_installed()

    with tempfile.TemporaryDirectory(prefix="exp-drive-download-") as tmp:
        download_root = Path(tmp)
        run_gdown(args.drive_url, download_root)

        located = {name: find_unique_dir(download_root, name) for name in TARGET_DIRS}
        for name, src in located.items():
            replace_tree(src, repo_root / name, force=args.force)

    print("Data setup complete.")
    for name in TARGET_DIRS:
        print(f"  - {repo_root / name}")


if __name__ == "__main__":
    main()
