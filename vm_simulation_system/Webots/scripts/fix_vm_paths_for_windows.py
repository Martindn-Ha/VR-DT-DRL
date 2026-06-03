#!/usr/bin/env python3
"""
Rewrite Linux VM absolute paths in Webots .wbt / .proto files for Windows.

Usage (from repo root):
  python vm_simulation_system/Webots/scripts/fix_vm_paths_for_windows.py

Defaults to the Webots project at repo-root updated_world/:
  ~/catkin_ws/src/vm_simulation_system/Webots/
  -> C:\\...\\VR-DT-DRL\\updated_world\\
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXTENSIONS = {".wbt", ".proto", ".wrl", ".wbproj"}

LINUX_VM_WEBOTS = "/home/seth/catkin_ws/src/vm_simulation_system/Webots"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _webots_path(p: Path) -> str:
    """Webots accepts forward slashes on Windows."""
    return p.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webots-root",
        type=Path,
        default=_repo_root() / "updated_world",
        help="Webots project root (contains worlds/ and protos/)",
    )
    parser.add_argument(
        "--linux-base",
        default=LINUX_VM_WEBOTS,
        help="Old absolute Webots prefix saved in world files from the VM",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    webots_root: Path = args.webots_root.resolve()
    win_base = _webots_path(webots_root)

    if not webots_root.is_dir():
        print(f"ERROR: Webots root not found: {webots_root}", file=sys.stderr)
        return 1

    linux_base = args.linux_base.rstrip("/")
    replacements = [
        (linux_base, win_base),
        (linux_base + "/", win_base + "/"),
    ]

    changed_files = 0
    total_replacements = 0

    for path in webots_root.rglob("*"):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        new_text = text
        file_replacements = 0
        for old, new in replacements:
            count = new_text.count(old)
            if count:
                new_text = new_text.replace(old, new)
                file_replacements += count
        if new_text != text:
            changed_files += 1
            total_replacements += file_replacements
            rel = path.relative_to(webots_root)
            print(f"{'[dry-run] ' if args.dry_run else ''}update {rel} ({file_replacements} replacements)")
            if not args.dry_run:
                path.write_text(new_text, encoding="utf-8")

    worlds = webots_root / "worlds" / "Environmentnewww.wbt"
    print()
    print(f"Webots root: {webots_root}")
    print(f"Windows base: {win_base}")
    print(f"Files updated: {changed_files}, replacements: {total_replacements}")
    if not (webots_root / "protos" / "textures").is_dir():
        print()
        print("WARNING: protos/textures/ is missing — copy the full Webots project into updated_world/.")
    if not worlds.is_file():
        print(f"WARNING: Expected world not found: {worlds}")
    else:
        print(f"Open in Webots: {worlds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
