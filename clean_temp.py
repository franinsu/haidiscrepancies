#!/usr/bin/env python3
"""Preview temporary files; --apply deletes them, --results includes all generated artifacts."""
import argparse
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--apply", action="store_true", help="Delete instead of previewing")
parser.add_argument("--results", action="store_true",
                    help="Include intermediate/, results/, and legacy data/processed/; delete only with --apply")
args = parser.parse_args()

root = Path(__file__).resolve().parent
caches = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".mplconfig"}
skip = {"data", "intermediate", "results", "checks", ".git", ".venv", "venv", "env", "node_modules"}
auxiliary = (".pyc", ".pyo", ".aux", ".fls", ".fdb_latexmk", ".toc", ".synctex.gz", ".nav", ".snm", ".vrb")
targets = []

if args.results:
    for path in (root / "intermediate", root / "results", root / "data/processed"):
        if path.is_dir() and not path.is_symlink() and not path.parent.is_symlink():
            targets.append(path)

for folder, directories, files in os.walk(root):
    folder = Path(folder)
    directories[:] = [name for name in directories
                      if name not in skip and not (folder / name).is_symlink()]
    for name in directories[:]:
        if name in caches:
            targets.append(folder / name)
            directories.remove(name)
    for name in files:
        path = folder / name
        if path.is_symlink():
            continue
        tex_log = path.suffix in {".log", ".out"} and path.with_suffix(".tex").is_file()
        if name == ".DS_Store" or name.endswith(auxiliary) or tex_log:
            targets.append(path)

print(f"{'Cleanup' if args.apply else 'Preview'} in {root}")
for path in sorted(targets):
    print(f"  {path.relative_to(root)}")
    if args.apply:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
print(f"{len(targets)} files/directories {'removed' if args.apply else 'selected (add --apply to delete)' }.")
