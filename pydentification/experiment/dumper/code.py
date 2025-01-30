import os
import shutil
import uuid
from pathlib import Path

PYTHON_EXTENSIONS = frozenset({".py", ".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".ini"})
FORBIDDEN_PREFIX = frozenset({"venv", ".ipynb_checkpoints", "__pycache__", ".git", ".pytest_cache"})


def load_gitignore() -> set[str]:
    """Load .gitignore from default name and root directory as set"""

    def not_comment(line: str) -> bool:
        return not (line.startswith("#") or line.isspace() or not line)

    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return set()

    with gitignore.open("r") as f:
        return set(filter(not_comment, f.read().splitlines()))


def save_code_snapshot(name: str, source_dir: str | Path):
    """Save only text-based files in a ZIP archive, excluding binary data files."""
    if isinstance(source_dir, str):
        source_dir = Path(source_dir)

    source_dir = Path(source_dir).resolve()  # ensure absolute path
    snapshot_filename = f"source_code_{name}"
    temp_dir = Path(f"temp_code_snapshot_{uuid.uuid4()}")  # append random UUID to avoid conflicts

    forbidden = FORBIDDEN_PREFIX | load_gitignore()  # union of forbidden prefixes and gitignore

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        if (
            str(temp_dir.absolute()) == root  # prevent copying the temp directory
            or temp_dir in root_path.parents
            or any(part.startswith(prefix) for prefix in forbidden for part in root_path.parts)
        ):
            dirs.clear()  # prevent descending into this directory
            continue  # skip to the next directory

        for file in files:
            file_path = root_path / file
            if file_path.suffix in PYTHON_EXTENSIONS:
                relative_path = file_path.relative_to(source_dir)
                dest_path = temp_dir / relative_path

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)

    shutil.make_archive(snapshot_filename, format="zip", root_dir=temp_dir)  # archive the directory
    shutil.rmtree(temp_dir)
