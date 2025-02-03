import os
import shutil
import uuid
from pathlib import Path

PYTHON_EXTENSIONS = frozenset({".py", ".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".ini"})
DEFAULT_FORBIDDEN_PREFIX = frozenset({"venv", ".ipynb_checkpoints", "__pycache__", ".git", ".pytest_cache"})


def _load_gitignore() -> set[str]:
    """Load .gitignore from default name and root directory as set"""

    def not_comment(line: str) -> bool:
        return not (line.startswith("#") or line.isspace() or not line)

    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return set()

    with gitignore.open("r") as f:
        return set(filter(not_comment, f.read().splitlines()))


def _skip_subdir(current: Path, archive_path: Path, forbidden_paths: frozenset[str]) -> bool:
    # prevent copying the temp directory, where the archive with source code is build
    if str(archive_path.absolute()) == current:
        return True
    # prevent copying the parent directory of the temp directory
    elif archive_path in current.parents:
        return True
    # prevent copying the forbidden paths from defaults and .gitignore
    elif any(part.startswith(prefix) for prefix in forbidden_paths for part in current.parts):
        return True
    return False


def save_code_snapshot(name: str, source_dir: str | Path):
    """Save only text-based files in a ZIP archive, excluding binary data files."""

    if isinstance(source_dir, str):
        source_dir = Path(source_dir)

    source_dir = Path(source_dir).resolve()  # ensure absolute path
    snapshot_filename = f"source_code_{name}"
    temp_dir = Path(f"temp_code_snapshot_{uuid.uuid4()}")  # append random UUID to avoid conflicts

    gitignore = _load_gitignore()
    forbidden = DEFAULT_FORBIDDEN_PREFIX | gitignore

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        if _skip_subdir(root_path, temp_dir, forbidden):
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
