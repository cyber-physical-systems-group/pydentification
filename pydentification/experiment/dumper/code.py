import os
import shutil
import uuid
from pathlib import Path


def _load_gitignore() -> set[str]:
    """Load .gitignore from default name and root directory as set"""

    def not_comment(line: str) -> bool:
        return not (line.startswith("#") or line.isspace() or not line)

    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return set()

    with gitignore.open("r") as f:
        return set(filter(not_comment, f.read().splitlines()))


def _skip_subdir(current: Path, archive_path: Path, forbidden_paths: set[str]) -> bool:
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


def save_code_snapshot(
    name: str,
    source_dir: str | Path,
    target_dir: str | Path,
    filter_prefix: set[str] = frozenset({"venv", ".ipynb_checkpoints", "__pycache__", ".git", ".pytest_cache"}),
    accept_suffix: set[str] = frozenset({".py", ".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".ini"}),
    use_gitignore: bool = True,
):
    """
    Save only text-based files in a ZIP archive, excluding binary data files.

    :param name: name of the archive file
    :param source_dir: path to the directory with source code
    :param target_dir: path to the directory where the archive will be saved
    :param filter_prefix: set of prefixes to exclude from the archive
    :param accept_suffix: set of suffixes to include in the archive
    :param use_gitignore: whether to use .gitignore file in the source directory for filter_prefix
    """
    if isinstance(source_dir, str):
        source_dir = Path(source_dir)

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    source_dir = Path(source_dir).resolve()  # ensure absolute path
    snapshot_path = target_dir / name
    temp_dir = target_dir / str(uuid.uuid4())  # create temp dir with unique name for copying files

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    if use_gitignore:
        filter_prefix |= _load_gitignore()  # union with .gitignore, if present

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)

        if _skip_subdir(root_path, temp_dir, filter_prefix):
            dirs.clear()  # prevent descending into this directory
            continue  # skip to the next directory

        for file in files:
            source_path = root_path / file
            if source_path.suffix in accept_suffix:
                relative_path = source_path.relative_to(os.getcwd())
                dest_path = temp_dir / relative_path

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(relative_path, dest_path)

    shutil.make_archive(str(snapshot_path), format="zip", root_dir=temp_dir)  # archive the directory
    shutil.rmtree(temp_dir)
