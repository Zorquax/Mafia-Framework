from __future__ import annotations

from pathlib import Path


def resolve_repo_path(path: str | Path) -> Path:
    """Resolves a possibly-relative path robustly, regardless of the
    directory the process was launched from.

    Tries, in order:
    1. The path as given, if it's already absolute.
    2. Relative to the current working directory, if it exists there.
    3. Relative to the repository root, if it exists there.

    Falls back to the cwd-relative candidate if none of the above exist,
    so a not-yet-created file (e.g. a fresh database) still resolves
    predictably to "wherever the user is running things from".
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate
