"""Filesystem tools: read, write, list, delete — all bounded to the workspace."""

from pathlib import Path
from harness.tools.registry import tool
from harness.config import WORKSPACE, PLAN_FILENAME


def _resolve_path(path: str) -> Path:
    """
    Resolve `path` against the workspace and confirm it stays inside.

    Raises ValueError if the resolved path escapes WORKSPACE — this is the
    hard constraint that prevents the agent from reading or modifying
    anything outside its sandbox.
    """
    # Step 1: join the user-supplied path against the workspace, then canonicalize.
    # Canonicalization expands `..` segments — without it, `../../../etc/passwd`
    # would slip past the check below.
    target = (WORKSPACE / path).resolve()

    # Step 2: confirm the canonical path is still inside WORKSPACE.
    # If `..` walked the path out of the workspace, this catches it.
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"path escapes workspace: {path}")

    return target


@tool
def read(path: str) -> str:
    """Read the contents of a file from the workspace, by path."""
    # Resolve safely, then read the file's text content in one cell
    return _resolve_path(path).read_text()


@tool
def write(path: str, content: str) -> str:
    """Write content to a file in the workspace.

    Overwrites the file if it exists; creates it (and any parent
    directories) if it doesn't. Path is relative to the workspace root.

    Cannot be used to write AGENTS.md — that file is memory-managed
    via the `remember` tool. Use `remember(category, entry)` instead.

    Cannot be used to write plan.md — that file is plan-managed via
    the `update_plan` tool. Use `update_plan(task, status)` instead.
    """
    # Step 1a: refuse AGENTS.md — memory writes must go through remember().
    if path == "AGENTS.md" or path.endswith("/AGENTS.md"):
        return (
            "[write] AGENTS.md is memory-mamnaged. Use the `remember` tool"
            "with a category and entry instead of write()."
        )

    # Step 1b: refuse plan.md — plan writes must go through update_plan(). 
    if path == PLAN_FILENAME or path.endswith(f"/{PLAN_FILENAME}"):
        return (
            f"[write] {PLAN_FILENAME} is plan-managed. Use the `update_plan` "
            f"tool with task and status arguments instead of write()."
        )
    
    # Step 1: resolve the target path safely.
    target = _resolve_path(path)

    # Step 2: ensure parent directories exist so nested paths like
    # `notes/research/findings.md` work without a prior mkdir call.
    target.parent.mkdir(parents=True, exist_ok=True)

    # Step 3: write the content (overwrites if the file exists).
    target.write_text(content)

    # Step 4: return a structured confirmation the model can verify against.
    return f"wrote {len(content)} bytes to {path}"


@tool
def append(path: str, content: str) -> str:
    """Append content to the end of a file in the workspace, creating it if
    needed. Use this to build large files in several smaller calls: write()
    the first section, then append() each following section."""

    if path == "AGENTS.md" or path.endswith("/AGENTS.md"):
        return (
            "[append] AGENTS.md is memory-managed. Use the `remember` tool "
            "with a category and entry instead."
        )

    if path == PLAN_FILENAME or path.endswith(f"/{PLAN_FILENAME}"):
        return (
            f"[write] {PLAN_FILENAME} is plan-managed. Use the `update_plan` "
            f"tool with task and status arguments instead of write()."
        )

    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("a") as f:
        f.write(content)

    return f"appended {len(content)} bytes to {path} (total {target.stat().st_size} bytes)"


@tool
def list(path: str = ".") -> str:
    """List the files and directories at the given path inside the workspace.
    Defaults to the workspace root. Returns one entry per line."""
    # Step 1: resolve the path safely. Defaults to workspace root.
    target = _resolve_path(path)

    # Step 2: refuse if the target isn't a directory — return an error string.
    if not target.is_dir():
        return f"error: not a directory: {path}"

    # Step 3: sort entries for deterministic output; append `/` to directories
    # so the model can distinguish them from files at a glance.
    entries = sorted(target.iterdir())

    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)


@tool
def mkdir(path: str) -> str:
    """Create a directory in the workspace, including any parent directories.
    No error if the directory already exists."""
    # Step 1: resolve the target path safely.
    target = _resolve_path(path)

    # Step 2: create the directory tree. `parents=True` builds the full chain
    # so the agent can create nested paths in one call. `exist_ok=True` makes
    # re-creating an existing directory a no-op rather than an error.
    target.mkdir(parents=True, exist_ok=True)

    return f"created directory {path}"


@tool
def delete(path: str) -> str:
    """Delete a file from the workspace. Will not delete directories."""
    # Step 1: resolve the path safely.
    target = _resolve_path(path)

    # Step 2: refuse directories
    if target.is_dir():
        return f"error: refusing to delete directory: {path}"

    # Step 3: delete the file and confirm
    target.unlink()
    
    return f"deleted {path}"

