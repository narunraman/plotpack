"""Build, read and verify figure recipes.

A recipe is a small ZIP:

    manifest.json    what's inside, hashes, git state, environment, parameters
    README.md        how to use the contents (for whoever unpacks it)
    source/<file>    the script or notebook that drew the figure
    data/<name>.*    the derived table(s) needed to redraw it

The stable boundary is *data plus plain source*: no pickled Matplotlib objects,
which break across library versions. Store the small derived table that the
plot needs, not raw inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

SCHEMA = "plotpack/1"
DEFAULT_MAX_DATA_BYTES = 25 * 1024 * 1024  # per data item; larger tables keep a sample + full hash
SAMPLE_ROWS = 1000
MAX_RECIPE_BYTES = 100 * 1024 * 1024
MAX_RECIPE_FILES = 1000
_PACKAGES = ("plotpack", "matplotlib", "numpy", "pandas", "polars", "seaborn", "scipy", "aquarel")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.") or "data"
    return name[:80]


# -- data ------------------------------------------------------------------------
@dataclass
class DataItem:
    name: str
    filename: str
    content: bytes
    meta: dict[str, Any] = field(default_factory=dict)


def _table_meta(df) -> dict[str, Any]:
    return {"rows": int(df.shape[0]), "columns": [str(c) for c in df.columns]}


def _from_object(name: str, obj: Any, max_bytes: int) -> DataItem:
    """Serialise one data object. DataFrames -> CSV, arrays -> .npy, paths -> copied."""
    mod = type(obj).__module__.split(".")[0]
    try:
        is_file = isinstance(obj, (str, Path)) and Path(obj).is_file()
    except OSError:
        is_file = False
    if isinstance(obj, Path) and not is_file:
        raise FileNotFoundError(f"data file not found: {obj}")
    if is_file:
        p = Path(obj)
        meta = {"format": p.suffix.lstrip(".") or "bin", "from_file": p.name}
        if p.stat().st_size > max_bytes:
            with p.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            return DataItem(name, _safe_name(p.name), b"", {**meta, "bytes_full": p.stat().st_size, "sha256_full": digest,
                                                "truncated": True, "note": "file too large to embed; only its hash is kept"})
        return DataItem(name, _safe_name(p.name), p.read_bytes(), meta)
    if mod in ("pandas", "polars") and hasattr(obj, "columns"):
        df = obj
        def csv(frame):
            return frame.write_csv().encode() if mod == "polars" else frame.to_csv(index=not _default_index(frame)).encode()
        raw = csv(df)
        meta = {"format": "csv", **_table_meta(df)}
        if len(raw) > max_bytes:
            sample = csv(df.head(SAMPLE_ROWS))
            meta.update(truncated=True, sample_rows=min(SAMPLE_ROWS, len(df)), bytes_full=len(raw),
                        sha256_full=_sha256(raw), note=f"table too large to embed; first {SAMPLE_ROWS} rows kept")
            raw = sample
            if len(raw) > max_bytes:
                raw = b""
                meta.update(sample_rows=0, note="sample too large to embed; only the full table's hash is kept")
        return DataItem(name, f"{_safe_name(name)}.csv", raw, meta)
    if mod == "numpy":
        import numpy as np

        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        raw = buf.getvalue()
        meta = {"format": "npy", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if len(raw) > max_bytes:
            meta.update(truncated=True, bytes_full=len(raw), sha256_full=_sha256(raw),
                        note="array too large to embed; only its hash is kept")
            raw = b""
        return DataItem(name, f"{_safe_name(name)}.npy", raw, meta)
    if isinstance(obj, (dict, list, tuple, int, float, str, bool)) or obj is None:
        raw = (json.dumps(obj, indent=2, default=str) + "\n").encode()
        meta = {"format": "json"}
        if len(raw) > max_bytes:
            meta.update(truncated=True, bytes_full=len(raw), sha256_full=_sha256(raw),
                        note="JSON too large to embed; only its hash is kept")
            raw = b""
        return DataItem(name, f"{_safe_name(name)}.json", raw, meta)
    raise TypeError(
        f"plotpack can't store data of type {type(obj).__name__!r} ({name!r}); pass a DataFrame, "
        "numpy array, JSON-serialisable value, or a path to a file"
    )


def _default_index(df) -> bool:
    import pandas as pd

    return isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1 and df.index.name is None


def normalise_data(data: Any, max_bytes: int = DEFAULT_MAX_DATA_BYTES) -> list[DataItem]:
    if max_bytes <= 0:
        raise ValueError("max_data_bytes must be positive")
    if data is None:
        return []
    if isinstance(data, Mapping):
        items = [_from_object(str(k), v, max_bytes) for k, v in data.items()]
    elif isinstance(data, (list, tuple)) and data and all(isinstance(x, (str, Path)) for x in data):
        items = [_from_object(Path(x).stem, x, max_bytes) for x in data]
    else:
        items = [_from_object("data", data, max_bytes)]
    seen: set[str] = set()
    for it in items:  # de-duplicate filenames
        original = it.filename
        n = 0
        while it.filename.casefold() in seen:
            n += 1
            stem, dot, ext = original.rpartition(".")
            it.filename = f"{stem}-{n}.{ext}" if dot else f"{it.filename}-{n}"
        seen.add(it.filename.casefold())
    return items


# -- provenance ------------------------------------------------------------------
def _git(cwd: Path, *args: str) -> str | None:
    import subprocess

    try:
        r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_state(path: Path) -> dict[str, Any] | None:
    cwd = path if path.is_dir() else path.parent
    root = _git(cwd, "rev-parse", "--show-toplevel")
    if not root:
        return None
    status = _git(cwd, "status", "--porcelain") or ""
    rel = None
    if path.is_file():
        try:
            rel = str(path.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            rel = None
    return {
        "commit": _git(cwd, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "source_modified": bool(rel and _git(cwd, "status", "--porcelain", "--", rel)),
        "path_in_repo": rel,
    }


def environment() -> dict[str, Any]:
    import importlib.metadata
    import platform

    env: dict[str, Any] = {"python": platform.python_version()}
    pkgs = {}
    for name in _PACKAGES:
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    env["packages"] = pkgs
    return env


# -- build / read -------------------------------------------------------------------
_README = """# {name}

Recipe for a figure, made with plotpack on {created}.

- `manifest.json`: what's in here, with SHA-256 hashes, git state and package versions
- `source/`: the code that drew the figure{source_note}
- `data/`: the data it plotted

Load the data:
{loaders}
Check nothing was altered: `plotpack verify <the .plotpack or figure file>`.
"""


def _loader(item: dict) -> str:
    fmt, path = item.get("format"), item["path"]
    if item.get("truncated") and not item.get("bytes"):
        return f"- `{item['name']}`: not embedded ({item.get('note')}); sha256 {item.get('sha256_full', '')[:16]}…"
    line = {"csv": f'pd.read_csv("{path}")', "npy": f'np.load("{path}")', "json": f'json.load(open("{path}"))'}.get(
        fmt, f'open("{path}", "rb").read()')
    return f"- `{item['name']}`: `{line}`" + (" (sample only)" if item.get("truncated") else "")


def build_recipe(*, name: str, data: list[DataItem], source: tuple[str, bytes, str] | None,
                 params: Mapping[str, Any] | None = None, git: dict | None = None,
                 outputs: list[str] | None = None, notes: str | None = None) -> bytes:
    """`source` is (filename, content, kind) where kind is "file" or "notebook"."""
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "name": name,
        "created_utc": created,
        "source": None,
        "data": [],
        "params": dict(params or {}),
        "git": git,
        "environment": environment(),
        "outputs": outputs or [],
    }
    if notes:
        manifest["notes"] = notes
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if source:
            fname, content, kind = source
            spath = f"source/{_safe_name(fname)}"
            z.writestr(spath, content)
            manifest["source"] = {"path": spath, "kind": kind, "bytes": len(content), "sha256": _sha256(content)}
        for it in data:
            entry = {"name": it.name, "path": f"data/{it.filename}", "bytes": len(it.content),
                     "sha256": _sha256(it.content) if it.content else None, **it.meta}
            if it.content:
                z.writestr(entry["path"], it.content)
            manifest["data"].append(entry)
        readme = _README.format(
            name=name, created=created,
            source_note=f" (`{manifest['source']['path']}`)" if manifest["source"] else " (none captured)",
            loaders="\n".join(_loader(d) for d in manifest["data"]) or "- (no data attached)",
        )
        z.writestr("README.md", readme)
        z.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    recipe = buf.getvalue()
    with open_recipe(recipe):
        pass
    return recipe


def open_recipe(recipe: bytes) -> zipfile.ZipFile:
    if len(recipe) > MAX_RECIPE_BYTES:
        raise ValueError("recipe exceeds the 100 MB limit")
    z = zipfile.ZipFile(io.BytesIO(recipe))
    try:
        members = z.infolist()
        if len(members) > MAX_RECIPE_FILES or sum(m.file_size for m in members) > MAX_RECIPE_BYTES:
            raise ValueError("recipe exceeds the 100 MB / 1000 member limit")
        seen = {}
        for member in members:  # refuse path tricks before anything is read or extracted
            n = member.filename
            p = PurePosixPath(n)
            normalized = unicodedata.normalize("NFC", p.as_posix()).casefold()
            if (p.is_absolute() or ".." in p.parts or "\\" in n or ":" in n
                    or stat.S_ISLNK(member.external_attr >> 16) or normalized in seen):
                raise ValueError(f"unsafe path in recipe: {n!r}")
            seen[normalized] = member.is_dir()
        for path in seen:
            for parent in PurePosixPath(path).parents:
                if parent.as_posix() in seen and not seen[parent.as_posix()]:
                    raise ValueError(f"conflicting paths in recipe: {path!r}")
    except Exception:
        z.close()
        raise
    return z


def read_manifest(recipe: bytes) -> dict[str, Any]:
    """The recipe's manifest, checked for the structure inspect/verify rely on (nothing in it is run)."""
    with open_recipe(recipe) as z:
        if "manifest.json" not in z.namelist():
            raise ValueError("recipe has no manifest.json")
        manifest = json.loads(z.read("manifest.json"))

    def invalid(what: str):
        raise ValueError(f"invalid {what} in recipe manifest")

    if not isinstance(manifest, dict):
        invalid("structure")
    data = manifest.get("data", [])
    if not isinstance(data, list) or not all(isinstance(d, dict) and isinstance(d.get("path"), str) for d in data):
        invalid("data entries")
    if str(manifest.get("schema")).startswith("paperpack."):
        manifest = _upgrade_paperpack(manifest)
    if manifest.get("schema") not in (SCHEMA, "paperpack.figure/v1"):
        raise ValueError(f"unsupported recipe schema {manifest.get('schema')!r}")
    source = manifest.get("source")
    if source is not None and not (isinstance(source, dict) and isinstance(source.get("path"), str)):
        invalid("source entry")
    for key in ("params", "environment", "git"):
        if not isinstance(manifest.get(key) or {}, dict):
            invalid(key)
    if not isinstance((manifest.get("environment") or {}).get("packages") or {}, dict):
        invalid("package versions")
    return manifest


def _upgrade_paperpack(m: dict) -> dict:
    return {
        "schema": m.get("schema"),
        "name": m.get("name"),
        "created_utc": m.get("created_utc"),
        "source": {"path": m.get("generator"), "kind": "file"} if m.get("generator") else None,
        "data": [{"name": Path(d["path"]).stem, **d} for d in m.get("data", [])],
        "params": m.get("parameters", {}),
        "git": None,
        "environment": {"packages": {k: v for k, v in m.get("environment", {}).items() if k != "python"},
                        "python": m.get("environment", {}).get("python")},
        "outputs": [],
    }


def verify(recipe: bytes) -> list[str]:
    """Problems found; empty means every stored file matches its recorded hash."""
    problems = []
    manifest = read_manifest(recipe)
    with open_recipe(recipe) as z:
        names = set(z.namelist())
        entries = list(manifest.get("data", []))
        if manifest.get("source"):
            entries.append({**manifest["source"], "name": "source"})
        for e in entries:
            if not e.get("sha256"):
                if e.get("bytes"):
                    problems.append(f"{e['path']}: embedded content has no recorded sha256")
                continue  # not embedded (too large): nothing to check
            if e["path"] not in names:
                problems.append(f"{e['path']}: listed in manifest but missing")
            elif _sha256(z.read(e["path"])) != e["sha256"]:
                problems.append(f"{e['path']}: contents do not match the recorded sha256")
    return problems
