"""plotpack: figures that carry their own data, code and provenance.

    import plotpack
    plotpack.save(fig, "response-curves", data=df)                       # one table
    plotpack.save(fig, "fig2", data={"curves": df, "ci": ci}, params={"seed": 0})

writes figures/response-curves.{pdf,png,svg} and response-curves.plotpack. Each
image file holds an identical recipe (data + generating source + git commit +
package versions + hashes), recoverable from that one file:

    plotpack inspect figures/response-curves.png
    plotpack unpack  figures/response-curves.pdf recovered/
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .embed import NoRecipe, embed_bytes, embed_file, extract_bytes, extract_file
from .recipe import (DEFAULT_MAX_DATA_BYTES, build_recipe, git_state, normalise_data, open_recipe,
                     read_manifest, verify as _verify_bytes)

__all__ = ["save", "attach", "inspect", "unpack", "verify", "read_table", "load_recipe", "NoRecipe"]
__version__ = "0.2.0"


def _source(path: Path) -> tuple[str, bytes, str]:
    content = path.read_bytes()
    if path.suffix == ".ipynb":
        import json

        notebook = json.loads(content)
        notebook["metadata"] = {}
        for cell in notebook.get("cells", []):
            cell["metadata"] = {}
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        content = (json.dumps(notebook, indent=2) + "\n").encode()
    return path.name, content, "notebook" if path.suffix == ".ipynb" else "file"


def _caller_source(stack_depth: int) -> tuple[tuple[str, bytes, str] | None, Path | None]:
    """The script or notebook that called plotpack; none from an interactive session."""
    frame = sys._getframe()
    try:
        for _ in range(stack_depth):
            frame = frame.f_back if frame else None
        fname = frame.f_code.co_filename if frame else ""
    finally:
        del frame
    # Jupyter in VS Code / JupyterLab exposes the notebook path
    main = sys.modules.get("__main__")
    for attr in ("__vsc_ipynb_file__", "__session__"):
        nb = getattr(main, attr, None)
        if isinstance(nb, str) and nb.endswith(".ipynb") and Path(nb).is_file():
            p = Path(nb)
            return _source(p), p
    p = Path(fname)
    if p.is_file() and p.suffix in (".py", ".ipynb"):
        return _source(p), p
    # Interactive history can contain unrelated work or credentials. Capture only
    # an identified source file; callers can supply an explicit exported script.
    return None, Path.cwd()


def save(fig, name: str, data: Any = None, *, params: Mapping[str, Any] | None = None,
         out: str | Path = "figures", formats: Iterable[str] = ("pdf", "png", "svg"),
         source: str | Path | None | bool = None, stamp: bool = False, include_git: bool | None = None,
         dpi: int = 300, notes: str | None = None,
         max_data_bytes: int = DEFAULT_MAX_DATA_BYTES, savefig_kwargs: Mapping[str, Any] | None = None,
         ) -> dict[str, Path]:
    """Save `fig` (anything with Matplotlib's ``savefig``) and embed its recipe.

    data     a DataFrame, numpy array, JSON-able value, file path, a list of paths, or a
             dict of any of those (keys become file names)
    params   anything worth recording (seeds, filters, model names...)
    source   the generating script; defaults to the caller's file or notebook.
             False omits source; notebook outputs and metadata are always omitted.
    stamp    also record when, and from which git commit, the figure was made (commit and dirty
             state; no branch or remote URL). Off by default: then the same data, script and
             parameters always give the same bytes, so a figure kept in git changes only when
             what it shows changes. (`include_git=True` is the older name for this.)
    Returns the written paths, keyed by format plus "plotpack". A file whose bytes would not
    change is left alone.
    """
    stamp = stamp or bool(include_git)
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("name must be a filename without directory components")
    formats = [f.lower().lstrip(".") for f in formats]
    if not formats or any(f not in ("png", "svg", "pdf") for f in formats):
        raise ValueError("formats must contain png, svg or pdf")
    if source is False:
        src, origin = None, Path.cwd()
    elif source is not None:
        sp = Path(source)
        src, origin = _source(sp), sp
    else:
        src, origin = _caller_source(2)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    recipe = build_recipe(
        name=name, data=normalise_data(data, max_data_bytes), source=src, params=params,
        git=git_state(origin) if stamp and origin else None, outputs=formats, notes=notes, stamp=stamp,
    )
    kwargs = {"bbox_inches": "tight", **(savefig_kwargs or {})}
    paths: dict[str, Path] = {}
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        _write_if_changed(p, embed_bytes(_render(fig, fmt, dpi, kwargs), recipe))
        paths[fmt] = p
    paths["plotpack"] = outdir / f"{name}.plotpack"
    _write_if_changed(paths["plotpack"], recipe)
    return paths


def _render(fig, fmt: str, dpi: int, kwargs: dict) -> bytes:
    """The figure's bytes in one format. For Matplotlib, without the creation date it would
    otherwise write and with fixed SVG ids, so the same figure always gives the same bytes."""
    buf = io.BytesIO()
    extra: dict[str, Any] = {"dpi": dpi} if fmt == "png" else {}
    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError:
        Figure = None
    if Figure is not None and isinstance(fig, Figure):
        if "metadata" not in kwargs:
            extra["metadata"] = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}.get(fmt, {})
        with matplotlib.rc_context({"svg.hashsalt": "plotpack"}), _stable_subset_names():
            fig.savefig(buf, format=fmt, **extra, **kwargs)
    else:
        fig.savefig(buf, format=fmt, **extra, **kwargs)
    return buf.getvalue()


@contextlib.contextmanager
def _stable_subset_names():
    """Matplotlib names a PDF's font subsets (the "ABCDEF+" before a font's name) from Python's
    hash() of the characters used; for text set with LaTeX it hashes a dict view, whose hash is
    its memory address, so every run gives a new name. While plotpack saves, the name comes from
    the characters themselves."""
    try:
        from matplotlib.backends.backend_pdf import PdfFile
    except ImportError:
        yield
        return
    original = PdfFile.__dict__.get("_get_subset_prefix")
    if original is None:
        yield
        return

    def prefix(charset):
        digest = hashlib.sha256(repr(sorted(charset)).encode()).digest()
        return "".join(chr(ord("A") + b % 26) for b in digest[:6]) + "+"

    PdfFile._get_subset_prefix = staticmethod(prefix)
    try:
        yield
    finally:
        PdfFile._get_subset_prefix = original


def _write_if_changed(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return  # the same bytes: keep the file (and its modification time) as it is
    pending = path.with_name(f".{path.name}.plotpack-tmp")
    pending.write_bytes(content)
    pending.replace(path)


def attach(image: str | Path, data: Any = None, *, name: str | None = None,
           params: Mapping[str, Any] | None = None, source: str | Path | None | bool = None,
           include_git: bool = True, notes: str | None = None) -> Path:
    """Embed a recipe into an existing PNG/SVG/PDF (e.g. one made by another tool)."""
    image = Path(image)
    if source is False:
        src, origin = None, Path.cwd()
    elif source is not None:
        sp = Path(source)
        src, origin = _source(sp), sp
    else:
        src, origin = _caller_source(2)
    recipe = build_recipe(name=name or image.stem, data=normalise_data(data), source=src, params=params,
                          git=git_state(origin) if include_git and origin else None, outputs=[image.suffix.lstrip(".")],
                          notes=notes, stamp=include_git)
    embed_file(image, recipe)
    return image


def load_recipe(path_or_bytes: str | Path | bytes) -> bytes:
    """The recipe ZIP bytes from a figure file, a .plotpack file, or raw bytes."""
    if isinstance(path_or_bytes, bytes):
        return extract_bytes(path_or_bytes)
    return extract_file(path_or_bytes)


def inspect(figure: str | Path | bytes) -> dict[str, Any]:
    """The manifest: data, source, git state, environment, parameters."""
    return read_manifest(load_recipe(figure))


def unpack(figure: str | Path | bytes, dest: str | Path) -> Path:
    """Extract the recipe (README, manifest, source, data) into `dest`."""
    dest = Path(dest)
    with open_recipe(load_recipe(figure)) as z:
        root = dest.resolve()
        for member in z.infolist():
            path = dest / member.filename
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"unsafe extraction path: {member.filename!r}")
            if not member.is_dir() and path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        dest.mkdir(parents=True, exist_ok=True)
        z.extractall(dest)
    return dest


def verify(figure: str | Path | bytes) -> list[str]:
    """Hash-check every embedded file; returns a list of problems (empty = intact)."""
    return _verify_bytes(load_recipe(figure))


def read_table(figure: str | Path | bytes, name: str | None = None):
    """Load an embedded CSV table as a pandas DataFrame (the first one if `name` is None)."""
    import pandas as pd

    recipe = load_recipe(figure)
    manifest = read_manifest(recipe)
    tables = [d for d in manifest["data"] if d.get("format") == "csv" and d.get("bytes")]
    if name is not None:
        tables = [d for d in tables if d["name"] == name]
    if not tables:
        raise KeyError(f"no embedded table{' named ' + repr(name) if name else ''}")
    with open_recipe(recipe) as z:
        return pd.read_csv(io.BytesIO(z.read(tables[0]["path"])))
