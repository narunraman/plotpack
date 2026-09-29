"""plotpack command line.

  plotpack inspect FIGURE [--json]     what's inside: data, source, git state, environment
  plotpack unpack FIGURE [DEST]        extract README, manifest, source and data
  plotpack verify FIGURE...            check every embedded file against its hash
  plotpack attach IMAGE --data F...    add a recipe to an existing PNG/SVG/PDF
FIGURE is a .png, .svg, .pdf or .plotpack file. Exit status: 0 fine, 1 verify found
changed files, 2 a file could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
import zlib
from pathlib import Path


def _human(n) -> str:
    if not isinstance(n, (int, float)):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _print_manifest(m: dict) -> None:
    print(f"{m.get('name')}  ({m.get('schema')}, created {m.get('created_utc')})")
    src = m.get("source")
    print(f"  source   {src['path']} ({src.get('kind')})" if src else "  source   (none captured)")
    g = m.get("git")
    if g:
        state = "uncommitted changes" if g.get("dirty") else "clean"
        if g.get("source_modified"):
            state += ", this script modified"
        print(f"  git      {str(g.get('commit') or '?')[:12]} ({state})")
    for d in m.get("data", []):
        shape = (f"{d['rows']} rows × {len(d.get('columns') or [])} cols" if "rows" in d
                 else f"shape {d['shape']}" if "shape" in d else str(d.get("format", "")))
        flag = f"  [{d['note']}]" if d.get("truncated") else ""
        print(f"  data     {d['path']:32s} {shape:22s} {_human(d.get('bytes'))}{flag}")
    if m.get("params"):
        print("  params   " + json.dumps(m["params"], default=str, ensure_ascii=False))
    env = m.get("environment") or {}
    pk = ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
    print(f"  env      python {env.get('python')}" + (f"; {pk}" if pk else ""))


def _error(path: Path, exc: Exception) -> str:
    if isinstance(exc, OSError) and exc.filename:
        return f"{exc.filename}: {exc.strerror or exc}"
    if isinstance(exc, (zipfile.BadZipFile, zlib.error)):
        return f"{path}: damaged recipe ({exc})"
    return f"{path}: {exc}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plotpack", description="Figures that carry their data, code and provenance.",
                                epilog="FIGURE is a .png, .svg, .pdf or .plotpack file. Nothing inside is ever run.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect", help="show what a figure contains")
    s.add_argument("figure", type=Path)
    s.add_argument("--json", action="store_true", help="print the full manifest as JSON")
    s = sub.add_parser("unpack", help="extract the recipe: README, manifest, source and data")
    s.add_argument("figure", type=Path)
    s.add_argument("dest", type=Path, nargs="?", help="a new or empty directory (default: FIGURE-recipe)")
    s = sub.add_parser("verify", help="check embedded files against their recorded hashes")
    s.add_argument("figures", type=Path, nargs="+", metavar="figure")
    s = sub.add_parser("attach", help="embed a recipe in an existing PNG, SVG or PDF")
    s.add_argument("image", type=Path)
    s.add_argument("--data", type=Path, nargs="*", default=[], metavar="FILE", help="data files to include")
    s.add_argument("--source", type=Path, metavar="FILE", help="the script that made the image")
    s.add_argument("--no-source", action="store_true", help="omit generating source")
    s.add_argument("--no-git", action="store_true", help="omit git provenance")
    s.add_argument("--name", help="recipe name (default: the image's file name)")
    args = p.parse_args(argv)
    if args.cmd == "attach" and args.source and args.no_source:
        p.error("--source and --no-source cannot be used together")

    from . import attach, inspect, unpack, verify

    errors = (ValueError, OSError, KeyError, zipfile.BadZipFile, zlib.error)
    if args.cmd == "verify":  # one line per figure; a bad file doesn't stop the rest
        status = 0
        for f in args.figures:
            try:
                problems = verify(f)
            except errors as exc:
                print(_error(f, exc), file=sys.stderr)
                status = 2
                continue
            print(f"{f}: " + ("OK" if not problems else "; ".join(problems)))
            status = max(status, 1 if problems else 0)
        return status
    try:
        if args.cmd == "inspect":
            m = inspect(args.figure)
            print(json.dumps(m, indent=2, default=str)) if args.json else _print_manifest(m)
        elif args.cmd == "unpack":
            print(unpack(args.figure, args.dest or Path(f"{args.figure.stem}-recipe")))
        elif args.cmd == "attach":
            attach(args.image, args.data or None, source=False if args.no_source else args.source,
                   include_git=not args.no_git, name=args.name)
            print(f"embedded recipe in {args.image}")
    except errors as exc:
        print(f"plotpack: {_error(getattr(args, 'figure', None) or args.image, exc)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
