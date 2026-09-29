"""Hide a recipe (a ZIP of bytes) inside PNG, SVG and PDF files, and get it back.

- PNG: a compressed ``iTXt`` chunk (keyword ``plotpack``) just before ``IEND``.
  Image viewers ignore unknown text chunks, so the picture is unchanged.
- SVG: a base64 ``<plotpack:recipe>`` element inside ``<metadata>``.
- PDF: an embedded file attachment (visible in PDF readers' attachment pane).

Figures written by the older *paperpack* kit (keyword ``Paperpack``, namespace
``urn:paperpack:figure:1``, attachment ``paperpack-recipe.plotpack``) are read too.
"""

from __future__ import annotations

import base64
import re
import struct
import stat
import tempfile
import zlib
from pathlib import Path

from .recipe import MAX_RECIPE_BYTES

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_KEYWORDS = (b"plotpack", b"Paperpack")          # write the first, read both
SVG_NAMESPACES = ("urn:plotpack:1", "urn:paperpack:figure:1")
PDF_ATTACHMENTS = ("plotpack.zip", "paperpack-recipe.plotpack")


class NoRecipe(ValueError):
    """The file has no plotpack recipe embedded."""


# -- PNG ------------------------------------------------------------------------
def _png_chunks(content: bytes):
    if not content.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    pos = len(PNG_SIGNATURE)
    while pos < len(content):
        if len(content) - pos < 12:
            raise ValueError("truncated PNG chunk")
        (length,) = struct.unpack(">I", content[pos : pos + 4])
        end = pos + 12 + length
        if end > len(content):
            raise ValueError("truncated PNG chunk")
        yield content[pos + 4 : pos + 8], content[pos + 8 : pos + 8 + length], content[pos:end]
        pos = end


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def _is_ours(kind: bytes, payload: bytes) -> bytes | None:
    """The keyword if this is a plotpack/paperpack iTXt chunk."""
    if kind != b"iTXt":
        return None
    for keyword in PNG_KEYWORDS:
        if payload.startswith(keyword + b"\0"):
            return keyword
    return None


def embed_png(content: bytes, recipe: bytes) -> bytes:
    out = bytearray(PNG_SIGNATURE)
    ended = False
    for kind, payload, raw in _png_chunks(content):
        if _is_ours(kind, payload):
            continue  # replace any previous recipe
        if kind == b"IEND":
            ended = True
            # iTXt: keyword \0, compression flag 1, method 0, empty language tag \0, empty translated keyword \0
            out += _chunk(b"iTXt", PNG_KEYWORDS[0] + b"\0\x01\x00\0\0" + zlib.compress(recipe, 9))
        out += raw
    if not ended:
        raise ValueError("PNG has no IEND chunk")
    return bytes(out)


def extract_png(content: bytes) -> bytes:
    for kind, payload, _ in _png_chunks(content):
        if keyword := _is_ours(kind, payload):
            decompressor = zlib.decompressobj()
            result = decompressor.decompress(payload[len(keyword) + 5 :], MAX_RECIPE_BYTES + 1)
            if len(result) > MAX_RECIPE_BYTES or decompressor.unconsumed_tail:
                raise ValueError("recipe exceeds the 100 MB limit")
            if not decompressor.eof:
                raise ValueError("truncated PNG recipe")
            return result
    raise NoRecipe("no plotpack recipe in this PNG")


# -- SVG ------------------------------------------------------------------------
_SVG_OPEN = re.compile(rb"<svg\b[^>]*>")
_SVG_RECIPE = re.compile(rb"<(?:plotpack|paperpack):recipe\b[^>]*>(.*?)</(?:plotpack|paperpack):recipe>", re.S)
_SVG_OLD = re.compile(rb'<metadata id="(?:plotpack|paperpack)">.*?</metadata>', re.S)


def embed_svg(content: bytes, recipe: bytes) -> bytes:
    content = _SVG_OLD.sub(b"", content)
    m = _SVG_OPEN.search(content)
    if not m:
        raise ValueError("not an SVG file (no <svg> element)")
    block = (
        f'<metadata id="plotpack"><plotpack:recipe xmlns:plotpack="{SVG_NAMESPACES[0]}" '
        f'encoding="base64+zip">'.encode() + base64.b64encode(recipe) + b"</plotpack:recipe></metadata>"
    )
    return content[: m.end()] + block + content[m.end() :]


def extract_svg(content: bytes) -> bytes:
    m = _SVG_RECIPE.search(content)
    if not m:
        raise NoRecipe("no plotpack recipe in this SVG")
    encoded = re.sub(rb"\s+", b"", m.group(1))
    if len(encoded) > ((MAX_RECIPE_BYTES + 2) // 3) * 4:
        raise ValueError("recipe exceeds the 100 MB limit")
    return base64.b64decode(encoded, validate=True)


# -- PDF ------------------------------------------------------------------------
def embed_pdf(content: bytes, recipe: bytes) -> bytes:
    import io

    from pypdf import PdfReader, PdfWriter
    from pypdf.errors import PyPdfError

    try:
        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(content)))
    except PyPdfError as exc:
        raise ValueError("invalid PDF file") from exc
    writer.add_attachment(PDF_ATTACHMENTS[0], recipe)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def extract_pdf(content: bytes) -> bytes:
    import io

    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        attachments = PdfReader(io.BytesIO(content)).attachments
    except PyPdfError as exc:
        raise ValueError("invalid PDF file") from exc
    for name in PDF_ATTACHMENTS:
        if attachments.get(name):
            return attachments[name][0]
    raise NoRecipe("no plotpack recipe in this PDF")


# -- dispatch -------------------------------------------------------------------
def sniff(content: bytes) -> str:
    """File type from content, whatever the file is called."""
    if content.startswith(PNG_SIGNATURE):
        return ".png"
    if content.startswith(b"%PDF"):
        return ".pdf"
    if content.startswith(b"PK\x03\x04"):
        return ".plotpack"
    if b"<svg" in content[:4096]:
        return ".svg"
    raise ValueError("unsupported file: expected a PNG, SVG, PDF or .plotpack")


_EMBED = {".png": embed_png, ".svg": embed_svg, ".pdf": embed_pdf}
_EXTRACT = {".png": extract_png, ".svg": extract_svg, ".pdf": extract_pdf}


def embed_bytes(content: bytes, recipe: bytes) -> bytes:
    return _EMBED[sniff(content)](content, recipe)


def extract_bytes(content: bytes) -> bytes:
    kind = sniff(content)
    return content if kind == ".plotpack" else _EXTRACT[kind](content)


def embed_file(path: str | Path, recipe: bytes) -> None:
    p = Path(path)
    if p.is_symlink():
        p = p.resolve()
    content = embed_bytes(p.read_bytes(), recipe)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent, prefix=".plotpack-", suffix=".tmp", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(content)
        pending.chmod(stat.S_IMODE(p.stat().st_mode))
        pending.replace(p)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def extract_file(path: str | Path) -> bytes:
    return extract_bytes(Path(path).read_bytes())
