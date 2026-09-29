from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plotpack
from plotpack import embed
from plotpack.cli import main as cli


@pytest.fixture
def fig():
    f, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [1, 3, 2])
    yield f
    plt.close(f)


@pytest.fixture
def df():
    return pd.DataFrame({"x": [0, 1, 2], "y": [1.0, 3.0, 2.0]})


def test_save_roundtrip_every_format(fig, df, tmp_path):
    paths = plotpack.save(fig, "curve", data={"curve": df, "arr": np.arange(6).reshape(2, 3)},
                          params={"seed": 7}, out=tmp_path)
    assert set(paths) == {"pdf", "png", "svg", "plotpack"}
    for fmt in ("pdf", "png", "svg", "plotpack"):
        m = plotpack.inspect(paths[fmt])
        assert m["name"] == "curve" and m["params"] == {"seed": 7}
        assert [d["path"] for d in m["data"]] == ["data/curve.csv", "data/arr.npy"]
        assert m["data"][0]["rows"] == 3 and m["data"][1]["shape"] == [2, 3]
        assert plotpack.verify(paths[fmt]) == []
    # the caller (this test file) is captured as the source
    assert plotpack.inspect(paths["png"])["source"]["path"] == "source/test_plotpack.py"
    pd.testing.assert_frame_equal(plotpack.read_table(paths["pdf"], "curve"), df)


def test_images_still_render_after_embedding(fig, tmp_path):
    paths = plotpack.save(fig, "c", out=tmp_path)
    assert plt.imread(paths["png"]).shape[2] in (3, 4)
    from pypdf import PdfReader
    assert len(PdfReader(paths["pdf"]).pages) == 1
    assert paths["svg"].read_text().count("<svg") == 1


def test_resaving_replaces_recipe_instead_of_stacking(fig, df, tmp_path):
    p = plotpack.save(fig, "c", data=df, out=tmp_path)["png"]
    plotpack.attach(p, data={"other": df.head(1)})
    chunks = [k for k, payload, _ in embed._png_chunks(p.read_bytes()) if embed._is_ours(k, payload)]
    assert len(chunks) == 1
    assert plotpack.inspect(p)["data"][0]["name"] == "other"


def test_unpack_writes_readme_manifest_source_data(fig, df, tmp_path):
    p = plotpack.save(fig, "c", data=df, out=tmp_path)["svg"]
    out = plotpack.unpack(p, tmp_path / "rec")
    assert {x.name for x in out.iterdir()} == {"README.md", "manifest.json", "source", "data"}
    assert "pd.read_csv(\"data/data.csv\")" in (out / "README.md").read_text()


def test_large_table_keeps_sample_and_full_hash(fig, tmp_path):
    big = pd.DataFrame({"v": np.arange(50_000)})
    m = plotpack.inspect(plotpack.save(fig, "big", data=big, out=tmp_path, max_data_bytes=10_000)["png"])
    d = m["data"][0]
    assert d["truncated"] and d["sample_rows"] == 1000 and d["rows"] == 50_000 and d["sha256_full"]


def test_verify_detects_tampering(fig, df, tmp_path):
    recipe = plotpack.save(fig, "c", data=df, out=tmp_path)["plotpack"].read_bytes()
    src = zipfile.ZipFile(io.BytesIO(recipe))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in src.namelist():
            z.writestr(n, b"x,y\n9,9\n" if n == "data/data.csv" else src.read(n))
    assert "data/data.csv" in plotpack.verify(buf.getvalue())[0]


def test_no_recipe_is_a_clear_error(fig, tmp_path):
    p = tmp_path / "plain.png"
    fig.savefig(p)
    with pytest.raises(plotpack.NoRecipe):
        plotpack.inspect(p)
    assert cli(["inspect", str(p)]) == 2


def test_reads_legacy_paperpack_figures(tmp_path):
    """Figures made by the old paperpack kit: PNG keyword 'Paperpack', paperpack.figure/v1 manifest."""
    import struct, zlib
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps({"schema": "paperpack.figure/v1", "name": "old",
                                                "generator": "source/ex.py", "parameters": {"a": 1},
                                                "data": [{"path": "data/t.csv", "bytes": 4, "sha256": "x"}],
                                                "environment": {"python": "3.12", "numpy": "2.0"}}))
    f, ax = plt.subplots(); ax.plot([0, 1]); p = tmp_path / "old.png"; f.savefig(p); plt.close(f)
    content = p.read_bytes()
    payload = b"Paperpack\0\x01\x00\0\0" + zlib.compress(buf.getvalue())
    chunk = struct.pack(">I", len(payload)) + b"iTXt" + payload + struct.pack(">I", zlib.crc32(b"iTXt" + payload))
    p.write_bytes(content[:-12] + chunk + content[-12:])
    m = plotpack.inspect(p)
    assert m["name"] == "old" and m["params"] == {"a": 1} and m["data"][0]["name"] == "t"


def test_git_state_recorded(fig, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    script = repo / "make_fig.py"
    script.write_text("print('hi')\n")
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    m = plotpack.inspect(plotpack.save(fig, "g", source=script, out=tmp_path)["png"])
    assert len(m["git"]["commit"]) == 40 and m["git"]["dirty"] is False
    assert m["git"]["path_in_repo"] == "make_fig.py" and m["source"]["kind"] == "file"
    assert "branch" not in m["git"] and "remote" not in m["git"]
    assert "platform" not in m["environment"]


def test_source_and_git_can_be_omitted(fig, tmp_path):
    paths = plotpack.save(fig, "public", data={"curve": [1, 2]}, source=False,
                          include_git=False, formats=["png"], out=tmp_path)
    manifest = plotpack.inspect(paths["png"])
    assert manifest["source"] is None and manifest["git"] is None
    assert manifest["data"][0]["path"] == "data/curve.json"


def test_notebook_omits_outputs_and_metadata(fig, tmp_path):
    notebook = tmp_path / "plot.ipynb"
    notebook.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"private_path": "/private/work"},
        "cells": [{"cell_type": "code", "source": ["plot()"], "execution_count": 42,
                   "metadata": {"private": "value"}, "outputs": [{"text": "private result"}]}]}))
    recipe = plotpack.save(fig, "nb", source=notebook, formats=["png"], out=tmp_path)["plotpack"]
    with zipfile.ZipFile(recipe) as archive:
        captured = json.loads(archive.read("source/plot.ipynb"))
    assert captured["metadata"] == {} and captured["cells"][0]["metadata"] == {}
    assert captured["cells"][0]["outputs"] == [] and captured["cells"][0]["execution_count"] is None
    assert captured["cells"][0]["source"] == ["plot()"]


@pytest.mark.parametrize("name", ["../escape", "..", "a\\b"])
def test_save_rejects_directory_names(fig, tmp_path, name):
    with pytest.raises(ValueError, match="filename"):
        plotpack.save(fig, name, out=tmp_path)


@pytest.mark.parametrize("member", ["../escape", "data/../escape", "data\\escape", "C:/escape"])
def test_unpack_rejects_cross_platform_paths(tmp_path, member):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(member, "payload")
    with pytest.raises(ValueError, match="unsafe path"):
        plotpack.unpack(buf.getvalue(), tmp_path / "out")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("names", [
    ("data/a.txt", "data/./a.txt"), ("data/A.txt", "data/a.txt"),
    ("data/café.txt", "data/cafe\u0301.txt"), ("data", "data/a.txt"),
])
def test_unpack_rejects_destination_collisions(tmp_path, names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.writestr(name, "payload")
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="paths? in recipe"):
        plotpack.unpack(buf.getvalue(), dest)
    assert not dest.exists()


def test_unpack_refuses_symlink_escape_and_overwrites(fig, tmp_path):
    recipe = plotpack.save(fig, "safe", data={"x": [1]}, formats=["png"], out=tmp_path)["plotpack"]
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "recovered"
    dest.mkdir()
    (dest / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe extraction"):
        plotpack.unpack(recipe, dest)
    assert list(outside.iterdir()) == []
    (dest / "data").unlink()
    plotpack.unpack(recipe, dest)
    with pytest.raises(FileExistsError, match="overwrite"):
        plotpack.unpack(recipe, dest)


def test_archive_expansion_limit(monkeypatch):
    from plotpack import recipe
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data/large.txt", "x" * 1025)
    monkeypatch.setattr(recipe, "MAX_RECIPE_BYTES", 1024)
    with pytest.raises(ValueError, match="limit"):
        recipe.open_recipe(buf.getvalue())


def test_json_limit_and_colliding_filenames():
    from plotpack.recipe import normalise_data
    items = normalise_data({"a": "x" * 1000}, max_bytes=10)
    assert items[0].content == b"" and items[0].meta["sha256_full"]
    names = [item.filename for item in normalise_data({"x": [1], "x?": [2], "x-1": [3], "x!": [4]})]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("name,content", [("bad.plotpack", b"PK\x03\x04broken"), ("bad.pdf", b"%PDF-1.4\nbroken"),
                                          ("notes.txt", b"hello")])
def test_unreadable_file_is_a_one_line_error(tmp_path, capsys, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    assert cli(["inspect", str(path)]) == 2
    assert cli(["verify", str(path), str(tmp_path / "missing.png")]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.count("\n") == 3 and "missing.png" in err


def test_help_does_not_import_heavy_modules():
    script = ("import sys\nfrom plotpack import cli\ntry:\n    cli.main(['--help'])\nexcept SystemExit:\n    pass\n"
              "assert not {'pypdf', 'matplotlib', 'numpy', 'pandas'} & set(sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0 and "Figures that carry" in result.stdout, result.stderr


@pytest.mark.parametrize("field,value", [("git", "invalid"), ("git", {"commit": 1}),
    ("environment", {"packages": []}), ("data", [{"path": "data/x", "bytes": "invalid"}]), ("data", [1])])
def test_malformed_manifest_cli_is_clean(tmp_path, capsys, field, value):
    path = tmp_path / "malformed.plotpack"
    manifest = {"schema": "plotpack/1", "data": [], field: value}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    assert cli(["inspect", str(path)]) in (0, 2)     # printed or rejected, never a traceback
    assert "Traceback" not in capsys.readouterr().err


def test_attach_missing_data_file_is_clean(fig, tmp_path, capsys):
    image = tmp_path / "figure.png"
    fig.savefig(image)
    before = image.read_bytes()
    assert cli(["attach", str(image), "--data", str(tmp_path / "missing.csv")]) == 2
    assert image.read_bytes() == before
    assert "Traceback" not in capsys.readouterr().err


