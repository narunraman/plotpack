# Plotpack

Figures that carry their data, generating code and provenance inside ordinary PNG,
SVG and PDF files. No embedded code is executed when reading or extracting a recipe.

## Install

Requires Python 3.12+. Install with `uv`:

```bash
uv add git+https://github.com/narunraman/plotpack
```

The figure object supplies `savefig` (for example, Matplotlib). To read CSV tables
as DataFrames, install the `tables` extra or add pandas separately.

## Save and recover

```python
import plotpack

paths = plotpack.save(fig, "curve", data={"points": df}, params={"seed": 0})
manifest = plotpack.inspect(paths["png"])
plotpack.unpack(paths["pdf"], "recovered")
```

```bash
plotpack inspect figures/curve.png --json
plotpack verify figures/curve.pdf figures/curve.png
plotpack unpack figures/curve.svg recovered/
plotpack attach existing.png --data points.csv --source draw.py
```

`unpack` refuses unsafe archive paths, symlink escapes, and overwriting existing
files. Use an empty destination. Malformed input produces a concise CLI error.
Recipes may contain at most 100 MB of expanded contents and 1,000 archive members.

## Sharing controls

By default, Plotpack captures the calling script or an identified notebook. Notebook
outputs, execution counts and metadata are removed; unrelated interactive history
is never captured. Scripts and notebook source can still contain private information.
Review the source, supplied data, parameters and notes before distributing figures.

```python
plotpack.save(fig, "public-curve", data=df, source=False, include_git=False)
```

For an existing image, use `plotpack attach image.png --data points.csv --no-source --no-git`.
Git provenance otherwise records commit, dirty state and relative source path; it
omits remote URLs and branch names. Environment metadata records Python and relevant
package versions.

Each data item has a default 25 MB embedding limit. Larger tables keep up to 1,000
rows plus a full-data hash; samples that exceed the limit are omitted. Oversized
files, arrays and JSON values keep only their full-data hash. Data used to replot
should be a small derived table rather than raw inputs. Legacy Paperpack figures
remain readable.
