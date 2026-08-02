"""Generic, non-ScanImage-specific helpers for combining multiple TIFF files.

When a user drags/selects several TIFF files into napari at once, this
module decides whether they can be stacked into one array, orders them
naturally, and infers a defensible label (e.g. ``T``) for the new axis
created by joining them - falling back to a neutral, non-committal label
when there isn't enough evidence to call it a timeseries.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import dask.array as da
import zarr
from natsort import natsorted
from tifffile import TiffFile

# Generic/undifferentiated axis codes used by tifffile itself for a plain
# multi-page TIFF (no OME/ImageJ/etc. metadata to say what the pages mean).
_AMBIGUOUS_AXIS_CODES = ("Q", "I")


def natural_sort(paths: Sequence[Any]) -> list[Any]:
    """Return `paths` sorted in human ('natural') order.

    E.g. ``img2.tif`` sorts before ``img10.tif``.
    """
    return natsorted(paths, key=str)


@dataclass
class CompatibilityReport:
    """Result of checking whether a set of files can be stacked together."""

    compatible: bool
    axes: str = ""
    dtype: Any = None
    reason: str = ""


def _normalized_signature(path: str) -> tuple[str, tuple[int, ...], Any]:
    """Return (axes, shape, dtype) for a file's first series.

    A leading axis is synthesized (as ``'Q'``, length 1) for plain 2D
    (``YX``) series, so every file's signature has a "joinable" leading axis
    for uniform downstream handling.
    """
    with TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        dtype = series.dtype
    if len(axes) == 2:
        axes = "Q" + axes
        shape = (1,) + shape
    return axes, shape, dtype


def check_compatible(paths: Sequence[str]) -> CompatibilityReport:
    """Check whether `paths` share a common shape/dtype/axes so they can be stacked.

    Only the first series of each file's metadata is inspected (no pixel
    data is read). The leading (joining) axis length is allowed to differ
    between files; every other dimension, plus dtype and axes, must match.
    """
    if not paths:
        return CompatibilityReport(compatible=False, reason="no files given")

    try:
        ref_axes, ref_shape, ref_dtype = _normalized_signature(paths[0])
    except Exception as exc:
        return CompatibilityReport(
            compatible=False, reason=f"failed to read {paths[0]!r}: {exc}"
        )

    for path in paths[1:]:
        try:
            axes, shape, dtype = _normalized_signature(path)
        except Exception as exc:
            return CompatibilityReport(
                compatible=False, reason=f"failed to read {path!r}: {exc}"
            )
        if axes != ref_axes or shape[1:] != ref_shape[1:] or dtype != ref_dtype:
            return CompatibilityReport(
                compatible=False,
                reason=(
                    f"{path!r} ({axes}, {shape}, {dtype}) is not compatible with "
                    f"{paths[0]!r} ({ref_axes}, {ref_shape}, {ref_dtype})"
                ),
            )

    return CompatibilityReport(compatible=True, axes=ref_axes, dtype=ref_dtype)


# Filename tokens that give high-confidence evidence about what a new,
# file-joining axis represents.
_AXIS_TOKEN_PATTERNS = {
    "T": re.compile(r"(?:^|[_\-\s])(?:t|time|frame)0*\d+", re.IGNORECASE),
    "Z": re.compile(r"(?:^|[_\-\s])(?:z|slice|plane)0*\d+", re.IGNORECASE),
}


@dataclass
class AxisGuess:
    """Result of inferring what a new file-joining axis represents."""

    label: str
    confidence: str  # 'high' or 'low'
    reason: str


def infer_join_axis(paths: Sequence[str]) -> AxisGuess:
    """Infer a label for the axis created by joining `paths` together.

    Returns a conservative, generic label (``'I'``) unless file names carry
    positive, unambiguous evidence that the join axis represents a specific
    dimension such as time (``'T'``) or Z (``'Z'``).
    """
    stems = [Path(p).stem for p in paths]

    for axis, pattern in _AXIS_TOKEN_PATTERNS.items():
        if all(pattern.search(stem) for stem in stems):
            return AxisGuess(
                label=axis,
                confidence="high",
                reason=f"file names contain an explicit '{axis}' token",
            )

    return AxisGuess(
        label="I",
        confidence="low",
        reason=(
            "no reliable evidence for what the new axis represents "
            "(a bare numeric counter does not confirm a timeseries)"
        ),
    )


def lazy_series_array(
    path: str, series_index: int = 0
) -> tuple[da.Array, str, tuple[int, ...]]:
    """Return (lazy dask array, axes, shape) for one series in a TIFF file."""
    with TiffFile(path) as tif:
        series = tif.series[series_index]
        axes = series.axes
        shape = series.shape
        store = series.aszarr()
    zarray = zarr.open(store, mode="r")
    array = da.from_zarr(zarray, chunks="auto")
    if len(axes) == 2:
        # synthesize a joinable leading axis for plain 2D series
        array = array.reshape((1,) + array.shape)
        axes = "Q" + axes
        shape = (1,) + shape
    return array, axes, shape


def build_multifile_layerdata(
    paths: Sequence[str],
) -> tuple[da.Array, str, AxisGuess]:
    """Lazily concatenate `paths` (already ordered) along their leading axis.

    Returns the combined dask array, the axes string of the combined array,
    and the `AxisGuess` used to label the (possibly re-labeled) leading axis.
    """
    arrays = []
    axes = None
    for path in paths:
        array, file_axes, _shape = lazy_series_array(path)
        arrays.append(array)
        axes = file_axes

    combined = da.concatenate(arrays, axis=0)

    if axes[0] in _AMBIGUOUS_AXIS_CODES:
        guess = infer_join_axis(paths)
        combined_axes = guess.label + axes[1:]
    else:
        guess = AxisGuess(
            label=axes[0],
            confidence="high",
            reason="axis already identified by the file's own series metadata",
        )
        combined_axes = axes

    return combined, combined_axes, guess


def get_multifile_metadata(
    reference_path: str, combined_axes: str, guess: AxisGuess
) -> dict[str, Any]:
    """Return napari layer metadata for a combined multi-file stack.

    Reuses the single-file metadata (colormaps, per-axis scale/units, etc.)
    derived from `reference_path`, then prepends a neutral scale/unit entry
    for the new joining axis and records the axis-label confidence in the
    layer's `metadata` dict for user inspection.
    """
    from napari_tiff.napari_tiff_metadata import get_metadata

    with TiffFile(reference_path) as tif:
        base_kwargs = get_metadata(tif)

    channel_axis = base_kwargs.get("channel_axis")
    if channel_axis is not None:
        base_kwargs["channel_axis"] = channel_axis + 1

    scale = base_kwargs.get("scale")
    units = base_kwargs.get("units")
    if isinstance(scale, tuple):
        base_kwargs["scale"] = (1.0,) + scale
    if isinstance(units, tuple):
        base_kwargs["units"] = ("pixel",) + units

    base_kwargs["axis_labels"] = tuple(c.lower() for c in combined_axes)

    extra_metadata = base_kwargs.setdefault("metadata", {})
    extra_metadata["multifile_join_axis"] = {
        "label": guess.label,
        "confidence": guess.confidence,
        "reason": guess.reason,
    }
    return base_kwargs


def log_warning(msg: str) -> None:
    """Log message with level WARNING."""
    logging.getLogger(__name__).warning(msg)
