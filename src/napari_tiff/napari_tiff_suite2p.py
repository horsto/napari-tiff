"""Suite2p output-folder handling.

Suite2p writes one subfolder per imaging plane (Z) directly inside its
output root (``plane0``, ``plane1``, ...), and inside each plane one
subfolder per registered channel of motion-corrected movie chunks
(``reg_tif`` = channel 0, ``reg_tif2`` = channel 1), each chunk itself a
multi-frame TIFF. This module detects a suite2p output root when dropped
directly, and stitches everything into one ``(T, Z[, C], Y, X)`` array:
chunks within a folder are concatenated along T (suite2p's own chunking
order is chronological, so this is a genuine timeseries - no axis-label
guessing needed), planes become Z, and reg_tif/reg_tif2 become C.
"""
import logging
import os
import re
from pathlib import Path
from typing import Any

import dask.array as da

from napari_tiff.napari_tiff_multifile import lazy_series_array, list_tiff_files_in_directory, natural_sort

_PLANE_DIR_RE = re.compile(r"^plane\d+$", re.IGNORECASE)
_CHANNEL_DIRS = ("reg_tif", "reg_tif2")  # channel 0, channel 1
_MARKER_FILES = frozenset({"ops.npy", "db.npy", "settings.npy", "run.log", "filelist.npy"})


def log_warning(msg: str) -> None:
    """Log message with level WARNING."""
    logging.getLogger(__name__).warning(msg)


def is_suite2p_output_directory(directory: str) -> bool:
    """Return whether `directory` looks like a suite2p output root.

    Requires at least one `planeN` subdirectory plus at least one known
    suite2p marker file directly inside `directory`, so an unrelated
    folder that merely has a `plane0`-named subdirectory isn't mistaken
    for one.
    """
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return False
    has_plane_dir = any(e.is_dir() and _PLANE_DIR_RE.match(e.name) for e in entries)
    has_marker = any(e.is_file() and e.name.lower() in _MARKER_FILES for e in entries)
    return has_plane_dir and has_marker


def find_plane_directories(directory: str) -> list[str]:
    """Return naturally-sorted `planeN` subdirectories directly inside `directory`."""
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return []
    return natural_sort(e.path for e in entries if e.is_dir() and _PLANE_DIR_RE.match(e.name))


def find_channel_directories(plane_dir: str) -> list[str]:
    """Return the `reg_tif`/`reg_tif2` subfolders present in `plane_dir`, in channel order."""
    return [
        str(Path(plane_dir) / name)
        for name in _CHANNEL_DIRS
        if (Path(plane_dir) / name).is_dir()
    ]


def build_suite2p_layerdata(directory: str) -> tuple[da.Array, str, str]:
    """Lazily build the combined `(T, Z[, C], Y, X)` array for a suite2p output folder.

    Returns the combined dask array, its axes string, and the path of the
    first TIFF chunk encountered (a reference file for deriving metadata).
    """
    plane_dirs = find_plane_directories(directory)
    if not plane_dirs:
        raise ValueError(f"no plane subfolders found in {directory!r}")

    channels_per_plane = [
        {Path(c).name for c in find_channel_directories(p)} for p in plane_dirs
    ]
    common_channels = [c for c in _CHANNEL_DIRS if all(c in s for s in channels_per_plane)]
    if not common_channels:
        raise ValueError(f"no common reg_tif/reg_tif2 folders across planes in {directory!r}")
    if any(set(common_channels) != s for s in channels_per_plane):
        log_warning(
            f"not all planes in {directory!r} have the same registered-channel "
            f"folders; only using the common channels {common_channels}"
        )

    reference_path = None
    per_plane_arrays = []
    for plane_dir in plane_dirs:
        channel_arrays = []
        for channel in common_channels:
            channel_dir = str(Path(plane_dir) / channel)
            files = list_tiff_files_in_directory(channel_dir)
            if not files:
                raise ValueError(f"no TIFF files found in {channel_dir!r}")
            if reference_path is None:
                reference_path = files[0]
            chunks = [lazy_series_array(f)[0] for f in files]
            channel_arrays.append(da.concatenate(chunks, axis=0))
        per_plane_arrays.append(
            da.stack(channel_arrays, axis=1) if len(channel_arrays) > 1 else channel_arrays[0]
        )

    # different planes may have (very slightly) different total frame counts;
    # truncate to the shortest so they can be stacked into one Z axis
    min_frames = min(arr.shape[0] for arr in per_plane_arrays)
    if any(arr.shape[0] != min_frames for arr in per_plane_arrays):
        log_warning(
            f"planes in {directory!r} have differing total frame counts; "
            f"truncating all to the shortest ({min_frames} frames)"
        )
    per_plane_arrays = [arr[:min_frames] for arr in per_plane_arrays]

    combined = da.stack(per_plane_arrays, axis=1)
    axes = "TZCYX" if len(common_channels) > 1 else "TZYX"
    return combined, axes, reference_path


def get_suite2p_metadata(reference_path: str, axes: str) -> dict[str, Any]:
    """Return napari layer metadata for a combined suite2p acquisition.

    Reuses `reference_path`'s own X/Y scale/units; T, Z, and C axes have no
    physical calibration available here, so they default to pixel spacing.
    """
    from napari_tiff.napari_tiff_metadata import get_metadata
    from tifffile import TiffFile

    with TiffFile(reference_path) as tif:
        base_kwargs = get_metadata(tif)
    base_scale = base_kwargs.get("scale")
    base_units = base_kwargs.get("units")

    xy_scale = dict(zip("YX", base_scale[-2:])) if base_scale else {}
    xy_units = dict(zip("YX", base_units[-2:])) if base_units else {}

    scale = []
    units = []
    channel_axis = None
    for i, axis in enumerate(axes):
        if axis in xy_scale:
            scale.append(xy_scale[axis])
            units.append(xy_units.get(axis, "pixel"))
        else:
            scale.append(1.0)
            units.append("pixel")
        if axis == "C":
            channel_axis = i

    return dict(
        name="suite2p",
        axis_labels=tuple(a.lower() for a in axes),
        scale=tuple(scale),
        units=tuple(units),
        channel_axis=channel_axis,
    )
