"""ScanImage-specific TIFF handling.

ScanImage timeseries can be:
- plain single-plane timeseries (``T, Y, X``)
- volumetric ("FastZ"/stack) timeseries, where each timepoint is stored on
  disk as a group of consecutive pages (Z-planes, sometimes plus a trailing
  "flyback" frame that must be dropped)
- split across several sibling files when a single acquisition exceeds a
  per-file frame limit

This module parses ScanImage's ``SI.*`` metadata (falling back to the
per-page ``Software`` tag when the file has no BigTIFF metadata header),
decides how to reshape the flat page stack accordingly, and locates/orders
the sibling files of a split acquisition.

Note: `find_scanimage_series_files` (auto-discovery from a single file) is
what handles an ordinary, single-file drag-and-drop of one part of a split
acquisition - no special napari "stack" interaction is needed for that
case. Passing an explicit list of files (which napari only does when files
are opened "as a stack", e.g. via *File > Open Files as Stack...*) instead
honors exactly that subset/order; see `napari_tiff.napari_tiff_reader.
scanimage_reader_function` and its module's `napari_get_reader` docstring
for the full explanation of when napari passes a list vs. a single path.
"""
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import dask.array as da
import tifffile
from tifffile import TiffFile

from napari_tiff.napari_tiff_multifile import lazy_series_array, natural_sort

# FrameData keys that are expected to legitimately vary between files
# belonging to the very same acquisition and must be excluded when checking
# whether two files share the same static configuration.
_PER_FILE_FRAMEDATA_KEYS = frozenset(
    {
        "SI.hScan2D.logFileCounter",
        "SI.hScan2D.logFilePath",
        "SI.hScan2D.logFileStem",
    }
)

# `<base>_<acquisition>_<fileIndex>.<ext>` (split/multi-file acquisition) or
# `<base>_<acquisition>.<ext>` (single-file acquisition).
_MULTI_FILE_RE = re.compile(r"^(?P<base>.+?)_(?P<acquisition>\d+)_(?P<file_index>\d+)$")
_SINGLE_FILE_RE = re.compile(r"^(?P<base>.+?)_(?P<acquisition>\d+)$")

_FRAME_NUMBER_RE = re.compile(r"frameNumbers\s*=\s*(-?\d+)")


def log_warning(msg: str) -> None:
    """Log message with level WARNING."""
    logging.getLogger(__name__).warning(msg)


def get_scanimage_framedata(tif: TiffFile) -> dict[str, Any]:
    """Return the ScanImage ``FrameData`` dict (non-varying acquisition settings).

    Prefers `tif.scanimage_metadata`, which is only populated for ScanImage
    BigTIFF files with the special metadata header. Falls back to parsing
    any page's `Software` tag, which carries the identical information on
    every ScanImage TIFF (BigTIFF or not).
    """
    meta = tif.scanimage_metadata
    framedata = (meta or {}).get("FrameData")
    if framedata:
        return framedata

    try:
        software = tif.pages[0].tags["Software"].value
    except Exception:
        return {}

    try:
        return tifffile.matlabstr2py(software)
    except Exception:
        return {}


@dataclass
class ScanImageDims:
    """Describes how to reshape a flat ScanImage page stack.

    `axes` is one of ``'TYX'`` (plain timeseries), ``'TZYX'`` (volumetric
    timeseries), or ``'IYX'`` (flat fallback - could not confirm structure).
    """

    axes: str
    frames_per_group: int
    frames_to_keep: int
    n_slices: int
    n_channels: int
    warning: str | None = None


def _flat_dims(reason: str | None = None) -> ScanImageDims:
    return ScanImageDims(
        axes="IYX",
        frames_per_group=1,
        frames_to_keep=1,
        n_slices=1,
        n_channels=1,
        warning=reason,
    )


def compute_scanimage_dimensions(
    framedata: dict[str, Any], total_pages: int
) -> ScanImageDims:
    """Decide how to reshape a ScanImage page stack into T[,Z],Y,X.

    Gates volumetric interpretation on `SI.hStackManager.enable`/
    `SI.hFastZ.enable` (never on the mere presence of slice-count fields,
    which can be stale leftovers from a previous, inactive configuration).
    Cross-validates the chosen on-disk group size against `total_pages`
    before committing to a reshape; falls back to a flat interpretation
    (with a warning) on any inconsistency.
    """
    if total_pages <= 0:
        return _flat_dims("no pages to interpret")

    channel_save = framedata.get("SI.hChannels.channelSave")
    if isinstance(channel_save, (list, tuple)):
        n_channels = len(channel_save)
    else:
        n_channels = 1

    if n_channels > 1:
        # Multi-channel on-disk frame ordering (interleaved per Z-plane vs.
        # grouped by channel) has not been validated against a real
        # multi-channel ScanImage file. Degrade safely rather than guess.
        return ScanImageDims(
            axes="IYX",
            frames_per_group=1,
            frames_to_keep=1,
            n_slices=1,
            n_channels=n_channels,
            warning=(
                "multi-channel ScanImage files are not yet supported for "
                "volumetric/channel-aware reshaping; falling back to a flat "
                "interpretation"
            ),
        )

    volumetric = bool(framedata.get("SI.hStackManager.enable")) or bool(
        framedata.get("SI.hFastZ.enable")
    )

    if not volumetric:
        return ScanImageDims(
            axes="TYX",
            frames_per_group=1,
            frames_to_keep=1,
            n_slices=1,
            n_channels=n_channels,
        )

    n_slices = framedata.get("SI.hStackManager.actualNumSlices")
    frames_per_volume = framedata.get("SI.hStackManager.numFramesPerVolume")
    frames_per_volume_flyback = framedata.get(
        "SI.hStackManager.numFramesPerVolumeWithFlyback"
    )

    if not n_slices or not frames_per_volume:
        return _flat_dims(
            "hStackManager/hFastZ reports volumetric acquisition, but slice "
            "count fields are missing; falling back to a flat interpretation"
        )

    try:
        n_slices = int(n_slices)
        frames_per_volume = int(frames_per_volume)
        on_disk_group = int(frames_per_volume_flyback or frames_per_volume)
    except (TypeError, ValueError):
        return _flat_dims(
            "could not parse hStackManager slice-count fields as integers"
        )

    if on_disk_group <= 0 or total_pages % on_disk_group:
        return _flat_dims(
            f"total page count ({total_pages}) is not evenly divisible by "
            f"the expected on-disk frames-per-volume ({on_disk_group}); "
            "falling back to a flat interpretation"
        )

    if not (0 < frames_per_volume <= on_disk_group):
        return _flat_dims("inconsistent frames-per-volume metadata")

    return ScanImageDims(
        axes="TZYX",
        frames_per_group=on_disk_group,
        frames_to_keep=frames_per_volume,
        n_slices=n_slices,
        n_channels=n_channels,
    )


def _parse_scanimage_filename(path: str) -> tuple[str, str, int | None]:
    """Return (base, acquisition, file_index) parsed from a ScanImage filename.

    `file_index` is `None` for the single-file naming form.
    """
    stem = Path(path).stem
    match = _MULTI_FILE_RE.match(stem)
    if match:
        return match["base"], match["acquisition"], int(match["file_index"])
    match = _SINGLE_FILE_RE.match(stem)
    if match:
        return match["base"], match["acquisition"], None
    return stem, "", None


def _comparable_framedata(framedata: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in framedata.items() if k not in _PER_FILE_FRAMEDATA_KEYS}


def _frame_number(page: Any) -> int | None:
    match = _FRAME_NUMBER_RE.search(page.description or "")
    return int(match.group(1)) if match else None


def warn_on_frame_number_gaps(paths: Sequence[str]) -> None:
    """Log a warning if selected files are not a contiguous timeline.

    This is advisory only: users may intentionally select a partial or
    non-contiguous subset of a split acquisition's files.
    """
    previous_last = None
    for path in paths:
        try:
            with TiffFile(path) as tif:
                first = _frame_number(tif.pages[0])
                last = _frame_number(tif.pages[-1])
        except Exception:
            return
        if previous_last is not None and first is not None and first != previous_last + 1:
            warnings.warn(
                f"selected ScanImage files are not contiguous: frame "
                f"{previous_last} is followed by frame {first} at {path!r} "
                "- the combined timeline will have a gap"
            )
        previous_last = last if last is not None else previous_last


def find_scanimage_series_files(path: str) -> list[str]:
    """Find and order sibling files belonging to the same ScanImage acquisition as `path`.

    Called from `scanimage_reader_function` whenever it receives a single
    path (the common case: an ordinary drag-and-drop of one file, with no
    napari "stack" interaction required from the user). Returns just
    `[path]` if `path` uses the single-file naming form, or if no confirmed
    siblings are found.
    """
    path = Path(path)
    base, acquisition, _file_index = _parse_scanimage_filename(str(path))
    if not acquisition:
        return [str(path)]

    candidates = [p for p in path.parent.glob(f"*{path.suffix}") if p.is_file()]
    siblings = []
    for candidate in candidates:
        c_base, c_acquisition, c_index = _parse_scanimage_filename(str(candidate))
        if c_base == base and c_acquisition == acquisition and c_index is not None:
            siblings.append(str(candidate))

    if len(siblings) <= 1:
        return [str(path)]

    siblings = natural_sort(siblings)

    reference_framedata = None
    confirmed = []
    for sibling in siblings:
        try:
            with TiffFile(sibling) as tif:
                if not tif.is_scanimage:
                    continue
                framedata = _comparable_framedata(get_scanimage_framedata(tif))
        except Exception as exc:
            log_warning(f"failed to inspect potential ScanImage sibling {sibling!r}: {exc}")
            continue

        if reference_framedata is None:
            reference_framedata = framedata
            confirmed.append(sibling)
        elif framedata == reference_framedata:
            confirmed.append(sibling)
        else:
            log_warning(
                f"{sibling!r} looked like a sibling of {path!r} by file name, "
                "but its ScanImage metadata differs - excluding it from the "
                "combined acquisition"
            )

    if not confirmed:
        return [str(path)]

    warn_on_frame_number_gaps(confirmed)
    return confirmed


def build_scanimage_layerdata(
    paths: Sequence[str], dims: ScanImageDims
) -> tuple[da.Array, str]:
    """Lazily build the combined, correctly-shaped array for a ScanImage acquisition."""
    volumes = []
    for path in paths:
        with TiffFile(path) as tif:
            n_pages = len(tif.pages)
        array, _axes, shape = lazy_series_array(path)
        if shape[0] != n_pages:
            # tifffile's declared series shape (from SI.hStackManager metadata,
            # e.g. framesPerSlice) can exceed the pages actually present on
            # disk, for example when a file is part of a longer acquisition
            # whose continuation files were not found/selected. Always trust
            # the real, on-disk page count to avoid addressing pages that
            # don't exist.
            log_warning(
                f"{path!r} declares {shape[0]} frames in its metadata, but "
                f"only {n_pages} are actually present on disk (it may be "
                "part of a longer acquisition whose continuation files were "
                "not found); using the actual page count"
            )
            array = array[:n_pages]

        n_groups, remainder = divmod(n_pages, dims.frames_per_group)
        if remainder:
            log_warning(
                f"{path!r} has {n_pages} pages, not a multiple of the expected "
                f"{dims.frames_per_group} frames per volume; the trailing "
                "incomplete group will be dropped"
            )
            array = array[: n_groups * dims.frames_per_group]

        y, x = array.shape[-2:]
        grouped = array.reshape((n_groups, dims.frames_per_group, y, x))
        kept = grouped[:, : dims.frames_to_keep]
        if dims.n_slices <= 1:
            kept = kept.reshape((n_groups, y, x))
        volumes.append(kept)

    combined = da.concatenate(volumes, axis=0)
    return combined, dims.axes
