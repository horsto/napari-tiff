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
the sibling files of a split acquisition. See `napari_tiff.
napari_tiff_reader.scanimage_reader_function` for how single-file
auto-discovery vs. an explicit file list are handled.
"""
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import dask.array as da
import tifffile
import zarr
from tifffile import TiffFile

from napari_tiff.napari_tiff_multifile import natural_sort

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

    `axes` is one of ``'TYX'``, ``'TCYX'``, ``'TZYX'``, ``'TZCYX'``, or
    ``'IYX'`` (flat fallback - could not confirm structure). On disk, one
    timepoint occupies `pages_per_step` consecutive pages: `z_group_size`
    Z-positions (channel-minor: each Z-position's `n_channels` pages are
    adjacent), of which only the first `z_keep` are real data (any excess
    is a trailing "flyback" position to drop).
    """

    axes: str
    pages_per_step: int
    z_group_size: int
    z_keep: int
    n_channels: int
    warning: str | None = None


def _flat_dims(reason: str | None = None) -> ScanImageDims:
    return ScanImageDims(
        axes="IYX",
        pages_per_step=1,
        z_group_size=1,
        z_keep=1,
        n_channels=1,
        warning=reason,
    )


def compute_scanimage_dimensions(
    framedata: dict[str, Any], total_pages: int
) -> ScanImageDims:
    """Decide how to reshape a ScanImage page stack into T[,Z][,C],Y,X.

    Gates volumetric interpretation on `SI.hStackManager.enable`/
    `SI.hFastZ.enable` (never on the mere presence of slice-count fields,
    which can be stale leftovers from a previous, inactive configuration).
    On disk, channels are always the fastest-varying axis (confirmed
    against real single- and multi-channel, flat and volumetric ScanImage
    files: consecutive pages share the same `frameNumbers`/Z-position and
    only differ by channel). Cross-validates the chosen page grouping
    against `total_pages` before committing to a reshape; falls back to a
    flat interpretation (with a warning) on any inconsistency, and for
    `SI.hStackManager.framesPerSlice > 1` volumetric acquisitions (whether
    that extra per-slice dimension nests inside or outside Z has not been
    verified against a real file).
    """
    if total_pages <= 0:
        return _flat_dims("no pages to interpret")

    channel_save = framedata.get("SI.hChannels.channelSave")
    n_channels = len(channel_save) if isinstance(channel_save, (list, tuple)) else 1

    volumetric = bool(framedata.get("SI.hStackManager.enable")) or bool(
        framedata.get("SI.hFastZ.enable")
    )

    if not volumetric:
        z_group_size = 1
        z_keep = 1
    else:
        frames_per_slice = framedata.get("SI.hStackManager.framesPerSlice")
        try:
            unsupported_repeat = int(frames_per_slice) != 1
        except (TypeError, ValueError):
            unsupported_repeat = False
        if unsupported_repeat:
            return _flat_dims(
                f"volumetric acquisitions with more than one frame per "
                f"Z-position (SI.hStackManager.framesPerSlice="
                f"{frames_per_slice}) are not yet supported for reshaping; "
                "falling back to a flat interpretation"
            )

        n_slices = framedata.get("SI.hStackManager.actualNumSlices")
        frames_per_volume = framedata.get("SI.hStackManager.numFramesPerVolume")
        frames_per_volume_flyback = framedata.get(
            "SI.hStackManager.numFramesPerVolumeWithFlyback"
        )
        if not n_slices or not frames_per_volume:
            return _flat_dims(
                "hStackManager/hFastZ reports volumetric acquisition, but "
                "slice count fields are missing; falling back to a flat "
                "interpretation"
            )
        try:
            frames_per_volume = int(frames_per_volume)
            z_group_size = int(frames_per_volume_flyback or frames_per_volume)
        except (TypeError, ValueError):
            return _flat_dims(
                "could not parse hStackManager slice-count fields as integers"
            )
        if not (0 < frames_per_volume <= z_group_size):
            return _flat_dims("inconsistent frames-per-volume metadata")
        z_keep = frames_per_volume

    pages_per_step = z_group_size * n_channels
    if pages_per_step <= 0 or total_pages % pages_per_step:
        return _flat_dims(
            f"total page count ({total_pages}) is not evenly divisible by "
            f"the expected pages per timepoint ({pages_per_step} = "
            f"{z_group_size} Z-position(s) x {n_channels} channel(s)); "
            "falling back to a flat interpretation"
        )

    if z_keep <= 1 and n_channels <= 1:
        axes = "TYX"
    elif z_keep <= 1:
        axes = "TCYX"
    elif n_channels <= 1:
        axes = "TZYX"
    else:
        axes = "TZCYX"

    return ScanImageDims(
        axes=axes,
        pages_per_step=pages_per_step,
        z_group_size=z_group_size,
        z_keep=z_keep,
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

    Returns just `[path]` if it uses the single-file naming form, or if no
    siblings pass the static-metadata check below.
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


def _lazy_flat_page_array(path: str) -> tuple[da.Array, tuple[int, ...]]:
    """Return a lazy, page-indexed ``(n_pages, Y, X)`` dask array for `path`.

    Built directly from each page's own store rather than `tif.series[0]`:
    tifffile's built-in ScanImage series-shape inference derives its "Z"
    axis purely from `SI.hStackManager.framesPerSlice` and the total page
    count, which disagrees with the true on-disk layout once channels are
    involved (`compute_scanimage_dimensions` implements the correct
    interpretation instead, so this always starts from the raw pages).
    """
    with TiffFile(path) as tif:
        page_shape = tif.pages[0].shape
        stores = [page.aszarr() for page in tif.pages]
    arrays = [da.from_zarr(zarr.open(store, mode="r")) for store in stores]
    combined = da.stack(arrays, axis=0)
    return combined, (len(arrays),) + page_shape


def build_scanimage_layerdata(
    paths: Sequence[str], dims: ScanImageDims
) -> tuple[da.Array, str]:
    """Lazily build the combined, correctly-shaped array for a ScanImage acquisition."""
    volumes = []
    for path in paths:
        array, shape = _lazy_flat_page_array(path)
        n_pages = shape[0]

        n_steps, remainder = divmod(n_pages, dims.pages_per_step)
        if remainder:
            log_warning(
                f"{path!r} has {n_pages} pages, not a multiple of the expected "
                f"{dims.pages_per_step} pages per timepoint; the trailing "
                "incomplete timepoint will be dropped"
            )
            array = array[: n_steps * dims.pages_per_step]

        y, x = shape[-2:]
        grouped = array.reshape((n_steps, dims.z_group_size, dims.n_channels, y, x))
        kept = grouped[:, : dims.z_keep]  # drop trailing flyback Z-position(s)

        if dims.z_keep <= 1 and dims.n_channels <= 1:
            kept = kept.reshape((n_steps, y, x))
        elif dims.z_keep <= 1:
            kept = kept.reshape((n_steps, dims.n_channels, y, x))
        elif dims.n_channels <= 1:
            kept = kept.reshape((n_steps, dims.z_keep, y, x))
        volumes.append(kept)

    combined = da.concatenate(volumes, axis=0)
    return combined, dims.axes
