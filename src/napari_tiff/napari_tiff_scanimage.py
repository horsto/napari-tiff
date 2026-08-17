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
from tifffile import TiffFile, TiffPageSeries

from napari_tiff.napari_tiff_multifile import natural_sort

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

    `axes` is `'T'` plus whichever of `'F'`, `'Z'`, `'C'` apply (in that
    order), plus `'YX'` - e.g. ``'TYX'``, ``'TZCYX'``, ``'TFZYX'`` - or
    ``'IYX'`` (flat fallback - could not confirm structure). `'Z'` -
    the only genuinely spatial non-displayed axis - is always kept
    immediately adjacent to `'Y,X'`, and `'F'` (repeated frames, a second
    time-like axis) is grouped next to `'T'` instead: napari's 3D
    rendering mode treats the *last* 3 axes of the array as the displayed
    volume, so this ordering ensures 3D mode renders true Z-depth rather
    than the frame-repeat axis.

    On disk, one timepoint (volume) occupies `pages_per_step` consecutive
    pages: `on_disk_raw_frames` raw frame-scans (pre-channel), each
    contributing `n_channels` adjacent pages (channels are always
    fastest-varying). Only the *first* `kept_raw_frames` raw frames are
    real data - any excess is a single trailing "flyback" raw frame
    appended after the whole volume (not per Z-position) to drop. The
    kept raw frames split evenly into `z_count` Z-positions of
    `frames_per_slice` repeated frames each, on disk Z-outer/frame-repeat-
    inner - `build_scanimage_layerdata` transposes this to frame-repeat-
    major/Z-minor to match `axes`.

    `on_disk_raw_frames`, `kept_raw_frames`, and `frames_per_slice` all
    describe what's actually *written to disk*, which can be smaller than
    what `SI.hStackManager` reports when `SI.hScan2D.logAverageFactor` > 1
    (ScanImage on-the-fly averages that many raw scanner frames into a
    single saved frame per Z-position). `log_average_factor`,
    `raw_frames_per_slice`, and `physical_frames_per_volume` preserve the
    *un-averaged* metadata values - needed for correctly sampling
    `SI.hStackManager.zs` (which has one entry per raw, pre-averaging
    frame) and for computing the true physical duration of a
    volume/frame-repeat step (averaging doesn't make the acquisition
    itself any faster) - and default to `1` for the common, non-averaged
    case.
    """

    axes: str
    pages_per_step: int
    on_disk_raw_frames: int
    kept_raw_frames: int
    z_count: int
    frames_per_slice: int
    n_channels: int
    log_average_factor: int = 1
    raw_frames_per_slice: int = 1
    physical_frames_per_volume: int = 1
    warning: str | None = None


def _flat_dims(reason: str | None = None) -> ScanImageDims:
    return ScanImageDims(
        axes="IYX",
        pages_per_step=1,
        on_disk_raw_frames=1,
        kept_raw_frames=1,
        z_count=1,
        frames_per_slice=1,
        n_channels=1,
        log_average_factor=1,
        raw_frames_per_slice=1,
        physical_frames_per_volume=1,
        warning=reason,
    )


def _compute_scanimage_structure(framedata: dict[str, Any]) -> ScanImageDims:
    """Decide the shape of one ScanImage timepoint, independent of page count.

    Does everything `compute_scanimage_dimensions` does *except* checking
    whether any particular file's `total_pages` is enough to hold even one
    timepoint - so this can be used to compare the *structure* two files'
    metadata implies (e.g. when checking whether several dropped files
    belong to the same acquisition) without one file's truncation making
    that comparison meaningless. `pages_per_step` is still computed here;
    `compute_scanimage_dimensions` is the one that decides whether a given
    `total_pages` actually satisfies it.

    Gates volumetric interpretation on `SI.hStackManager.enable`/
    `SI.hFastZ.enable` (never on the mere presence of slice-count fields,
    which can be stale leftovers from a previous, inactive configuration).
    On disk, channels are always the fastest-varying axis, and
    `SI.hStackManager.framesPerSlice` repeated frames are captured at each
    Z-position before moving to the next one. Any flyback overhead
    (`numFramesPerVolumeWithFlyback - numFramesPerVolume`) is always a
    single raw frame appended once at the very end of the whole volume's
    real Z x frame-repeat data - never multiplied by `framesPerSlice` or
    folded into an extra Z-position - confirmed against real single- and
    multi-volume acquisitions with 1-2 channels and `framesPerSlice` of 1
    or 20. Cross-validates `actualNumSlices x framesPerSlice ==
    numFramesPerVolume` before committing to a reshape; falls back to a
    flat interpretation (with a warning) on metadata inconsistencies.
    """
    channel_save = framedata.get("SI.hChannels.channelSave")
    n_channels = len(channel_save) if isinstance(channel_save, (list, tuple)) else 1

    volumetric = bool(framedata.get("SI.hStackManager.enable")) or bool(
        framedata.get("SI.hFastZ.enable")
    )

    if not volumetric:
        z_count = 1
        frames_per_slice = 1
        kept_raw_frames = 1
        on_disk_raw_frames = 1
        raw_frames_per_slice = 1
        # logAverageFactor applies regardless of whether a stack is being
        # acquired: each saved frame still represents that many averaged
        # raw scans, so the true physical time per (flat) T-step scales
        # with it even though the file structure itself is unaffected.
        try:
            log_average_factor = int(framedata.get("SI.hScan2D.logAverageFactor") or 1)
            if log_average_factor < 1:
                log_average_factor = 1
        except (TypeError, ValueError):
            log_average_factor = 1
        physical_frames_per_volume = log_average_factor
    else:
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
            n_slices = int(n_slices)
            frames_per_volume = int(frames_per_volume)
            physical_frames_per_volume = int(frames_per_volume_flyback or frames_per_volume)
        except (TypeError, ValueError):
            return _flat_dims(
                "could not parse hStackManager slice-count fields as integers"
            )
        if not (0 < frames_per_volume <= physical_frames_per_volume):
            return _flat_dims("inconsistent frames-per-volume metadata")
        has_flyback = physical_frames_per_volume > frames_per_volume

        try:
            raw_frames_per_slice = int(framedata.get("SI.hStackManager.framesPerSlice"))
            if raw_frames_per_slice < 1:
                raw_frames_per_slice = 1
        except (TypeError, ValueError):
            raw_frames_per_slice = 1

        if n_slices * raw_frames_per_slice != frames_per_volume:
            return _flat_dims(
                f"actualNumSlices ({n_slices}) x framesPerSlice "
                f"({raw_frames_per_slice}) does not match numFramesPerVolume "
                f"({frames_per_volume}); falling back to a flat interpretation"
            )

        # SI.hScan2D.logAverageFactor on-the-fly averages that many raw
        # scanner frames into a single saved frame at each Z-position
        # before writing to disk, so what's actually on disk uses
        # raw_frames_per_slice / logAverageFactor frames per Z-position -
        # not the raw framesPerSlice value the rest of hStackManager's
        # metadata (and `SI.hStackManager.zs`) is expressed in terms of.
        try:
            log_average_factor = int(framedata.get("SI.hScan2D.logAverageFactor") or 1)
            if log_average_factor < 1:
                log_average_factor = 1
        except (TypeError, ValueError):
            log_average_factor = 1

        if raw_frames_per_slice % log_average_factor:
            return _flat_dims(
                f"framesPerSlice ({raw_frames_per_slice}) is not evenly "
                f"divisible by logAverageFactor ({log_average_factor}); "
                "falling back to a flat interpretation"
            )
        frames_per_slice = raw_frames_per_slice // log_average_factor

        z_count = n_slices
        kept_raw_frames = z_count * frames_per_slice
        on_disk_raw_frames = kept_raw_frames + (1 if has_flyback else 0)

    pages_per_step = on_disk_raw_frames * n_channels
    if pages_per_step <= 0:
        return _flat_dims("could not determine a positive pages-per-timepoint count")

    # 'F' (repeated frames, a second time-like axis) is grouped with 'T'
    # rather than 'Z': napari's 3D rendering mode treats the *last* 3 axes
    # of the array as the displayed volume, so 'Z' - the only genuinely
    # spatial non-displayed axis - must stay immediately adjacent to 'Y,X'
    # (channel_axis removes 'C' from each per-channel layer entirely, so
    # its position relative to 'Z' doesn't matter here).
    axes = "T" + "".join(
        letter
        for letter, size in (("F", frames_per_slice), ("Z", z_count), ("C", n_channels))
        if size > 1
    ) + "YX"

    return ScanImageDims(
        axes=axes,
        pages_per_step=pages_per_step,
        on_disk_raw_frames=on_disk_raw_frames,
        kept_raw_frames=kept_raw_frames,
        z_count=z_count,
        frames_per_slice=frames_per_slice,
        n_channels=n_channels,
        log_average_factor=log_average_factor,
        raw_frames_per_slice=raw_frames_per_slice,
        physical_frames_per_volume=physical_frames_per_volume,
    )


def compute_scanimage_dimensions(
    framedata: dict[str, Any], total_pages: int
) -> ScanImageDims:
    """Decide how to reshape a ScanImage page stack into T[,Z][,F][,C],Y,X.

    Computes the acquisition's structure via `_compute_scanimage_structure`
    (see there for the reshaping rules), then checks that `total_pages`
    covers at least one full timepoint of that structure; falls back to a
    flat interpretation (with a warning) if it doesn't. A page count that
    covers at least one full timepoint but is not an exact multiple of the
    expected pages-per-timepoint is interpreted as an acquisition that
    stopped mid-timepoint; `build_scanimage_layerdata` drops that trailing
    incomplete timepoint and preserves the confirmed structure.
    """
    if total_pages <= 0:
        return _flat_dims("no pages to interpret")

    dims = _compute_scanimage_structure(framedata)
    if dims.warning:
        return dims

    if total_pages < dims.pages_per_step:
        return _flat_dims(
            f"total page count ({total_pages}) does not contain one complete "
            f"timepoint ({dims.pages_per_step} page(s) = {dims.on_disk_raw_frames} "
            f"raw frame(s) x {dims.n_channels} channel(s)); "
            "falling back to a flat interpretation"
        )
    return dims


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


def _scanimage_structural_signature(tif: TiffFile) -> tuple[Any, ...]:
    """Return the subset of a ScanImage file's properties that matter for concatenation.

    Two files can be safely stitched together if (and only if) they agree
    on this signature - the shape one timepoint implies
    (`_compute_scanimage_structure`) plus the raw page geometry
    (`build_scanimage_layerdata` reshapes every file's pages using the
    *first* file's structure, so a Y/X or dtype mismatch would silently
    corrupt the result). Deliberately *not* full FrameData equality:
    many fields legitimately drift between files of the very same real
    acquisition without indicating any structural difference - e.g. a
    resonant scanner's actually-measured frequency (and everything
    derived from it: line/frame period, frame/volume rate, the per-line
    pixel mask), or per-file PMT/detector offset recalibration - and
    chasing those down field-by-field as they're discovered doesn't
    scale. Raises if the file can't be interpreted as ScanImage at all.
    """
    if not tif.is_scanimage:
        raise ValueError("not a ScanImage file")
    framedata = get_scanimage_framedata(tif)
    structure = _compute_scanimage_structure(framedata)
    if structure.warning:
        raise ValueError(structure.warning)
    page0 = tif.pages[0]
    return (
        structure.axes,
        structure.pages_per_step,
        structure.z_count,
        structure.frames_per_slice,
        structure.n_channels,
        structure.log_average_factor,
        page0.shape,
        str(page0.dtype),
    )


def filter_compatible_scanimage_files(paths: Sequence[str]) -> list[str]:
    """Return the subset of `paths` structurally compatible for concatenation.

    The first path that yields a valid structural signature (see
    `_scanimage_structural_signature`) becomes the reference; every other
    path is kept only if its own signature matches. Non-ScanImage,
    unreadable, or structurally-uninterpretable paths, and paths whose
    signature differs from the reference, are dropped (with a warning)
    rather than risking a wrong reshape; order is preserved for whatever
    remains. Used both for filename-based sibling auto-discovery
    (`find_scanimage_series_files`) and for an explicit multi-file
    selection (`napari_tiff_reader.scanimage_reader_function`).
    """
    reference_signature = None
    confirmed = []
    for candidate in paths:
        try:
            with TiffFile(candidate) as tif:
                signature = _scanimage_structural_signature(tif)
        except Exception as exc:
            log_warning(
                f"{candidate!r} could not be matched to the rest of this "
                f"selection ({exc}); excluding it from the combined acquisition"
            )
            continue

        if reference_signature is None:
            reference_signature = signature
            confirmed.append(candidate)
        elif signature == reference_signature:
            confirmed.append(candidate)
        else:
            log_warning(
                f"{candidate!r} has a different structure (planes, channels, "
                "frame-averaging, or frame size/dtype) than the rest of this "
                "selection; excluding it from the combined acquisition"
            )

    return confirmed


def find_scanimage_series_files(path: str) -> list[str]:
    """Find and order sibling files belonging to the same ScanImage acquisition as `path`.

    Returns just `[path]` if it uses the single-file naming form, or if no
    siblings pass the static-metadata check in `_filter_compatible_scanimage_files`.
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
    confirmed = filter_compatible_scanimage_files(siblings)

    if not confirmed:
        return [str(path)]

    warn_on_frame_number_gaps(confirmed)
    return confirmed


def _lazy_flat_page_array(path: str) -> tuple[da.Array, tuple[int, ...]]:
    """Return a lazy, page-indexed ``(n_pages, Y, X)`` dask array for `path`.

    Built from a single, manually-shaped `TiffPageSeries` covering all raw
    pages - not `tif.series[0]`, whose built-in ScanImage series-shape
    inference derives its "Z" axis purely from `SI.hStackManager.
    framesPerSlice` and the total page count, which disagrees with the true
    on-disk layout once channels are involved (`compute_scanimage_dimensions`
    implements the correct interpretation instead, so this always starts
    from a flat, explicitly-(re)shaped view of the raw pages). Constructing
    one `TiffPageSeries` (and so one underlying Zarr store/file cache) for
    the whole file - rather than one `page.aszarr()` store per page - keeps
    per-file overhead constant instead of growing with page count, which
    matters for large acquisitions (thousands of pages): each store/file
    cache is itself a nontrivial Python object, and building `n_pages` of
    them just to immediately `da.stack` them back together is pure waste.
    Chunking is unaffected either way - a Zarr store built from a
    multi-page series still exposes one chunk per page, so lazy per-frame
    access during scrubbing is identical.
    """
    with TiffFile(path) as tif:
        pages = list(tif.pages)
        page_shape = pages[0].shape
        dtype = pages[0].dtype
        series = TiffPageSeries(
            pages, shape=(len(pages),) + page_shape, dtype=dtype, axes="IYX"
        )
        store = series.aszarr()
    array = da.from_zarr(zarr.open(store, mode="r"))
    return array, (len(pages),) + page_shape


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
        # first split into raw frame-scans x channels, and drop the
        # trailing flyback raw frame(s) appended once at the end of the
        # volume's real data (not per Z-position - see ScanImageDims)
        by_raw_frame = array.reshape((n_steps, dims.on_disk_raw_frames, dims.n_channels, y, x))
        kept = by_raw_frame[:, : dims.kept_raw_frames]
        # the on-disk order within the kept raw frames is Z-outer,
        # frame-repeat-inner; split that out first, then transpose so the
        # *exposed* array is frame-repeat-major, Z-minor (matching `axes`
        # above, with Z kept adjacent to Y,X)
        kept = kept.reshape((n_steps, dims.z_count, dims.frames_per_slice, dims.n_channels, y, x))
        kept = kept.transpose(0, 2, 1, 3, 4, 5)

        # squeeze out whichever of F/Z/C are trivial (size 1), matching
        # `dims.axes`; squeezing never reorders data, so this is safe
        # regardless of which combination of axes is present.
        kept_shape = [n_steps]
        kept_shape += [
            size for size in (dims.frames_per_slice, dims.z_count, dims.n_channels) if size > 1
        ]
        kept_shape += [y, x]
        kept = kept.reshape(tuple(kept_shape))
        volumes.append(kept)

    combined = da.concatenate(volumes, axis=0)
    return combined, dims.axes
