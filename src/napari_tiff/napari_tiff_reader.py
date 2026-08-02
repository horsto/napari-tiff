"""
This modeul is a napari reader for TIFF image files.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import os
import dask.array as da
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tifffile import TIFF, TiffFile, TiffSequence

from napari_tiff.napari_tiff_metadata import get_metadata
from napari_tiff.napari_tiff_multifile import (
    build_multifile_layerdata,
    check_compatible,
    get_multifile_metadata,
    list_tiff_files_in_directory,
    natural_sort,
)
from napari_tiff.napari_tiff_scanimage import (
    build_scanimage_layerdata,
    compute_scanimage_dimensions,
    find_scanimage_series_files,
    get_scanimage_framedata,
    warn_on_frame_number_gaps,
)
from napari_tiff.napari_tiff_suite2p import (
    build_suite2p_layerdata,
    get_suite2p_metadata,
    is_suite2p_output_directory,
)

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]


def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """Implements napari_get_reader hook specification.

    Dispatches on `path`: a suite2p output folder (see `napari_tiff_suite2p`)
    goes to `suite2p_reader_function`; any other directory goes to
    `directory_reader_function`; a ScanImage file (`TiffFile.is_scanimage`)
    to `scanimage_reader_function`; more than one other TIFF to
    `multifile_reader_function`; otherwise the plain `reader_function`.

    `path` is only a list when napari opens files "as a stack" (`viewer.
    open(paths, stack=True)`, *File > Open Files as Stack...*, or
    Shift-drag - though Shift is not always reliably detected on macOS, so
    the menu action is more dependable). A plain drag calls this hook once
    per file instead. This plugin also accepts a bare directory path
    (declared via `accepts_directories` in napari.yaml), which napari does
    *not* pre-expand for us.

    Parameters
    ----------
    path : str or list of str
        Path to a file or directory, or a list of paths (stack mode only).

    Returns
    -------
    function or None
        A reader function for `path`, or None if the format isn't supported.
    """
    paths = path if isinstance(path, list) else [path]
    first_path = paths[0]

    if len(paths) == 1 and os.path.isdir(first_path):
        if is_suite2p_output_directory(first_path):
            return suite2p_reader_function
        return directory_reader_function if list_tiff_files_in_directory(first_path) else None

    first_path_lower = first_path.lower()
    if first_path_lower.endswith("zip"):
        return zip_reader
    if not any(first_path_lower.endswith(ext) for ext in TIFF.FILE_EXTENSIONS):
        return None

    try:
        with TiffFile(first_path) as tif:
            is_scanimage = tif.is_scanimage
    except Exception:
        is_scanimage = False

    if is_scanimage:
        return scanimage_reader_function
    if len(paths) > 1:
        return multifile_reader_function
    return reader_function


def suite2p_reader_function(path: PathLike) -> List[LayerData]:
    """Return napari LayerData for a suite2p output folder.

    Stitches each plane's registered-movie chunks (`reg_tif`/`reg_tif2`)
    into one `(T, Z[, C], Y, X)` array; see `napari_tiff_suite2p` for
    details.
    """
    directory = str(path[0] if isinstance(path, list) else path)
    data, axes, reference_path = build_suite2p_layerdata(directory)
    metadata_kwargs = get_suite2p_metadata(reference_path, axes)
    return [(data, metadata_kwargs, "image")]


def directory_reader_function(path: PathLike) -> List[LayerData]:
    """Return napari LayerData for a directory dropped/selected directly.

    Lists the directory's TIFF files (non-recursive, naturally sorted) and
    dispatches them exactly as `napari_get_reader` would.
    """
    directory = path[0] if isinstance(path, list) else path
    files = list_tiff_files_in_directory(str(directory))
    if not files:
        raise ValueError(f"no TIFF files found in directory {directory!r}")

    try:
        with TiffFile(files[0]) as tif:
            is_scanimage = tif.is_scanimage
    except Exception:
        is_scanimage = False

    if is_scanimage:
        return scanimage_reader_function(files)
    if len(files) > 1:
        return multifile_reader_function(files)
    return reader_function(files[0])


def reader_function(path: PathLike) -> List[LayerData]:
    """Return a list of LayerData tuples from path or list of paths."""
    # TODO: LSM
    with TiffFile(path) as tif:
        try:
            layerdata = tifffile_reader(tif)
        except Exception as exc:
            # fallback to imagecodecs
            log_warning(f"tifffile: {exc}")
            layerdata = imagecodecs_reader(path)
    return layerdata


def scanimage_reader_function(path: PathLike) -> List[LayerData]:
    """Return napari LayerData for a ScanImage acquisition.

    A single `path` auto-discovers and stitches its split-acquisition
    siblings (see `find_scanimage_series_files`). A list `path` (napari's
    stack mode) is stitched as given instead - any subset, any order,
    naturally sorted by file index. Falls back to `reader_function` on the
    first path if anything above fails.
    """
    try:
        if isinstance(path, list):
            paths = natural_sort([str(p) for p in path])
            warn_on_frame_number_gaps(paths)
        else:
            paths = find_scanimage_series_files(str(path))

        with TiffFile(paths[0]) as tif:
            framedata = get_scanimage_framedata(tif)
            dims = compute_scanimage_dimensions(framedata, len(tif.pages))
            metadata_kwargs = get_metadata(tif)

        data, _axes = build_scanimage_layerdata(paths, dims)
        return [(data, metadata_kwargs, "image")]
    except Exception as exc:
        log_warning(f"scanimage reader: {exc}; falling back to generic tiff reader")
        first = path[0] if isinstance(path, list) else path
        return reader_function(first)


def multifile_reader_function(path: PathLike) -> List[LayerData]:
    """Combine multiple (non-ScanImage) TIFF files into one layer.

    Only called with a list of more than one file (napari's stack mode).
    Files are naturally sorted and checked for compatibility; incompatible
    sets fall back to independent layers via `reader_function`.
    """
    paths = natural_sort([str(p) for p in path]) if isinstance(path, list) else [str(path)]

    report = check_compatible(paths)
    if not report.compatible:
        log_warning(f"multi-file reader: {report.reason}; opening files as independent layers")
        layers = []
        for p in paths:
            layers.extend(reader_function(p))
        return layers

    try:
        data, axes, guess = build_multifile_layerdata(paths)
        metadata_kwargs = get_multifile_metadata(paths[0], axes, guess)
        return [(data, metadata_kwargs, "image")]
    except Exception as exc:
        log_warning(f"multi-file reader: {exc}; opening files as independent layers")
        layers = []
        for p in paths:
            layers.extend(reader_function(p))
        return layers


def zip_reader(path: PathLike) -> List[LayerData]:
    """Return napari LayerData from sequence of TIFF in ZIP file."""
    with TiffSequence(container=path) as ims:
        data = ims.asarray()
    return [(data, {}, "image")]


def tifffile_reader(tif: TiffFile) -> List[LayerData]:
    """Return napari LayerData from image series in TIFF file."""
    nlevels = len(tif.series[0].levels)
    if nlevels > 1:
        import zarr
        store = tif.aszarr(multiscales=True)
        group = zarr.open_group(store=store, mode='r')
        try:
            datasets = group.attrs['ome']['multiscales'][0]['datasets']
        except KeyError:
            datasets = group.attrs['multiscales'][0]['datasets']

        # using group.attrs to get multiscales is recommended by cgohlke
        # default dask chunk is 128MiB, so use 1 MiB, which is more reasonable for visualization
        data = [da.from_zarr(group[path_dict['path']], chunks='1 MiB') for path_dict in datasets]
        # assert array shapes are in descending order for napari multiscale image
        shapes = [arr.shape for arr in data]
        assert shapes == list(reversed(sorted(shapes)))
    else:
        # explicitly use series[0] to get the data
        data = tif.series[0].asarray()

    metadata_kwargs = get_metadata(tif)

    return [(data, metadata_kwargs, "image")]


def imagecodecs_reader(path: PathLike):
    """Return napari LayerData from first page in TIFF file."""
    from imagecodecs import imread

    return [(imread(path), {}, "image")]


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)
