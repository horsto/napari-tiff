"""
This modeul is a napari reader for TIFF image files.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import dask.array as da
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tifffile import TIFF, TiffFile, TiffSequence

from napari_tiff.napari_tiff_metadata import get_metadata
from napari_tiff.napari_tiff_multifile import (
    build_multifile_layerdata,
    check_compatible,
    get_multifile_metadata,
    natural_sort,
)
from napari_tiff.napari_tiff_scanimage import (
    build_scanimage_layerdata,
    compute_scanimage_dimensions,
    find_scanimage_series_files,
    get_scanimage_framedata,
    warn_on_frame_number_gaps,
)

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]


def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """Implements napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    paths = path if isinstance(path, list) else [path]
    first_path = paths[0]
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

    If `path` is a single file belonging to a split (multi-file)
    acquisition, automatically discovers and stitches in all of its
    sibling files. If `path` is already a list, that exact set of files
    (which may be any subset, in any order, of a split acquisition) is
    stitched instead, naturally sorted by file index.
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
    """Return napari LayerData combining multiple (non-ScanImage) TIFF files.

    Files are naturally sorted and checked for structural compatibility
    before being combined; incompatible files are instead opened as
    independent layers.
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
