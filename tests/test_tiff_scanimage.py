from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffFile

from base_data import (
    SCANIMAGE_SOFTWARE_SINGLE,
    scanimage_flat_two_channel_tiff,
    scanimage_frames_per_slice_tiff,
    scanimage_split_files,
    scanimage_timeseries_tiff,
    scanimage_volumetric_tiff,
    scanimage_volumetric_two_channel_tiff,
    write_scanimage_tiff,
)
from napari_tiff.napari_tiff_reader import (
    directory_reader_function,
    napari_get_reader,
    scanimage_reader_function,
)
from napari_tiff.napari_tiff_scanimage import (
    compute_scanimage_dimensions,
    find_scanimage_series_files,
    get_scanimage_framedata,
)


def test_compute_scanimage_dimensions_flat_timeseries():
    """Plain timeseries: not volumetric, one frame per timepoint."""
    framedata = {"SI.hStackManager.enable": False, "SI.hFastZ.enable": False}
    dims = compute_scanimage_dimensions(framedata, total_pages=6)
    assert dims.axes == "TYX"
    assert dims.pages_per_step == 1
    assert dims.z_group_size == 1
    assert dims.z_keep == 1
    assert dims.warning is None


def test_compute_scanimage_dimensions_ignores_stale_slice_count():
    """A disabled stack manager must not trigger volumetric reshaping, even
    if leftover slice-count fields are still present from a previous config.
    """
    framedata = {
        "SI.hStackManager.enable": False,
        "SI.hFastZ.enable": False,
        "SI.hStackManager.numSlices": 11,
        "SI.hStackManager.actualNumSlices": 11,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=10)
    assert dims.axes == "TYX"


def test_compute_scanimage_dimensions_volumetric_with_flyback():
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hFastZ.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
        "SI.hStackManager.framesPerSlice": 1,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=8)
    assert dims.axes == "TZYX"
    assert dims.pages_per_step == 4
    assert dims.z_group_size == 4
    assert dims.z_keep == 3
    assert dims.warning is None


def test_compute_scanimage_dimensions_page_count_mismatch_falls_back():
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=10)  # not divisible by 4
    assert dims.axes == "IYX"
    assert dims.warning is not None


def test_compute_scanimage_dimensions_flat_multichannel():
    """Non-volumetric multi-channel: channels are the fastest-varying axis."""
    framedata = {
        "SI.hStackManager.enable": False,
        "SI.hChannels.channelSave": [1, 2],
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=20)
    assert dims.axes == "TCYX"
    assert dims.pages_per_step == 2
    assert dims.n_channels == 2
    assert dims.warning is None


def test_compute_scanimage_dimensions_volumetric_multichannel():
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
        "SI.hChannels.channelSave": [1, 2],
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=16)  # 2 volumes
    assert dims.axes == "TZCYX"
    assert dims.pages_per_step == 8
    assert dims.z_group_size == 4
    assert dims.z_keep == 3
    assert dims.n_channels == 2
    assert dims.warning is None


def test_compute_scanimage_dimensions_frames_per_slice_falls_back():
    """framesPerSlice > 1 (repeat frames per Z-position) is not yet
    supported for volumetric reshaping and must degrade safely.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
        "SI.hStackManager.framesPerSlice": 20,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=8)
    assert dims.axes == "IYX"
    assert dims.warning is not None


def test_get_scanimage_framedata_fallback_to_software_tag(scanimage_timeseries_tiff):
    """Synthetic fixtures have no BigTIFF metadata header, so FrameData must
    come from the per-page Software tag fallback.
    """
    path, _data = scanimage_timeseries_tiff
    with TiffFile(path) as tif:
        assert not tif.scanimage_metadata
        framedata = get_scanimage_framedata(tif)
    assert framedata["SI.hStackManager.enable"] is False


def test_reader_selected_for_scanimage_file(scanimage_timeseries_tiff):
    path, _data = scanimage_timeseries_tiff
    assert napari_get_reader(path) is scanimage_reader_function


def test_reader_plain_scanimage_timeseries(scanimage_timeseries_tiff):
    path, data = scanimage_timeseries_tiff
    layer_data_list = scanimage_reader_function(path)
    assert len(layer_data_list) == 1
    result, kwargs, layer_type = layer_data_list[0]
    assert layer_type == "image"
    assert result.shape == data.shape
    assert_array_equal(result.compute(), data)
    assert kwargs["axis_labels"] == ("t", "y", "x")


def test_reader_volumetric_scanimage_drops_flyback(scanimage_volumetric_tiff):
    path, data = scanimage_volumetric_tiff
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _ = layer_data_list[0]
    assert result.shape == (2, 3, 4, 4)
    expected = data.reshape(2, 4, 4, 4)[:, :3]
    assert_array_equal(result.compute(), expected)
    assert kwargs["axis_labels"] == ("t", "z", "y", "x")


def test_reader_single_file_autodiscovers_all_siblings(scanimage_split_files):
    """Opening just one file of a split acquisition stitches in all of it."""
    paths, datas = scanimage_split_files
    layer_data_list = scanimage_reader_function(paths[0])
    result, _kwargs, _ = layer_data_list[0]
    assert result.shape == (6, 3, 4, 4)  # 2 volumes/file * 3 files
    expected = np.concatenate([d.reshape(2, 4, 4, 4)[:, :3] for d in datas], axis=0)
    assert_array_equal(result.compute(), expected)


def test_reader_explicit_full_set(scanimage_split_files):
    paths, datas = scanimage_split_files
    layer_data_list = scanimage_reader_function(list(paths))
    result, _kwargs, _ = layer_data_list[0]
    expected = np.concatenate([d.reshape(2, 4, 4, 4)[:, :3] for d in datas], axis=0)
    assert_array_equal(result.compute(), expected)


def test_reader_explicit_noncontiguous_subset_warns(scanimage_split_files):
    """Selecting files 1 and 3 (skipping 2) should still stitch correctly,
    but log a warning about the frame-number gap.
    """
    paths, datas = scanimage_split_files
    with pytest.warns(UserWarning, match="not contiguous"):
        layer_data_list = scanimage_reader_function([paths[0], paths[2]])
    result, _kwargs, _ = layer_data_list[0]
    expected = np.concatenate(
        [datas[0].reshape(2, 4, 4, 4)[:, :3], datas[2].reshape(2, 4, 4, 4)[:, :3]],
        axis=0,
    )
    assert_array_equal(result.compute(), expected)


def test_build_scanimage_layerdata_uses_real_page_count(scanimage_timeseries_tiff):
    """Regression test: earlier versions derived shape from `tif.series[0]`,
    whose own ScanImage inference (from `SI.hStackManager.framesPerSlice`)
    can disagree with the true page count once a file is part of a longer
    acquisition, or has multiple channels. Reading pages directly sidesteps
    this entirely; the resulting array must always match the real page count.
    """
    path, data = scanimage_timeseries_tiff  # 6 real pages on disk
    layer_data_list = scanimage_reader_function(path)
    result, _kwargs, _layer_type = layer_data_list[0]
    assert result.shape == data.shape
    assert_array_equal(result.compute(), data)


def test_reader_flat_two_channel_deinterleaves_correctly(scanimage_flat_two_channel_tiff):
    """Regression test for the reported bug: a 2-channel, non-volumetric
    file must be split into (T, C, Y, X), not misread as a truncated file.
    """
    path, data = scanimage_flat_two_channel_tiff  # 5 steps x 2 channels = 10 pages
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (5, 2, 4, 4)
    # channel_axis is an index into the data array; axis_labels/scale/units
    # describe only the *remaining* (non-channel) dimensions, per napari's
    # convention for channel_axis (see get_tiff_metadata/get_ome_tiff_metadata)
    assert kwargs["channel_axis"] == 1
    assert kwargs["axis_labels"] == ("t", "y", "x")
    assert len(kwargs["scale"]) == 3
    assert len(kwargs["units"]) == 3
    expected = data.reshape(5, 2, 4, 4)
    assert_array_equal(result.compute(), expected)


def test_reader_flat_two_channel_layer_can_be_added_to_viewer(
    scanimage_flat_two_channel_tiff,
):
    """End-to-end regression test: napari itself must accept the returned
    layer kwargs (it splits `data` by `channel_axis` and reuses `scale`/
    `units`/`axis_labels` for each per-channel sub-layer, so a mismatched
    length raises `ValueError: could not broadcast ...` at this point).
    """
    from napari.components import ViewerModel

    path, _data = scanimage_flat_two_channel_tiff
    result, kwargs, _layer_type = scanimage_reader_function(path)[0]
    viewer = ViewerModel()
    viewer.add_image(result, **kwargs)
    assert len(viewer.layers) == 2  # one per channel


def test_reader_volumetric_two_channel_deinterleaves_and_drops_flyback(
    scanimage_volumetric_two_channel_tiff,
):
    path, data = scanimage_volumetric_two_channel_tiff  # 2 volumes, z_group=8, 2ch
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (2, 7, 2, 4, 4)  # flyback (8th Z-position) dropped
    assert kwargs["channel_axis"] == 2
    assert kwargs["axis_labels"] == ("t", "z", "y", "x")
    assert len(kwargs["scale"]) == 4
    assert len(kwargs["units"]) == 4
    expected = data.reshape(2, 8, 2, 4, 4)[:, :7]
    assert_array_equal(result.compute(), expected)


def test_reader_volumetric_two_channel_layer_can_be_added_to_viewer(
    scanimage_volumetric_two_channel_tiff,
):
    from napari.components import ViewerModel

    path, _data = scanimage_volumetric_two_channel_tiff
    result, kwargs, _layer_type = scanimage_reader_function(path)[0]
    viewer = ViewerModel()
    viewer.add_image(result, **kwargs)
    assert len(viewer.layers) == 2


def test_reader_frames_per_slice_falls_back_to_flat(scanimage_frames_per_slice_tiff):
    """framesPerSlice > 1 is not yet supported; must still open (flat), not error."""
    path, data = scanimage_frames_per_slice_tiff
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == data.shape
    assert kwargs["axis_labels"] == ("i", "y", "x")


def test_reader_selected_for_directory_of_scanimage_files(scanimage_split_files):
    paths, _datas = scanimage_split_files
    directory = str(Path(paths[0]).parent)
    assert napari_get_reader(directory) is directory_reader_function


def test_directory_reader_stitches_scanimage_split_files(scanimage_split_files):
    """Dropping the containing folder itself (not the files inside it)
    must resolve to, and stitch, the same ScanImage acquisition.
    """
    paths, datas = scanimage_split_files
    directory = str(Path(paths[0]).parent)
    layer_data_list = directory_reader_function(directory)
    result, _kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (6, 3, 4, 4)
    expected = np.concatenate([d.reshape(2, 4, 4, 4)[:, :3] for d in datas], axis=0)
    assert_array_equal(result.compute(), expected)


def test_find_scanimage_series_files_single_naming_form_has_no_siblings(tmp_path):
    path = tmp_path / "myacquisition_00007.tif"
    write_scanimage_tiff(path, SCANIMAGE_SOFTWARE_SINGLE, n_pages=3)
    result = find_scanimage_series_files(str(path))
    assert result == [str(path)]


def test_find_scanimage_series_files_rejects_mismatched_metadata(tmp_path):
    """A file that looks like a sibling by name, but has different static
    metadata, must be excluded from the combined acquisition.
    """
    from base_data import SCANIMAGE_SOFTWARE_SPLIT

    path1 = tmp_path / "acq_00002_00001.tif"
    path2 = tmp_path / "acq_00002_00002.tif"
    write_scanimage_tiff(path1, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=1)
    # different software content (single-plane config) masquerading as a sibling
    write_scanimage_tiff(path2, SCANIMAGE_SOFTWARE_SINGLE, n_pages=8, start_frame_number=9)

    result = find_scanimage_series_files(str(path1))
    assert result == [str(path1)]
