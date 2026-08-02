import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffFile

from base_data import (
    SCANIMAGE_SOFTWARE_SINGLE,
    scanimage_split_files,
    scanimage_timeseries_tiff,
    scanimage_volumetric_tiff,
    write_scanimage_tiff,
)
from napari_tiff.napari_tiff_reader import napari_get_reader, scanimage_reader_function
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
    assert dims.frames_per_group == 1
    assert dims.frames_to_keep == 1
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
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=8)
    assert dims.axes == "TZYX"
    assert dims.frames_per_group == 4
    assert dims.frames_to_keep == 3
    assert dims.n_slices == 3
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


def test_compute_scanimage_dimensions_multichannel_falls_back():
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hChannels.channelSave": [1, 2],
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=8)
    assert dims.axes == "IYX"
    assert dims.n_channels == 2
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
