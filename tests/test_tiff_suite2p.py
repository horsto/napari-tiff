import numpy as np
from numpy.testing import assert_array_equal

from base_data import suite2p_output_dir, suite2p_output_dir_two_channels, write_suite2p_output
from napari_tiff.napari_tiff_reader import napari_get_reader, suite2p_reader_function
from napari_tiff.napari_tiff_suite2p import (
    build_suite2p_layerdata,
    find_channel_directories,
    find_plane_directories,
    is_suite2p_output_directory,
)


def test_is_suite2p_output_directory_true(suite2p_output_dir):
    directory, _data = suite2p_output_dir
    assert is_suite2p_output_directory(directory)


def test_is_suite2p_output_directory_false_without_marker_file(tmp_path):
    (tmp_path / "plane0").mkdir()
    assert not is_suite2p_output_directory(str(tmp_path))


def test_is_suite2p_output_directory_false_without_plane_dir(tmp_path):
    (tmp_path / "ops.npy").write_bytes(b"")
    assert not is_suite2p_output_directory(str(tmp_path))


def test_find_plane_directories_naturally_sorted(tmp_path):
    for name in ("plane10", "plane2", "plane1"):
        (tmp_path / name).mkdir()
    dirs = find_plane_directories(str(tmp_path))
    assert [d.split("/")[-1] for d in dirs] == ["plane1", "plane2", "plane10"]


def test_find_channel_directories(suite2p_output_dir_two_channels):
    directory, _data = suite2p_output_dir_two_channels
    plane0 = find_plane_directories(directory)[0]
    channels = [c.split("/")[-1] for c in find_channel_directories(plane0)]
    assert channels == ["reg_tif", "reg_tif2"]


def test_reader_selected_for_suite2p_directory(suite2p_output_dir):
    directory, _data = suite2p_output_dir
    assert napari_get_reader(directory) is suite2p_reader_function


def test_reader_not_selected_for_plain_directory_with_plane_name_but_no_marker(tmp_path):
    (tmp_path / "plane0").mkdir()
    assert napari_get_reader(str(tmp_path)) is None


def test_suite2p_reader_stitches_planes_single_channel(suite2p_output_dir):
    directory, data = suite2p_output_dir
    layer_data_list = suite2p_reader_function(directory)
    result, kwargs, layer_type = layer_data_list[0]
    assert layer_type == "image"

    n_planes = len({p for p, _c in data})
    expected_frames = data[(0, "reg_tif")].shape[0]
    assert result.shape == (expected_frames, n_planes, 4, 4)
    assert kwargs["axis_labels"] == ("t", "z", "y", "x")
    assert kwargs["channel_axis"] is None

    for plane in range(n_planes):
        assert_array_equal(result[:, plane].compute(), data[(plane, "reg_tif")])


def test_suite2p_reader_stitches_planes_and_channels(suite2p_output_dir_two_channels):
    directory, data = suite2p_output_dir_two_channels
    layer_data_list = suite2p_reader_function(directory)
    result, kwargs, _layer_type = layer_data_list[0]

    n_planes = len({p for p, _c in data})
    expected_frames = data[(0, "reg_tif")].shape[0]
    assert result.shape == (expected_frames, n_planes, 2, 4, 4)
    assert kwargs["axis_labels"] == ("t", "z", "c", "y", "x")
    assert kwargs["channel_axis"] == 2

    for plane in range(n_planes):
        assert_array_equal(result[:, plane, 0].compute(), data[(plane, "reg_tif")])
        assert_array_equal(result[:, plane, 1].compute(), data[(plane, "reg_tif2")])


def test_build_suite2p_layerdata_truncates_mismatched_plane_lengths(tmp_path):
    """Planes with differing total frame counts must still stack, truncated
    to the shortest, rather than raising.
    """
    directory, data = write_suite2p_output(tmp_path / "suite2p", n_planes=2, n_chunks=2, frames_per_chunk=3)
    # shorten plane1's second chunk to simulate an uneven acquisition
    import tifffile

    short_chunk = np.random.randint(0, 100, (1, 4, 4)).astype(np.int16)
    tifffile.imwrite(
        f"{directory}/plane1/reg_tif/chunk001.tif", short_chunk, photometric="minisblack"
    )

    combined, axes, _reference = build_suite2p_layerdata(directory)
    assert axes == "TZYX"
    assert combined.shape[0] == 4  # 3 (chunk0) + 1 (shortened chunk1) frames for plane1
    assert_array_equal(combined[:, 0].compute(), data[(0, "reg_tif")][:4])


def test_build_suite2p_layerdata_raises_without_planes(tmp_path):
    (tmp_path / "empty").mkdir()
    try:
        build_suite2p_layerdata(str(tmp_path / "empty"))
        assert False, "expected ValueError"
    except ValueError:
        pass
