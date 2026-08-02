import numpy as np
import tifffile
from numpy.testing import assert_array_equal

from napari_tiff.napari_tiff_reader import multifile_reader_function, napari_get_reader
from napari_tiff.napari_tiff_multifile import (
    check_compatible,
    infer_join_axis,
    natural_sort,
)


def _write_plain_tiff(path, data, **kwargs):
    tifffile.imwrite(path, data, imagej=False, **kwargs)
    return str(path)


def test_natural_sort_orders_numbers_correctly():
    files = ["img10.tif", "img2.tif", "img1.tif"]
    assert natural_sort(files) == ["img1.tif", "img2.tif", "img10.tif"]


def test_check_compatible_matching_files(tmp_path):
    data = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    paths = [
        _write_plain_tiff(tmp_path / "a.tif", data),
        _write_plain_tiff(tmp_path / "b.tif", data),
    ]
    report = check_compatible(paths)
    assert report.compatible


def test_check_compatible_mismatched_shape(tmp_path):
    paths = [
        _write_plain_tiff(tmp_path / "a.tif", np.zeros((5, 5), dtype=np.uint8)),
        _write_plain_tiff(tmp_path / "b.tif", np.zeros((6, 6), dtype=np.uint8)),
    ]
    report = check_compatible(paths)
    assert not report.compatible
    assert "b.tif" in report.reason


def test_check_compatible_mismatched_dtype(tmp_path):
    paths = [
        _write_plain_tiff(tmp_path / "a.tif", np.zeros((5, 5), dtype=np.uint8)),
        _write_plain_tiff(tmp_path / "b.tif", np.zeros((5, 5), dtype=np.uint16)),
    ]
    report = check_compatible(paths)
    assert not report.compatible


def test_infer_join_axis_explicit_time_token():
    paths = ["scan_t001.tif", "scan_t002.tif", "scan_t003.tif"]
    guess = infer_join_axis(paths)
    assert guess.label == "T"
    assert guess.confidence == "high"


def test_infer_join_axis_explicit_z_token():
    paths = ["stack_z01.tif", "stack_z02.tif"]
    guess = infer_join_axis(paths)
    assert guess.label == "Z"
    assert guess.confidence == "high"


def test_infer_join_axis_bare_counter_is_not_assumed_to_be_time():
    paths = ["img_00001.tif", "img_00002.tif", "img_00003.tif"]
    guess = infer_join_axis(paths)
    assert guess.label == "I"
    assert guess.confidence == "low"


def test_infer_join_axis_unrelated_names():
    paths = ["cortex.tif", "hippocampus.tif"]
    guess = infer_join_axis(paths)
    assert guess.label == "I"
    assert guess.confidence == "low"


def test_reader_selected_for_multiple_plain_tiffs(tmp_path):
    data = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    paths = [
        _write_plain_tiff(tmp_path / "a.tif", data),
        _write_plain_tiff(tmp_path / "b.tif", data),
    ]
    assert napari_get_reader(paths) is multifile_reader_function


def test_multifile_reader_combines_compatible_plain_images(tmp_path):
    data_a = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    data_b = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    paths = [
        _write_plain_tiff(tmp_path / "img_00001.tif", data_a),
        _write_plain_tiff(tmp_path / "img_00002.tif", data_b),
    ]
    layer_data_list = multifile_reader_function(paths)
    assert len(layer_data_list) == 1
    result, kwargs, layer_type = layer_data_list[0]
    assert layer_type == "image"
    assert result.shape == (2, 5, 5)
    assert_array_equal(result.compute(), np.stack([data_a, data_b]))
    # a bare numeric counter must not be labeled as a timeseries
    assert kwargs["axis_labels"][0] == "i"
    assert kwargs["metadata"]["multifile_join_axis"]["label"] == "I"


def test_multifile_reader_labels_axis_time_with_explicit_token(tmp_path):
    data_a = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    data_b = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    paths = [
        _write_plain_tiff(tmp_path / "scan_t001.tif", data_a),
        _write_plain_tiff(tmp_path / "scan_t002.tif", data_b),
    ]
    layer_data_list = multifile_reader_function(paths)
    _result, kwargs, _layer_type = layer_data_list[0]
    assert kwargs["axis_labels"][0] == "t"
    assert kwargs["metadata"]["multifile_join_axis"]["label"] == "T"


def test_multifile_reader_falls_back_to_independent_layers_when_incompatible(tmp_path):
    paths = [
        _write_plain_tiff(tmp_path / "a.tif", np.zeros((5, 5), dtype=np.uint8)),
        _write_plain_tiff(tmp_path / "b.tif", np.zeros((6, 6), dtype=np.uint8)),
    ]
    layer_data_list = multifile_reader_function(paths)
    assert len(layer_data_list) == 2


def test_multifile_reader_concatenates_existing_leading_axis(tmp_path):
    """Files that are themselves already multi-frame (e.g. partial
    timeseries chunks) should extend that same axis, not add a new one.
    """
    chunk_a = np.random.randint(0, 255, (3, 5, 5)).astype(np.uint8)
    chunk_b = np.random.randint(0, 255, (4, 5, 5)).astype(np.uint8)
    paths = [
        _write_plain_tiff(tmp_path / "chunk_00001.tif", chunk_a, photometric="minisblack"),
        _write_plain_tiff(tmp_path / "chunk_00002.tif", chunk_b, photometric="minisblack"),
    ]
    layer_data_list = multifile_reader_function(paths)
    result, _kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (7, 5, 5)
    assert_array_equal(result.compute(), np.concatenate([chunk_a, chunk_b], axis=0))
