from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffFile

from base_data import (
    SCANIMAGE_SOFTWARE_SINGLE,
    SCANIMAGE_SOFTWARE_SPLIT,
    SCANIMAGE_SOFTWARE_SPLIT_DRIFT,
    scanimage_flat_two_channel_tiff,
    scanimage_frames_per_slice_multivolume_tiff,
    scanimage_frames_per_slice_single_volume_tiff,
    scanimage_log_average_tiff,
    scanimage_split_files,
    scanimage_timeseries_tiff,
    scanimage_volumetric_tiff,
    scanimage_volumetric_tiff_truncated,
    scanimage_volumetric_two_channel_tiff,
    write_scanimage_tiff,
    write_scanimage_tiff_multi_truncated,
)
from napari_tiff.napari_tiff_reader import (
    directory_reader_function,
    napari_get_reader,
    scanimage_reader_function,
)
from napari_tiff.napari_tiff_metadata import get_scanimage_metadata
from napari_tiff.napari_tiff_scanimage import (
    compute_scanimage_dimensions,
    filter_compatible_scanimage_files,
    find_scanimage_series_files,
    get_scanimage_framedata,
)


def test_compute_scanimage_dimensions_flat_timeseries():
    """Plain timeseries: not volumetric, one frame per timepoint."""
    framedata = {"SI.hStackManager.enable": False, "SI.hFastZ.enable": False}
    dims = compute_scanimage_dimensions(framedata, total_pages=6)
    assert dims.axes == "TYX"
    assert dims.pages_per_step == 1
    assert dims.on_disk_raw_frames == 1
    assert dims.kept_raw_frames == 1
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
    assert dims.on_disk_raw_frames == 4
    assert dims.kept_raw_frames == 3
    assert dims.z_count == 3
    assert dims.warning is None


def test_compute_scanimage_dimensions_trailing_incomplete_timepoint_still_reshapes():
    """A page count that isn't an exact multiple of the expected
    pages-per-timepoint, but does cover at least one full timepoint, is a
    real, legitimate case: an acquisition stopped mid-timepoint (e.g. the
    user aborted recording). The confirmed structure must be kept -
    `build_scanimage_layerdata` is responsible for dropping the trailing
    partial timepoint, not `compute_scanimage_dimensions` falling back to
    an unstructured flat interpretation.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=10)  # 2 full volumes + 2 extra pages
    assert dims.axes == "TZYX"
    assert dims.pages_per_step == 4
    assert dims.warning is None


def test_compute_scanimage_dimensions_too_few_pages_for_one_timepoint_falls_back():
    """Fewer pages than a single timepoint needs is a genuine failure - not
    a trailing-partial-timepoint case - and must still fall back.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=2)  # < 4
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
    assert dims.on_disk_raw_frames == 4
    assert dims.kept_raw_frames == 3
    assert dims.z_count == 3
    assert dims.n_channels == 2
    assert dims.warning is None


def test_compute_scanimage_dimensions_inconsistent_frames_per_slice_falls_back():
    """actualNumSlices x framesPerSlice must equal numFramesPerVolume (a
    real ScanImage invariant); if it doesn't, the metadata is inconsistent
    and reshaping must degrade safely rather than guess.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 3,
        "SI.hStackManager.numFramesPerVolume": 3,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
        "SI.hStackManager.framesPerSlice": 20,  # 3 x 20 != 3
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=8)
    assert dims.axes == "IYX"
    assert dims.warning is not None


def test_compute_scanimage_dimensions_single_volume_frames_per_slice():
    """framesPerSlice > 1 for a single volume (no flyback): repeated
    frames nest inside Z, producing a `'F'` axis.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 7,
        "SI.hStackManager.numFramesPerVolume": 140,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 140,
        "SI.hStackManager.framesPerSlice": 20,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=140)
    assert dims.axes == "TFZYX"
    assert dims.on_disk_raw_frames == 140
    assert dims.kept_raw_frames == 140
    assert dims.z_count == 7
    assert dims.frames_per_slice == 20
    assert dims.pages_per_step == 140
    assert dims.warning is None


def test_compute_scanimage_dimensions_multivolume_frames_per_slice():
    """framesPerSlice > 1 combined with a flyback (multiple volumes) is
    now supported: the flyback is always a single trailing raw frame
    appended once per volume, regardless of framesPerSlice.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 7,
        "SI.hStackManager.numFramesPerVolume": 140,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 141,
        "SI.hStackManager.framesPerSlice": 20,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=1410)  # 10 volumes
    assert dims.axes == "TFZYX"
    assert dims.on_disk_raw_frames == 141
    assert dims.kept_raw_frames == 140
    assert dims.z_count == 7
    assert dims.frames_per_slice == 20
    assert dims.pages_per_step == 141
    assert dims.warning is None


def test_compute_scanimage_dimensions_log_average_factor_shrinks_on_disk_frames():
    """SI.hScan2D.logAverageFactor > 1 on-the-fly averages that many raw
    scanner frames into a single saved frame per Z-position, so what's on
    disk uses framesPerSlice / logAverageFactor frames per Z-position -
    not the raw framesPerSlice value the rest of hStackManager's metadata
    is expressed in terms of (reported bug: a real file with
    framesPerSlice=logAverageFactor collapsed to 1 on-disk frame per
    Z-position was misdiagnosed as a metadata inconsistency).
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 40,
        "SI.hStackManager.numFramesPerVolume": 16000,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 16000,
        "SI.hStackManager.framesPerSlice": 400,
        "SI.hChannels.channelSave": [1, 2],
        "SI.hScan2D.logAverageFactor": 400,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=80)  # 40 slices x 2 channels
    assert dims.axes == "TZCYX"
    assert dims.frames_per_slice == 1
    assert dims.z_count == 40
    assert dims.on_disk_raw_frames == 40
    assert dims.kept_raw_frames == 40
    assert dims.pages_per_step == 80
    assert dims.log_average_factor == 400
    assert dims.raw_frames_per_slice == 400
    assert dims.physical_frames_per_volume == 16000
    assert dims.warning is None


def test_compute_scanimage_dimensions_log_average_factor_keeps_f_axis_and_flyback():
    """framesPerSlice / logAverageFactor > 1 still produces an `'F'` axis,
    and a flyback frame (raw metadata says numFramesPerVolumeWithFlyback
    > numFramesPerVolume) is still added as a single extra on-disk page,
    unaffected by averaging.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 7,
        "SI.hStackManager.numFramesPerVolume": 140,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 141,
        "SI.hStackManager.framesPerSlice": 20,
        "SI.hScan2D.logAverageFactor": 4,
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=36)
    assert dims.axes == "TFZYX"
    assert dims.frames_per_slice == 5  # 20 / 4
    assert dims.z_count == 7
    assert dims.kept_raw_frames == 35  # 7 x 5
    assert dims.on_disk_raw_frames == 36  # + 1 flyback
    assert dims.pages_per_step == 36
    assert dims.log_average_factor == 4
    assert dims.raw_frames_per_slice == 20
    assert dims.physical_frames_per_volume == 141
    assert dims.warning is None


def test_compute_scanimage_dimensions_log_average_factor_must_divide_evenly():
    """If framesPerSlice isn't an exact multiple of logAverageFactor, the
    averaging model doesn't apply cleanly - fall back rather than guess.
    """
    framedata = {
        "SI.hStackManager.enable": True,
        "SI.hStackManager.actualNumSlices": 7,
        "SI.hStackManager.numFramesPerVolume": 140,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 141,
        "SI.hStackManager.framesPerSlice": 20,
        "SI.hScan2D.logAverageFactor": 3,  # 20 % 3 != 0
    }
    dims = compute_scanimage_dimensions(framedata, total_pages=141)
    assert dims.axes == "IYX"
    assert dims.warning is not None


def test_reader_log_average_factor_reshapes_correctly(scanimage_log_average_tiff):
    """Reader-level regression test for the reported bug."""
    path, data = scanimage_log_average_tiff  # 2 volumes, 72 pages, 36/volume
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (2, 5, 7, 4, 4)
    assert kwargs["axis_labels"] == ("t", "f", "z", "y", "x")
    # on disk: Z-outer, frame-repeat-inner within the kept raw frames
    expected = data.reshape(2, 36, 4, 4)[:, :35].reshape(2, 7, 5, 4, 4).transpose(0, 2, 1, 3, 4)
    assert_array_equal(result.compute(), expected)


def test_scale_uses_physical_not_on_disk_frame_counts(scanimage_log_average_tiff):
    """T (and F) axis scale must reflect the true physical acquisition
    time - which on-the-fly averaging shrinks on disk but does not
    actually speed up - not the reduced on-disk frame counts.
    """
    path, _data = scanimage_log_average_tiff
    with TiffFile(path) as tif:
        kwargs = get_scanimage_metadata(tif)
    frame_rate = 30.0021  # SI.hRoiManager.scanFrameRate in software_log_average.txt
    t_index = kwargs["axis_labels"].index("t")
    f_index = kwargs["axis_labels"].index("f")
    z_index = kwargs["axis_labels"].index("z")
    assert kwargs["scale"][t_index] == pytest.approx(141 / frame_rate)
    assert kwargs["scale"][f_index] == pytest.approx(4 / frame_rate)
    assert kwargs["scale"][z_index] == pytest.approx(5.0)


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


def test_reader_drops_trailing_incomplete_volume(scanimage_volumetric_tiff_truncated):
    """Regression test for a real-world edge case: an acquisition stopped
    mid-volume (10 pages on disk = 2 complete 4-page volumes + 2 leftover
    pages of a 3rd, incomplete one). The confirmed volumetric structure
    must be kept and the trailing partial volume simply dropped - not
    discarded entirely into an unstructured flat interpretation.
    """
    path, data = scanimage_volumetric_tiff_truncated
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _ = layer_data_list[0]
    assert result.shape == (2, 3, 4, 4)
    expected = data[:8].reshape(2, 4, 4, 4)[:, :3]
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
    but log a warning about the frame-number gap: frame 8 is followed by
    17, neither continuing (9) nor restarting a new acquisition (1), so
    this really does look like a missing chunk of the same acquisition.
    """
    paths, datas = scanimage_split_files
    with pytest.warns(UserWarning, match="missing a chunk"):
        layer_data_list = scanimage_reader_function([paths[0], paths[2]])
    result, _kwargs, _ = layer_data_list[0]
    expected = np.concatenate(
        [datas[0].reshape(2, 4, 4, 4)[:, :3], datas[2].reshape(2, 4, 4, 4)[:, :3]],
        axis=0,
    )
    assert_array_equal(result.compute(), expected)


def test_reader_explicit_list_with_truncated_file_still_concatenates(tmp_path):
    """A file that stopped mid-volume must not be excluded outright - only
    its own trailing incomplete volume is dropped (same as the single-file
    case), so it can still be concatenated with complete files from the
    same acquisition, regardless of its position in the list.
    """
    path1 = tmp_path / "acq_00003_00001.tif"
    path2 = tmp_path / "acq_00003_00002.tif"
    _p1, data1 = write_scanimage_tiff(
        path1, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=1
    )
    # 2 complete 4-page volumes + 2 leftover pages of a 3rd, incomplete one
    _p2, data2 = write_scanimage_tiff_multi_truncated(
        path2,
        SCANIMAGE_SOFTWARE_SPLIT,
        n_complete_steps=2,
        extra_pages=2,
        z_group_size=4,
        n_channels=1,
    )
    # the truncated file's frame numbers restart at 1 - treated as a new,
    # separate acquisition rather than a gap, so no warning is expected
    layer_data_list = scanimage_reader_function([str(path1), str(path2)])
    assert len(layer_data_list) == 1
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (4, 3, 4, 4)  # 2 complete volumes from each file
    expected = np.concatenate(
        [
            data1.reshape(2, 4, 4, 4)[:, :3],
            data2[:8].reshape(2, 4, 4, 4)[:, :3],
        ],
        axis=0,
    )
    assert_array_equal(result.compute(), expected)
    assert kwargs["axis_labels"] == ("t", "z", "y", "x")


def test_reader_explicit_list_excludes_incompatible_file_and_opens_it_independently(
    tmp_path,
):
    """Dropping several files together (napari's stack mode) must sanity-
    check that they're actually compatible (same planes/channels/etc.),
    not blindly reshape every file using the first file's structure. A
    file with genuinely different ScanImage metadata is excluded from the
    combined stack, but still opened as its own independent layer so
    nothing is silently lost.
    """
    path1 = tmp_path / "acq_00004_00001.tif"
    path2 = tmp_path / "acq_00004_00002.tif"
    incompatible = tmp_path / "unrelated_00001.tif"
    _p1, data1 = write_scanimage_tiff(
        path1, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=1
    )
    _p2, data2 = write_scanimage_tiff(
        path2, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=9
    )
    _p3, data3 = write_scanimage_tiff(incompatible, SCANIMAGE_SOFTWARE_SINGLE, n_pages=5)

    layer_data_list = scanimage_reader_function(
        [str(incompatible), str(path2), str(path1)]
    )
    assert len(layer_data_list) == 2  # 1 combined stack + 1 independent layer

    shapes = {ld[0].shape for ld in layer_data_list}
    assert shapes == {(4, 3, 4, 4), (5, 4, 4)}

    combined = next(ld for ld in layer_data_list if ld[0].shape == (4, 3, 4, 4))
    expected_combined = np.concatenate(
        [data1.reshape(2, 4, 4, 4)[:, :3], data2.reshape(2, 4, 4, 4)[:, :3]], axis=0
    )
    assert_array_equal(combined[0].compute(), expected_combined)

    independent = next(ld for ld in layer_data_list if ld[0].shape == (5, 4, 4))
    assert_array_equal(independent[0].compute(), data3)


def test_reader_explicit_list_names_layer_as_stitched(tmp_path):
    """A combined multi-file layer's name must reflect that it was
    stitched together - not just look like the first file's own name,
    which would hide that other files' data was merged in.
    """
    path1 = tmp_path / "acq_00006_00001.tif"
    path2 = tmp_path / "acq_00006_00002.tif"
    write_scanimage_tiff(path1, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=1)
    write_scanimage_tiff(path2, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=9)

    layer_data_list = scanimage_reader_function([str(path1), str(path2)])
    assert len(layer_data_list) == 1
    _result, kwargs, _layer_type = layer_data_list[0]
    assert kwargs["name"] == "acq_00006_00001 (stitched, 2 files)"


def test_reader_single_file_does_not_rename_layer(scanimage_timeseries_tiff):
    """A single file (nothing stitched) keeps napari's default naming
    (derived from the file path) - no `name` override needed or wanted.
    """
    path, _data = scanimage_timeseries_tiff
    _result, kwargs, _layer_type = scanimage_reader_function(path)[0]
    assert "name" not in kwargs


def test_filter_compatible_scanimage_files_ignores_scanner_timing_drift(tmp_path):
    """Regression test for a real-world case: two genuinely compatible
    split-acquisition files whose resonant scanner frequency (and
    everything derived from it - line/frame period, frame/volume rate,
    the per-line pixel mask) and PMT offsets drifted slightly between
    files, as real hardware always does. These fields must not be
    mistaken for a structural incompatibility.
    """
    path1 = tmp_path / "acq_00005_00001.tif"
    path2 = tmp_path / "acq_00005_00002.tif"
    write_scanimage_tiff(path1, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=1)
    write_scanimage_tiff(path2, SCANIMAGE_SOFTWARE_SPLIT_DRIFT, n_pages=8, start_frame_number=9)

    result = filter_compatible_scanimage_files([str(path1), str(path2)])
    assert result == [str(path1), str(path2)]


def test_filter_compatible_scanimage_files_keeps_only_matching_metadata(tmp_path):
    path1 = tmp_path / "a.tif"
    path2 = tmp_path / "b.tif"
    path3 = tmp_path / "c.tif"
    write_scanimage_tiff(path1, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=1)
    write_scanimage_tiff(path2, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=9)
    write_scanimage_tiff(path3, SCANIMAGE_SOFTWARE_SINGLE, n_pages=5)

    result = filter_compatible_scanimage_files([str(path1), str(path2), str(path3)])
    assert result == [str(path1), str(path2)]


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


def test_reader_multivolume_frames_per_slice_reshapes_correctly(
    scanimage_frames_per_slice_multivolume_tiff,
):
    """Regression test for the third reported case: a multi-volume,
    `framesPerSlice=20` acquisition must reshape into `(T, F, Z, Y, X)`
    (F before Z, so Z stays adjacent to Y,X for correct napari 3D
    rendering), correctly dropping the single trailing flyback frame per
    volume (not a full extra Z-position's worth of repeated frames).
    """
    path, data = scanimage_frames_per_slice_multivolume_tiff  # 2 volumes, 282 pages
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (2, 20, 7, 4, 4)
    assert kwargs["axis_labels"] == ("t", "f", "z", "y", "x")
    # on disk: Z-outer, frame-repeat-inner within the kept raw frames
    expected = data.reshape(2, 141, 4, 4)[:, :140].reshape(2, 7, 20, 4, 4).transpose(0, 2, 1, 3, 4)
    assert_array_equal(result.compute(), expected)


def test_reader_multivolume_frames_per_slice_layer_can_be_added_to_viewer(
    scanimage_frames_per_slice_multivolume_tiff,
):
    from napari.components import ViewerModel

    path, _data = scanimage_frames_per_slice_multivolume_tiff
    result, kwargs, _layer_type = scanimage_reader_function(path)[0]
    viewer = ViewerModel()
    viewer.add_image(result, **kwargs)
    assert len(viewer.layers) == 1
    assert viewer.layers[0].ndim == 5


def test_reader_single_volume_frames_per_slice_reshapes_correctly(
    scanimage_frames_per_slice_single_volume_tiff,
):
    """Regression test for the second reported case: a single-volume,
    `framesPerSlice=20` acquisition (7 Z-positions, 20 repeated frames
    each, no flyback) must reshape into `(T, F, Z, Y, X)` (not `TZFYX`:
    Z must stay adjacent to Y,X so napari's 3D mode renders true depth
    instead of the frame-repeat axis - see the reported napari-3D-mode
    bug), not fall back.
    """
    path, data = scanimage_frames_per_slice_single_volume_tiff  # 140 pages
    layer_data_list = scanimage_reader_function(path)
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (1, 20, 7, 4, 4)
    assert kwargs["axis_labels"] == ("t", "f", "z", "y", "x")
    # on disk: Z-outer, frame-repeat-inner; exposed array is transposed to F-major, Z-minor
    expected = data.reshape(1, 7, 20, 4, 4).transpose(0, 2, 1, 3, 4)
    assert_array_equal(result.compute(), expected)


def test_z_spacing_correct_with_frames_per_slice_single_volume(
    scanimage_frames_per_slice_single_volume_tiff,
):
    """Regression test: `SI.hStackManager.zs` repeats each Z-position's
    value `framesPerSlice` times in a row (Z-outer, frame-repeat-inner),
    so naively differencing consecutive entries wildly underestimates the
    real step size (mostly-zero diffs between repeats dilute the real
    steps). Must sample every `framesPerSlice`-th entry instead.
    """
    path, _data = scanimage_frames_per_slice_single_volume_tiff
    with TiffFile(path) as tif:
        kwargs = get_scanimage_metadata(tif)
    z_index = kwargs["axis_labels"].index("z")
    assert kwargs["scale"][z_index] == pytest.approx(5.0)


def test_z_spacing_correct_with_frames_per_slice_multivolume(
    scanimage_frames_per_slice_multivolume_tiff,
):
    path, _data = scanimage_frames_per_slice_multivolume_tiff
    with TiffFile(path) as tif:
        kwargs = get_scanimage_metadata(tif)
    z_index = kwargs["axis_labels"].index("z")
    assert kwargs["scale"][z_index] == pytest.approx(5.0)


def test_reader_single_volume_frames_per_slice_layer_can_be_added_to_viewer(
    scanimage_frames_per_slice_single_volume_tiff,
):
    from napari.components import ViewerModel

    path, _data = scanimage_frames_per_slice_single_volume_tiff
    result, kwargs, _layer_type = scanimage_reader_function(path)[0]
    viewer = ViewerModel()
    viewer.add_image(result, **kwargs)


def test_reader_handles_large_page_count_efficiently(tmp_path):
    """Regression test: `_lazy_flat_page_array` must build one shared Zarr
    store for the whole file (via a manually-shaped `TiffPageSeries`), not
    one store per page - the latter makes both file-opening and per-frame
    scrubbing scale badly with page count (confirmed separately: ~4x slower
    file-opening and ~10x slower per-frame fetches at 10,000 pages).
    """
    path = tmp_path / "scanimage_large_00001.tif"
    n_pages = 2000
    path, data = write_scanimage_tiff(
        path, SCANIMAGE_SOFTWARE_SINGLE, n_pages=n_pages, shape=(4, 4)
    )
    layer_data_list = scanimage_reader_function(str(path))
    result, kwargs, _layer_type = layer_data_list[0]
    assert result.shape == (n_pages, 4, 4)
    # spot-check a handful of individual single-page fetches (this is what
    # scrubbing actually does - not a full-array compute)
    for i in (0, 1, n_pages // 2, n_pages - 1):
        assert_array_equal(result[i].compute(), data[i])


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
