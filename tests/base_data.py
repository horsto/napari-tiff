import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

SCANIMAGE_DATA_DIR = Path(__file__).parent / "data" / "scanimage"
SCANIMAGE_SOFTWARE_SINGLE = (SCANIMAGE_DATA_DIR / "software_single.txt").read_text()
SCANIMAGE_SOFTWARE_SPLIT = (SCANIMAGE_DATA_DIR / "software_split.txt").read_text()
SCANIMAGE_SOFTWARE_2CH = (SCANIMAGE_DATA_DIR / "software_2ch.txt").read_text()
SCANIMAGE_SOFTWARE_VOL_2CH = (SCANIMAGE_DATA_DIR / "software_vol_2ch.txt").read_text()
SCANIMAGE_SOFTWARE_FRAMES_PER_SLICE = (
    SCANIMAGE_DATA_DIR / "software_frames_per_slice.txt"
).read_text()


def example_data_filepath(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_filepath.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=False)
    return example_data_filepath


def example_data_zipped_filepath(tmp_path, original_data):
    example_tiff_filepath = str(tmp_path / "myfile.tif")
    tifffile.imwrite(example_tiff_filepath, original_data, imagej=False)
    example_zipped_filepath = str(tmp_path / "myfile.zip")
    with zipfile.ZipFile(example_zipped_filepath, "w") as myzip:
        myzip.write(example_tiff_filepath)
    os.remove(example_tiff_filepath)  # not needed now the zip file is saved
    return example_zipped_filepath


def example_data_tiff(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_tiff.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=False)
    return tifffile.TiffFile(example_data_filepath)


def example_data_imagej(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_imagej.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=True)
    return tifffile.TiffFile(example_data_filepath)


def example_data_ometiff(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_ometiff.ome.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=False)
    return tifffile.TiffFile(example_data_filepath)


@pytest.fixture
def example_data_shaped_singleton(tmp_path):
    """Example 'shaped' tiff with a leading singleton dimension.

    A plain (non-OME, non-ImageJ) tiff written by tifffile with a recorded shape that carries a
    size-1 leading axis (e.g. ``{"shape": [1, 2, 14, Y, X]}``). 
    Since tifffile 2026.5.2 the series view squeezes this axis (4D) n contrast to ``TiffFile.asarray()``.
    """
    example_data_filepath = str(tmp_path / "test-shaped-singleton.tif")
    data = np.ones((1, 2, 14, 32, 32), dtype=np.float32)
    tifffile.imwrite(example_data_filepath, data)
    return tifffile.TiffFile(example_data_filepath)


@pytest.fixture(scope="session")
def imagej_hyperstack_image(tmp_path_factory):
    """ImageJ hyperstack tiff image.

    Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
    micron^3 to an ImageJ hyperstack formatted TIFF file:
    """
    filename = tmp_path_factory.mktemp("data") / "imagej_hyperstack.tif"

    volume = np.random.randn(6, 57, 256, 256).astype("float32")
    image_labels = [f"{i}" for i in range(volume.shape[0] * volume.shape[1])]
    metadata = {
        "spacing": 3.947368,
        "unit": "um",
        "finterval": 1 / 10,
        "fps": 10.0,
        "axes": "TZYX",
        "Labels": image_labels,
    }
    tifffile.imwrite(
        filename,
        volume,
        imagej=True,
        resolution=(1.0 / 2.6755, 1.0 / 2.6755),
        metadata=metadata,
    )
    return (filename, metadata)


def _scanimage_frame_description(frame_number: int) -> str:
    """Return a minimal, but realistically-formatted, ScanImage per-page description."""
    return (
        f"frameNumbers = {frame_number}\n"
        "acquisitionNumbers = 1\n"
        f"frameNumberAcquisition = {frame_number}\n"
        f"frameTimestamps_sec = {frame_number * 0.0333:.9f}\n"
    )


def write_scanimage_tiff(
    path,
    software: str,
    n_pages: int,
    start_frame_number: int = 1,
    shape=(4, 4),
    dtype=np.int16,
):
    """Write a small, synthetic-but-realistic ScanImage-like BigTIFF file.

    Embeds real ScanImage `Software` tag content (captured from an actual
    acquisition) so `tifffile.is_scanimage`/`scanimage_metadata` parsing
    behaves exactly as on real files. `metadata=None` disables tifffile's
    own "shaped" JSON description, which would otherwise take priority over
    ScanImage detection when building `TiffFile.series`.
    """
    data = np.random.randint(0, 2**14, (n_pages,) + shape).astype(dtype)
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for i in range(n_pages):
            frame_number = start_frame_number + i
            tif.write(
                data[i],
                software=software,
                description=_scanimage_frame_description(frame_number),
                contiguous=False,
                metadata=None,
            )
    return path, data


@pytest.fixture
def scanimage_timeseries_tiff(tmp_path):
    """Plain (non-volumetric) ScanImage timeseries: `hStackManager.enable=False`."""
    path = tmp_path / "scanimage_single_00001.tif"
    path, data = write_scanimage_tiff(path, SCANIMAGE_SOFTWARE_SINGLE, n_pages=6)
    return str(path), data


@pytest.fixture
def scanimage_volumetric_tiff(tmp_path):
    """Volumetric ScanImage timeseries with a flyback frame per volume.

    `actualNumSlices=3`, `numFramesPerVolumeWithFlyback=4`: 2 volumes are
    written as 8 on-disk pages (frames 4 and 8 are the flyback frames to be
    dropped).
    """
    path = tmp_path / "scanimage_vol_00001.tif"
    path, data = write_scanimage_tiff(path, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8)
    return str(path), data


def write_scanimage_tiff_multi(
    path,
    software: str,
    n_steps: int,
    z_group_size: int = 1,
    n_channels: int = 1,
    shape=(4, 4),
    dtype=np.int16,
):
    """Write a synthetic ScanImage-like BigTIFF with T steps x Z-positions x
    channels, channel-minor within each Z-position (matching the confirmed
    real on-disk order). Each page's pixel values encode
    `1000*step + 10*z + channel` so tests can verify correct de-interleaving
    and flyback-dropping independent of random data.
    """
    pages = []  # list of (pixel_value, frame_number)
    frame_number = 1
    for step in range(n_steps):
        for z in range(z_group_size):
            for channel in range(n_channels):
                pages.append((1000 * step + 10 * z + channel, frame_number))
            frame_number += 1

    data = np.array(
        [np.full(shape, value, dtype=dtype) for value, _fn in pages]
    )
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for i, (_value, fn) in enumerate(pages):
            tif.write(
                data[i],
                software=software,
                description=_scanimage_frame_description(fn),
                contiguous=False,
                metadata=None,
            )
    return path, data


@pytest.fixture
def scanimage_flat_two_channel_tiff(tmp_path):
    """Plain (non-volumetric) 2-channel ScanImage timeseries: 5 timepoints."""
    path = tmp_path / "scanimage_flat_2ch_00001.tif"
    path, data = write_scanimage_tiff_multi(
        path, SCANIMAGE_SOFTWARE_2CH, n_steps=5, z_group_size=1, n_channels=2
    )
    return str(path), data


@pytest.fixture
def scanimage_volumetric_two_channel_tiff(tmp_path):
    """Volumetric 2-channel ScanImage timeseries: 2 volumes, real metadata
    (`actualNumSlices=7`, on-disk Z-group-with-flyback=8, channelSave=[1,2]).
    """
    path = tmp_path / "scanimage_vol_2ch_00001.tif"
    path, data = write_scanimage_tiff_multi(
        path, SCANIMAGE_SOFTWARE_VOL_2CH, n_steps=2, z_group_size=8, n_channels=2
    )
    return str(path), data


@pytest.fixture
def scanimage_frames_per_slice_tiff(tmp_path):
    """Volumetric acquisition with `SI.hStackManager.framesPerSlice=20`.

    Not yet supported for reshaping (unverified Z/frame-repeat nesting
    order); used to test the flat-fallback gate.
    """
    path = tmp_path / "scanimage_frames_per_slice_00001.tif"
    path, data = write_scanimage_tiff(
        path, SCANIMAGE_SOFTWARE_FRAMES_PER_SLICE, n_pages=5
    )
    return str(path), data


@pytest.fixture
def scanimage_split_files(tmp_path):
    """A 3-file split volumetric ScanImage acquisition, naturally-ordered.

    Mirrors the real `<base>_<acquisition>_<fileIndex>.tif` naming
    convention: 2 volumes (8 pages) per file, with `frameNumbers`
    contiguous across file boundaries (1-8, 9-16, 17-24).
    """
    paths = []
    datas = []
    for i, start in enumerate((1, 9, 17), start=1):
        path = tmp_path / f"scanimage_split_00001_0000{i}.tif"
        path, data = write_scanimage_tiff(
            path, SCANIMAGE_SOFTWARE_SPLIT, n_pages=8, start_frame_number=start
        )
        paths.append(str(path))
        datas.append(data)
    return paths, datas


@pytest.fixture
def example_data_multiresolution(tmp_path):
    """Example multi-resolution tiff file.

    Write a multi-dimensional, multi-resolution (pyramidal), multi-series OME-TIFF
    file with metadata. Sub-resolution images are written to SubIFDs. Limit
    parallel encoding to 2 threads.

    This example code reproduced from tifffile.py, see:
    https://github.com/cgohlke/tifffile/blob/2b5a5208008594976d4627bcf01355fc08837592/tifffile/tifffile.py#L649-L688
    """
    example_data_filepath = str(tmp_path / "test-pyramid.ome.tif")
    data = np.random.randint(0, 255, (8, 2, 512, 512, 3), "uint8")
    subresolutions = 2  # so 3 resolution levels in total
    pixelsize = 0.29  # micrometer
    with tifffile.TiffWriter(example_data_filepath, bigtiff=True) as tif:
        metadata = {
            "axes": "TCYXS",
            "SignificantBits": 8,
            "TimeIncrement": 0.1,
            "TimeIncrementUnit": "s",
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["Channel 1", "Channel 2"]},
            "Plane": {"PositionX": [0.0] * 16, "PositionXUnit": ["µm"] * 16},
        }
        options = dict(
            photometric="rgb",
            tile=(128, 128),
            compression="jpeg",
            resolutionunit="CENTIMETER",
            maxworkers=2,
        )
        tif.write(
            data,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options,
        )
        # write pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        for level in range(subresolutions):
            mag = 2 ** (level + 1)
            tif.write(
                data[..., ::mag, ::mag, :],
                subfiletype=1,
                resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                **options,
            )
        return tifffile.TiffFile(example_data_filepath)
