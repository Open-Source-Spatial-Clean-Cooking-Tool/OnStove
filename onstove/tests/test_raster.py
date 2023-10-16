# tests for raster.py module
import pytest
import os
from OnStove.onstove.layer import VectorLayer, RasterLayer
from OnStove.onstove.raster import (
    align_raster,
    mask_raster,
    normalize,
    reproject_raster,
    merge_rasters,
    resample
)


@pytest.fixture
def sample_vector_layer():
    # set path
    vect_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "vector.geojson"
    )
    # Create a VectorLayer object with sample data
    vect = VectorLayer(path=vect_path)
    return vect


@pytest.fixture
def sample_raster_layer():
    # set path
    rast_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "raster.tif"
    )
    # Create RasterLayer object
    raster_file = RasterLayer(path=rast_path)
    return raster_file


@pytest.fixture
def sample_raster_layer_1():
    # set path
    rast_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "raster1.tif"
    )
    # Create RasterLayer object
    raster_file = RasterLayer(path=rast_path)
    return raster_file


@pytest.fixture
def sample_raster_layer_2():
    # set path
    rast_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "raster2.tif"
    )
    # Create RasterLayer object
    raster_file = RasterLayer(path=rast_path)
    return raster_file


@pytest.fixture
def raster_path():
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "raster.tif"
    )
    return path


@pytest.fixture
def vector_path():
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "vector.geojson"
    )
    return path


@pytest.fixture
def output_path():
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "output"
    )
    return path


def test_align_raster(sample_raster_layer, sample_raster_layer_2):
    destination, out_meta = align_raster(
        raster_1=sample_raster_layer,
        raster_2=sample_raster_layer_2,
        method="nearest",
        compression="DEFLATE"
    )
    assert destination is not None
    assert isinstance(out_meta, dict)
    assert out_meta["crs"] == sample_raster_layer.meta["crs"]
    assert out_meta["transform"] == sample_raster_layer.meta["transform"]
    assert out_meta["dtype"] == sample_raster_layer_2.meta["dtype"]


def test_mask_raster(raster_path, vector_path, output_path):
    path = os.path.join(
        output_path,
        "masked_raster.tif"
    )
    mask_raster(
        raster_path=raster_path,
        mask_layer=vector_path,
        output_file=path,
        nodata=0,
        compression="NONE"
    )
    assert os.path.exists(path)
    # compares file sizes in bytes
    assert os.stat(path).st_size < os.stat(raster_path).st_size
    print(f"\nmasked raster: {os.stat(path).st_size} bytes")
    print(f"\noriginal raster: {os.stat(raster_path).st_size} bytes")


def test_reproject_raster(raster_path):
    raster = reproject_raster(
        raster_path=raster_path,
        dst_crs=3857,
        cell_width=None,
        cell_height=None,
        method="nearest",
        compression="DEFLATE"
    )
    # print("crs:", raster[-1]["crs"])
    assert raster[-1]["crs"] == 3857


def test_merge_rasters(sample_raster_layer_1, sample_raster_layer_2, output_path):
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "*.tif"
    )
    output = os.path.join(
        output_path,
        "merged_raster.tif"
    )
    merge_rasters(
        files_path=path,
        dst_crs=3857,
        outpul_file=output
    )
    assert os.path.exists(output)


def test_resample(sample_raster_layer, raster_path):
    data, transform = resample(
        raster_path=raster_path,
        height=20,
        width=20,
        method="bilinear"
    )
    assert data is not None
    assert transform is not sample_raster_layer.meta["transform"]


def test_normalize(raster_path, output_path):
    output = os.path.join(
        output_path,
        "normalized.tif"
    )
    normalize(
        raster=raster_path,
        output_file=output,
        inverse=False,
        meta=None,
        buffer=True
    )

    assert os.path.exists(output)
    normalized = RasterLayer(output)
    assert normalized.normalization == "MinMax"
