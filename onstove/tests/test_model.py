# Test for models.py
import os.path
import shutil
import pytest
from OnStove.onstove.model import DataProcessor, MCA, OnStove
from OnStove.onstove.layer import VectorLayer, RasterLayer


@pytest.fixture
def sample_vector_layer():
    # set path
    vect_path = r"data/sample_mask.geojson"
    # Create a VectorLayer object with sample data
    vect = VectorLayer(path=vect_path)
    return vect


@pytest.fixture
def sample_raster_layer():
    # set path
    rast_path = r"data/sample_raster.tif"
    # Create RasterLayer object
    raster_file = RasterLayer(path=rast_path)
    return raster_file

@pytest.fixture
def model_object():
    model = OnStove(project_crs=3857)
    return model

@pytest.fixture
def data_object():
    data = DataProcessor(project_crs=3857, cell_size=(1000, 1000))
    return data


# tests for DataProcessor
def test_model(model_object):
    # test if model is instance OnStove
    assert isinstance(model_object, OnStove)


def test_get_layers():
    pass


def test_add_layer(model_object):
    path = r"data/sample_raster3.tif"
    model_object.add_layer(
        category='Demographics',
        name='Population',
        path=path,
        layer_type='raster',
        base_layer=True
    )
    assert len(model_object.layers.items()) > 0


def test_add_mask_layer(data_object):
    path = r"data/sample_mask.geojson"
    data_object.add_mask_layer(
        path=path
    )
    assert isinstance(data_object, DataProcessor)


def test_mask_layers(data_object, sample_raster_layer):
    path = r"data/sample_mask.geojson"
    data_object.add_mask_layer(
        category='Administrative',
        name='County_boundaries',
        path=path,
    )
    data_object.mask_layers(
        "all",
        True,
        True
    )
    assert isinstance(data_object, DataProcessor)


def test_get_distance_rasters(data_object):
    data_object.get_distance_rasters(
        "all",
        False
    )
    assert isinstance(data_object, DataProcessor)


def test_to_pickle(model_object):
    name = "model_object.pkl"
    model_object.to_pickle(name=name)
    assert os.path.exists(name)
