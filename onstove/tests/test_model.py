# Test for models.py
import os
import shutil
import geopandas as gpd
import pytest
from onstove.model import DataProcessor, MCA, OnStove
from onstove.layer import VectorLayer, RasterLayer


@pytest.fixture
def sample_vector_layer():
    # set path
    vect_path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Administrative",
        "Country_boundaries",
        "Country_boundaries.geojson")
    # Create a VectorLayer object with sample data
    vect = VectorLayer(path=vect_path)
    return vect


@pytest.fixture
def sample_raster_layer():
    # set path
    rast_path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Urban",
        "Urban.tif"
    )
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


@pytest.fixture
def mca_object():
    mca = MCA(project_crs=3857)
    return mca


# tests for DataProcessor
def test_model(model_object):
    # test if model is instance OnStove
    assert isinstance(model_object, OnStove)


def test_get_layers():
    pass


def test_add_layer(model_object):
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Population",
        "Population.tif"
    )
    model_object.add_layer(
        category='Demographics',
        name='Population',
        path=path,
        layer_type='raster',
        base_layer=True
    )
    assert len(model_object.layers.items()) > 0


def test_add_mask_layer(data_object):
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Administrative",
        "Country_boundaries",
        "Country_boundaries.geojson")
    data_object.add_mask_layer(
        path=path
    )
    assert isinstance(data_object, DataProcessor)


def test_mask_layers(data_object, sample_raster_layer):
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Administrative",
        "Country_boundaries",
        "Country_boundaries.geojson")
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
    assert data_object.layers is not None


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


# MCA

def test_set_demand_index():
    pass


def test_set_supply_index():
    pass


def test_set_clean_cooking_index():
    pass


def test_assistance_need_index():
    pass


# OnStove
def test_read_scenario_data(model_object):# TODO
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "RWA_scenario_file.csv"
    )
    model_object.read_scenario_data(path, delimiter=',')
    assert model_object.specs is not None
    assert isinstance(model_object.specs, dict)


def test_population_to_dataframe(model_object):
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Population",
        "Population.tif"
    )
    model_object.add_layer(
        category="Demographics",
        name="Population",
        path=path,
        layer_type="raster",
        base_layer="True"
    )
    model_object.population_to_dataframe()
    assert model_object.gdf is not None
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)


def test_calibrate_urban_rural_split(model_object):#TODO
    """path = r"data/RWA/Demographics/Population/Population.tif"
    model_object.add_layer(
        category="Demographics",
        name="Population",
        path=path,
        layer_type="raster",
        base_layer="True"
    )
    model_object.population_to_dataframe()"""

    ghs_path = r"data/RWA/Demographics/Urban/Urban.tif"
    #model_object.calibrate_urban_rural_split(ghs_path)
    assert model_object.gdf is not None
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)


def test_extract_wealth_index(model_object):
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Population",
        "Population.tif"
    )
    model_object.add_layer(
        category="Demographics",
        name="Population",
        path=path,
        layer_type="raster",
        base_layer="True"
    )
    model_object.population_to_dataframe()
    # wealth index
    wealth_idx = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Wealth",
        "RWA_relative_wealth_index.csv"
    )
    # extract wealth index and add to gdf
    model_object.extract_wealth_index(wealth_index=wealth_idx, file_type="csv")
    assert model_object.gdf is not None
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)


def test_distance_to_electricity(model_object):#TODO
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Population/Population.tif"
    )

    model_object.add_layer(
        category="Demographics",
        name="Population",
        path=path,
        layer_type="raster",
        base_layer="True"
    )
    model_object.population_to_dataframe()
    mv_path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Electricity",
        "MV_lines",
        "MV_lines.geojson"
    )
    mv_lines = VectorLayer(
        "Electricity",
        "MV_lines",
        mv_path,
        distance_method="proximity"
    )
    model_object.distance_to_electricity(mv_lines=mv_lines)
    assert model_object.gdf is not None
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)


def test_raster_to_dataframe(model_object):#TODO
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Demographics",
        "Population/Population.tif"
    )

    model_object.add_layer(
        category="Demographics",
        name="Population",
        path=path,
        layer_type="raster",
        base_layer="True"
    )
    model_object.population_to_dataframe()
    ntl = os.path.join(
        "onstove",
        "tests",
        "data",
        "RWA",
        "Electricity",
        "Night_time_lights",
        "Night_time_lights.tif"
    )
    model_object.raster_to_dataframe(
        ntl,
        name="Night_lights",
        method="read",
        fill_nodata_method="interpolate"
    )
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)
    assert model_object.gdf["Night_lights"] is not None

