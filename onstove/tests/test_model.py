# Test for models.py
import os
import geopandas as gpd
import pytest
from onstove.model import DataProcessor, MCA, OnStove
from onstove.layer import VectorLayer, RasterLayer


@pytest.fixture
def sample_vector_layer():
    """Vector layer fixture"""

    # set path
    vect_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "Administrative",
        "Country_boundaries",
        "Country_boundaries.geojson")
    # Create a VectorLayer object with sample data
    vect = VectorLayer(path=vect_path)
    return vect


@pytest.fixture
def sample_raster_layer():
    """Raster layer fixture"""

    # set path
    rast_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
    """Model object fixture"""

    model = OnStove(project_crs=3857)
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "RWA_prep_file.csv"
    )
    model.read_scenario_data(path, delimiter=',')
    return model


@pytest.fixture
def data_object():
    """Data object fixture"""

    data = DataProcessor(project_crs=3857, cell_size=(1000, 1000))
    return data


@pytest.fixture
def mca_object():
    """MCA object fixture"""

    mca = MCA(project_crs=3857)
    return mca


@pytest.fixture
def output_path():
    """Output path fixture"""

    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "output"
    )
    return path


# tests for DataProcessor
def test_model(model_object):
    """Test if model exists

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    # test if model is instance OnStove
    assert isinstance(model_object, OnStove)


def test_add_layer(model_object):
    """Test for add layer function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
    """Test for add mask layer function

    Parameters
    ----------
    data_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "Administrative",
        "Country_boundaries",
        "Country_boundaries.geojson")
    data_object.add_mask_layer(
        path=path
    )
    assert isinstance(data_object, DataProcessor)


def test_mask_layers(data_object, sample_raster_layer):
    """Test for mask layers function

    Parameters
    ----------
    data_object: Model
                Instance of Model class.
    sample_raster_layer: Raster Layer
                Instance of Raster class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
    """Test for get distance rasters function

    Parameters
    ----------
    data_object: Model
                Instance of Model class.
    """

    data_object.get_distance_rasters(
        "all",
        False
    )
    assert isinstance(data_object, DataProcessor)


def test_to_pickle(model_object, output_path):
    """Test for test to pickle function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    output_path: str
                Output path.
    """

    model_object.output_directory = output_path
    name = "model_object.pkl"
    model_object.to_pickle(name=name)

    assert os.path.exists(
        os.path.join(
            model_object.output_directory,
            "model_object.pkl"
        )
    )


# OnStove
def test_read_scenario_data(model_object):
    """Test for read scenario data function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "RWA_scenario_file.csv"
    )
    model_object.read_scenario_data(path, delimiter=',')
    assert model_object.specs is not None
    assert isinstance(model_object.specs, dict)


def test_population_to_dataframe(model_object):
    """Test for population to dataframe function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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


def test_calibrate_urban_rural_split(model_object):
    """Test for calibrate urban rural split function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "Demographics",
        "Population",
        "Population.tif")
    model_object.add_layer(
        category="Demographics",
        name="Population",
        path=path,
        layer_type="raster",
        base_layer="True"
    )
    model_object.population_to_dataframe()

    ghs_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "Demographics",
        "Urban",
        "Urban.tif"
    )
    model_object.calibrate_urban_rural_split(ghs_path)
    assert model_object.gdf is not None
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)


def test_extract_wealth_index(model_object):
    """Test for extract wealth index function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
        "tests_data",
        "RWA",
        "Demographics",
        "Wealth",
        "RWA_relative_wealth_index.csv"
    )
    # extract wealth index and add to gdf
    model_object.extract_wealth_index(wealth_index=wealth_idx, file_type="csv")
    assert model_object.gdf is not None
    assert isinstance(model_object.gdf, gpd.GeoDataFrame)


def test_distance_to_electricity(model_object):
    """Test for distance to electricity function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
        "tests_data",
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


def test_raster_to_dataframe(model_object):
    """Test for raster to dataframe function

    Parameters
    ----------
    model_object: Model
                Instance of Model class.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
    ntl = os.path.join(
        "onstove",
        "tests",
        "tests_data",
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
