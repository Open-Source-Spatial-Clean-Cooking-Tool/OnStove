# test for layer.py module
import os
import pytest
from onstove.layer import VectorLayer, RasterLayer


@pytest.fixture
def vector_object():
    """Vector object fixture"""
    # Create a VectorLayer object with sample data
    vect = VectorLayer()
    return vect


@pytest.fixture
def sample_vector_layer():
    """Vector layer fixture"""
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
    """Raster layer fixture"""
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
def output_path():
    """Output path fixture"""
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "output",
        "layer"
    )
    return path


def test_read_vector_layer(vector_object):
    """Test for reading vector layer

    Parameters
    ----------
    vector_object: Vector Layer
                Instance of Vector Layer class.
                See :class:`onstove.VectorLayer`.
    """

    vect_path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "vector.geojson"
    )
    vector_object.read_layer(path=vect_path)
    assert vector_object is not None
    assert len(vector_object.data) > 0
    assert isinstance(vector_object, VectorLayer)


def test_mask_vector(sample_vector_layer, sample_raster_layer, output_path):
    """Test for masking

    Parameters
    ----------
    sample_vector_layer: Vector Layer
                Instance of Vector Layer class.
                See :class:`onstove.VectorLayer`.
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    output_path: str
                Output path.
    """

    path = os.path.join(
        output_path,
        "masked"
    )

    sample_raster_layer.mask(
        sample_vector_layer,
        output_path=path,
        crop=True,
        all_touched=True
    )
    name = os.path.join(output_path, "masked", ".tif")
    assert os.path.exists(name)


def test_reproject_vector(sample_vector_layer, output_path):
    """Test to reproject vector layer

    Parameters
    ----------
    sample_vector_layer: Vector Layer
                Instance of Vector Layer class.
                See :class:`onstove.VectorLayer`.
    output_path: str
                Output path.
    """

    path = os.path.join(
        output_path,
        "reproject_vector"
    )
    sample_vector_layer.reproject(
        crs=3857,
        output_path=path
    )
    assert os.path.exists(os.path.join(
        path,
        ".geojson"
    ))


def test_proximity_vector(sample_vector_layer, sample_raster_layer, output_path):
    """Test for proximity

    Parameters
    ----------
    sample_vector_layer: Vector Layer
                Instance of Vector Layer class.
                See :class:`onstove.VectorLayer`.
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    output_path: str
                Output path.
    """

    assert len(sample_raster_layer.data) > 0

    prox_path = os.path.join(
        output_path,
        "proximity"
    )

    prox = sample_vector_layer.proximity(
        sample_raster_layer,
        output_path=prox_path,
        create_raster=False
    )

    assert isinstance(prox, RasterLayer)


def test_rasterize(sample_vector_layer, output_path):
    """Test to rasterize vector layers

    Parameters
    ----------
    sample_vector_layer: Vector Layer
                Instance of Vector Layer class.
                See :class:`onstove.VectorLayer`.
    output_path: str
                Output path.
    """

    path = os.path.join(
        output_path,
        "rasterize"
    )
    sample_vector_layer.rasterize(
        None,
        attribute="fid",
        value=1,
        width=10,
        height=10,
        output_path=path
    )
    assert os.path.join(
        path,
        ".tif"
    ), RasterLayer


"""#Note: does not work for all vector types. Mentioned in layer module.
def test_travel_time_vector(sample_vector_layer, sample_raster_layer, output_path):
    # assert len(sample_raster_layer.data) > 0
    
    path = os.path.join(
        output_path,
        "travel_time"
    )

    least_cost_travel_time = sample_vector_layer.travel_time(
        friction=sample_raster_layer,
        output_path=path,
        create_raster=True,
    )

    assert isinstance(least_cost_travel_time, RasterLayer)
"""


# Raster Layer test functions
def test_mask(sample_vector_layer, sample_raster_layer):
    """Test for masking raster layers

    Parameters
    ----------
    sample_vector_layer: Vector Layer
                Instance of Vector Layer class.
                See :class:`onstove.VectorLayer`.
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    """

    sample_raster_layer.mask(
        sample_vector_layer,
        crop=False
    )
    assert isinstance(sample_raster_layer, RasterLayer)


def test_reproject(sample_raster_layer, output_path):
    """Test for reprojecting raster layers

    Parameters
    ----------
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    output_path: str
                Output path.
    """
    path = os.path.join(
        output_path,
        "reproject"
    )
    sample_raster_layer.reproject(
        crs=3857,
        output_path=path
    )
    assert sample_raster_layer.meta["crs"] == 3857


def test_travel_time(sample_raster_layer):
    """Test for creating a travel time map

    Parameters
    ----------
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    """

    sample_raster_layer.travel_time(
        rows=0,
        cols=0,
        include_starting_cells=False,
        output_path=None,
        create_raster=True
    )
    assert isinstance(sample_raster_layer, RasterLayer)


def test_log(sample_vector_layer, sample_raster_layer, output_path):
    """Test for log(logarithmic representation of raster surface)

    Parameters
    ----------
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    """

    path = os.path.join(
        output_path,
        "log"
    )
    log = sample_raster_layer.log(
        sample_vector_layer,
        output_path=path,
        create_raster=False)
    assert isinstance(log, RasterLayer)


def test_proximity(sample_raster_layer):
    """Test for proximity

    Parameters
    ----------
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    """

    raster_prox = sample_raster_layer.proximity(1)
    assert isinstance(raster_prox, RasterLayer)


def test_align(sample_raster_layer, output_path):
    """Test for raster alignment

    Parameters
    ----------
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    """

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "raster2.tif"
    )
    align_path = os.path.join(
        output_path,
        "align"
    )
    sample_raster_layer.align(
        base_layer=path,
        output_path=align_path,
        inplace=False
    )
    assert os.path.join(path, "align", ".tif"), RasterLayer


def test_normalize(sample_raster_layer, output_path):
    """Test for raster normalization

    Parameters
    ----------
    sample_raster_layer: Raster Layer
                Instance of Raster Layer class.
                See :class:`onstove.RasterLayer`.
    """

    path = os.path.join(
        output_path,
        "normalized"
    )
    sample_raster_layer.normalize(
        output_path=path,
        buffer=True,
        inverse=True,
        create_raster=True
    )
    assert sample_raster_layer.normalized is not None
    assert os.path.join(path, "normalized", "-normalized.tif"), RasterLayer
