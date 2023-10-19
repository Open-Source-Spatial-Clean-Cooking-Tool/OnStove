import pytest
import os

from onstove.model import OnStove


@pytest.fixture
def output_path():
    path = os.path.join(
        "onstove",
        "tests",
        "data",
        "output"
    )
    return path


# fixtures
@pytest.fixture
def model_object():
    model = OnStove()

    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA"
    )
    model = model.read_model(os.path.join(path, 'model.pkl'))
    path2 = os.path.join(
        'onstove',
        'tests',
        'data',
        'RWA',
        'RWA_scenario_file.csv')
    model.read_scenario_data(path2, delimiter=',')

    model.lpg = model.techs['LPG']
    model.ci_biomass = model.techs['Collected_Improved_Biomass']
    model.ct_biomass = model.techs['Collected_Traditional_Biomass']
    model.charcoal_ics = model.techs['Charcoal ICS']
    model.electricity = model.techs['Electricity']
    model.biogas = model.techs['Biogas']

    return model


def test_infra_cost(model_object):
    model_object.lpg.infrastructure_cost(model=model_object)
    assert model_object.lpg.discounted_infra_cost is not None


# Class LPG
def test_transportation_cost(model_object):
    model_object.lpg.transportation_cost(model=model_object)


def test_carb(model_object):
    model_object.lpg.carb(model=model_object)


def test_infrastructure_cost(model_object):
    model_object.lpg.infrastructure_cost(model=model_object)
    assert model_object.lpg.discounted_infra_cost is not None


def test_infrastructure_salvage(model_object):
    val = model_object.lpg.infrastructure_salvage(
        model=model_object,
        cost=model_object.lpg.cylinder_cost,
        life=model_object.lpg.cylinder_life
    )
    assert val is not None
    assert val > 0.0


# Class Charcoal
def test_carbon_intesity(model_object):
    model_object.charcoal_ics.get_carbon_intensity(model=model_object)
    assert model_object.charcoal_ics.carbon_intensity is not None


def test_production_emissions(model_object):
    hh_emissions = model_object.charcoal_ics.production_emissions(model=model_object)
    assert hh_emissions is not None


# Electricity
def test_get_capacity_cost(model_object):
    model_object.electricity.get_capacity_cost(model=model_object)
    assert model_object.electricity.capacity_cost is not None
    assert model_object.electricity.capacity_cost >= 0


def test_get_carbon_intensity(model_object):
    model_object.electricity.get_carbon_intensity(model=model_object)
    assert model_object.electricity.carbon_intensity is not None
    assert model_object.electricity.carbon_intensity >= 0


def test_grid_capacity_cost(model_object):
    model_object.electricity.get_grid_capacity_cost()
    assert model_object.electricity.grid_capacity_cost is not None
    assert model_object.electricity.grid_capacity_cost >= 0


def test_grid_salvage(model_object):
    grid_salv = model_object.electricity.grid_salvage(model=model_object)
    assert grid_salv is not None
    assert grid_salv > 0


def test_carb_electricity(model_object):
    model_object.electricity.carb(model=model_object)
    assert model_object.electricity.carbon is not None


# Biogas
def test_read_friction(model_object):
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "Biomass",
        "Friction",
        "Friction.tif"
    )
    frict = model_object.biogas.read_friction(model=model_object, friction_path=path)
    assert frict is not None


def test_required_energy_hh(model_object):
    required_energy = model_object.biogas.required_energy_hh(model=model_object)
    assert required_energy is not None
    assert required_energy > 0


def test_get_collection_time(model_object):
    model_object.biogas.get_collection_time(model=model_object)
    assert model_object.biogas.time_of_collection is not None
    assert model_object.biogas.time_of_collection.all() > 0


def test_available_biogas(model_object):
    model_object.biogas.available_biogas(model=model_object)
    assert model_object.gdf['biogas_energy'] is not None
    assert model_object.gdf['biogas_energy'].all() >= 0


def test_recalibrate_livestock(model_object):
    path = os.path.join(
        "onstove",
        "tests",
        "tests_data",
        "RWA",
        "Biogas",
        "Livestock"
    )
    buffaloes = os.path.join(path, "buffaloes", "buffaloes.tif")
    cattles = os.path.join(path, "cattles", "cattles.tif")
    goats = os.path.join(path, "goats", "goats.tif")
    pigs = os.path.join(path, "pigs", "pigs.tif")
    poultry = os.path.join(path, "poultry", "poultry.tif")
    sheeps = os.path.join(path, "sheeps", "sheeps.tif")

    model_object.biogas.recalibrate_livestock(
        model=model_object,
        buffaloes=buffaloes,
        cattles=cattles,
        goats=goats,
        pigs=pigs,
        poultry=poultry,
        sheeps=sheeps
    )

    assert model_object.gdf['Buffaloes'].all() is not None
    assert model_object.gdf['Cattles'].all() is not None
    assert model_object.gdf['Poultry'].all() is not None
    assert model_object.gdf['Goats'].all() is not None
    assert model_object.gdf['Pigs'].all() is not None
    assert model_object.gdf['Sheeps'].all() is not None


def test_total_time(model_object):
    model_object.biogas.total_time(model=model_object)
    assert model_object.biogas.total_time_yr is not None
    assert model_object.biogas.total_time_yr.all() >= 0
