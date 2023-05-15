import requests, zipfile, yaml

def download_data(country):
    path = "https://onstove-tests-data.s3.amazonaws.com/Trieste/country_dict.yaml"
    response = requests.get(path)
    open("country_dict.yaml", "wb").write(response.content)

    with open("country_dict.yaml", 'r') as file:
        country_dict = yaml.safe_load(file)

    tech_specs = country_dict[country]['tech']
    response = requests.get(tech_specs)
    open("tech_specs.csv", "wb").write(response.content)

    soc_specs = country_dict[country]['soc']
    response = requests.get(soc_specs)
    open("soc_specs.csv", "wb").write(response.content)

    gis_data = country_dict[country]['GIS']
    response = requests.get(gis_data)
    open("gis_data.zip", "wb").write(response.content)

    with zipfile.ZipFile("gis_data.zip", "r") as zip_ref:
        zip_ref.extractall("gis_data")

    temperature = country_dict["Temp"]
    response = requests.get(temperature)
    open(r"gis_data\Temperature\Temperature.tif", "wb").write(response.content)

    urban = country_dict["Urban"]
    response = requests.get(urban)
    open(r"gis_data\Urban\Urban.tif", "wb").write(response.content)

    for livestock in ['buffaloes','cattles','poultry','goats','pigs','sheeps']:
        livestocks = country_dict['Livestock'][livestock]
        response = requests.get(livestocks)
        open(f"gis_data/Livestock/{livestock}/{livestock}.tif", "wb").write(response.content)
