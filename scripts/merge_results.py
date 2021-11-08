import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from shapely import wkt

df_all = pd.DataFrame({'country': [], 'max_benefit_tech': [], 'geometry': []})

print('Reading country results')
for file, country in zip(snakemake.input.results, snakemake.params.countries):
    print(f'    - {country}')
    df = pd.read_csv(file)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    df['max_benefit_tech'] += ' and '
    df = df.groupby('index').agg({'max_benefit_tech': 'sum',
                                  'geometry': 'first'})
    df['country'] = country
    df['max_benefit_tech'] = df['max_benefit_tech'].str[0:-5]
    df_all = df_all.append(df, ignore_index=True)

summary_all = pd.DataFrame({'max_benefit_tech': [], 'Calibrated_pop': [],
                            'maximum_net_benefit': [], 'deaths_avoided': [],
                            'health_costs_avoided': [], 'time_saved': [],
                            'reduced_emissions': [], 'investment_costs': [],
                            'fuel_costs': [], 'emissions_costs_saved': []})

print('Reading country summaries')
for file, country in zip(snakemake.input.summaries, snakemake.params.countries):
    print(f'    - {country}')
    summary = pd.read_csv(file)
    summary['country'] = country
    summary_all = summary_all.append(summary, ignore_index=True)

print('Writing Africa summary')
summary_all = summary_all.append(summary_all.sum(numeric_only=True), ignore_index=True)
summary_all['max_benefit_tech'] = summary_all['max_benefit_tech'].fillna('Total')
summary_all['country'] = summary_all['country'].fillna('Africa')
summary_africa = summary_all.groupby('max_benefit_tech').sum()
summary_africa.to_csv(snakemake.output.summary)

print('Creating max benefit technology map')
df_all['max_benefit_tech'] = df_all['max_benefit_tech'].str.replace('_', ' ')
tech_codes = {tech: i for i, tech in enumerate(df_all['max_benefit_tech'].unique())}
df_all['max_benefit_tech_code'] = [tech_codes[s] for s in df_all['max_benefit_tech']]

df_all['geometry'] = df_all['geometry'].apply(wkt.loads)
gdf_all = gpd.GeoDataFrame(df_all, crs='epsg:3857')

boundaries = gpd.read_file(snakemake.input.boundaries)
boundaries.to_crs(3857, inplace=True)

total_bounds = boundaries['geometry'].total_bounds
height = round((total_bounds[3] - total_bounds[1]) / 1000)
width = round((total_bounds[2] - total_bounds[0]) / 1000)
transform = rasterio.transform.from_bounds(*total_bounds, width, height)
rasterized = features.rasterize(
                        ((g, v) for v, g in zip(gdf_all['max_benefit_tech_code'].values, gdf_all['geometry'].values)),
                        out_shape=(height, width),
                        transform=transform,
                        all_touched=True,
                        fill=111,
                        dtype=rasterio.uint8)

with rasterio.open(
        snakemake.output.map, 'w',
        driver='GTiff',
        dtype=rasterized.dtype,
        count=1,
        crs=3857,
        width=width,
        height=height,
        transform=transform,
        nodata=111
) as dst:
    dst.write(rasterized, indexes=1)
