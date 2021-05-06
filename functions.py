import os
import glob
import numpy as np
from math import sqrt
from heapq import heapify, heappush, heappop
import rasterio
import rasterio.mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.fill import fillnodata
from rasterio import features
import geopandas as gpd
import fiona
import shapely
from osgeo import gdal, osr


def proximity_raster(src_filename, dst_filename, values, compression):
    src_ds = gdal.Open(src_filename)
    srcband=src_ds.GetRasterBand(1)
    dst_filename=dst_filename

    drv = gdal.GetDriverByName('GTiff')
    dst_ds = drv.Create(dst_filename,
                        src_ds.RasterXSize, src_ds.RasterYSize, 1,
                        gdal.GetDataTypeByName('Float32'),
                        options=['COMPRESS={}'.format(compression)])

    dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )
    dst_ds.SetProjection( src_ds.GetProjectionRef() )

    dstband = dst_ds.GetRasterBand(1)

    gdal.ComputeProximity(srcband,dstband,["VALUES={}".format(','.join([str(i) for i in values])),
                                           "DISTUNITS=GEO"])
    srcband = None
    dstband = None
    src_ds = None
    dst_ds = None
    
def mask_raster(raster_path, mask_layer, outpul_file, nodata=0, compression='NONE'):
    if isinstance(mask_layer, str):
        with fiona.open(mask_layer, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            crs = 'EPSG:4326'
    else:
        shapes = [mask_layer.dissolve().geom.loc[0]]
        crs = mask_layer.crs

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=nodata)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform,
                     'compress': compression,
                     'nodata': nodata,
                     "crs": crs})
    
    with rasterio.open(outpul_file, "w", **out_meta) as dest:
        dest.write(out_image)
    
    
def reproject_raster(raster_path, dst_crs, outpul_file, compression='NONE'):
    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': compression
        })

        with rasterio.open(outpul_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
         
        
def sample_raster(path, gdf):
    with rasterio.open(path) as src:
        return [float(val) for val in src.sample([(x.coords.xy[0][0], 
                                                   x.coords.xy[1][0]) for x in 
                                                   gdf['geometry']])]

def friction_start_points(friction, in_points, out_raster):
    start = gpd.read_file(in_points)
    row_list = []
    col_list = []
    with rasterio.open(friction) as src:  
        arr = src.read(1)
        for index,row in start.iterrows():
            rows, cols = rasterio.transform.rowcol(src.transform, row["geometry"].x, row["geometry"].y)
            arr[rows][cols] = 0
        
            out_meta = src.meta
               
            row_list.append(rows)
            col_list.append(cols)
        
    with rasterio.open(out_raster, 'w', **out_meta) as dst:
        dst.write(arr, indexes = 1)
        dst.close()
        
    return row_list, col_list
        
def merge_rasters(files_path, dst_crs, outpul_file):
    files = glob.glob(files_path)
    src_files_to_mosaic = []
    
    for fp in files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": dst_crs
                 }
                )
    
    with rasterio.open(outpul_file, "w", **out_meta) as dest:
        dest.write(mosaic)
        
        
def rasterize(vector_layer, raster_extent_path, outpul_file, value=None,
              nodata=None, fill = 1, compression='NONE', dtype=rasterio.uint8, all_touched=False):
    
    vector_layer = vector_layer.rename(columns={'geometry': 'geom'})
    if value:
        dff = vector_layer[[value, 'geom']]
        shapes = ((g, v) for v, g in zip(dff[value].values, dff['geom'].values))
    else:
        shapes = ((g, 1) for g in vector_layer['geom'].values)

    with rasterio.open(raster_extent_path) as src:
        image = features.rasterize(
                    shapes,
                    out_shape=src.shape,
                    transform=src.transform,
                    all_touched=all_touched)

        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": src.height,
                         "width": src.width,
                         "transform": src.transform,
                         'compress': compression,
                         'dtype': dtype,
                         "crs": src.crs,
                         'nodata': fill})

        with rasterio.open(outpul_file, 'w', **out_meta) as dst:
            dst.write(image, indexes=1)
           
                        
def normalize(raster_path):
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        nodata = src.nodata

        raster[raster!=nodata] = raster[raster!=nodata] / (np.nanmax(raster[raster!=nodata]) - np.nanmin(raster[raster!=nodata]))
        raster[raster<0] = np.nan
        
    return raster
        
        
def index(rasters, weights):
    raster = []
    for r, w in zip(rasters, weights):
        raster.append(w * r)

    return sum(raster) / sum(weights)

def lpg_transportation_cost(travel_time):
    
    """"The cost of transporting LPG. See https://iopscience.iop.org/article/10.1088/1748-9326/6/3/034002/pdf for the formula 
    
    Transportation cost = (2 * diesel consumption per h * national diesel price * travel time)/transported LPG
    
    Total cost = (LPG cost + Transportation cost)/efficiency of LPG stoves
    
    
    Each truck is assumed to transport 2,000 kg LPG 
    (3.5 MT truck https://www.wlpga.org/wp-content/uploads/2019/09/2019-Guide-to-Good-Industry-Practices-for-LPG-Cylinders-in-the-
    Distribution-Channel.pdf)
    National diesel price in Nepal is assumed to be 0.88 USD/l
    Diesel consumption per h is assumed to be 14 l/h (14 l/100km)
    (https://www.iea.org/reports/fuel-consumption-of-cars-and-vans)
    LPG cost in Nepal is assumed to be 19 USD per cylinder (1.34 USD/kg)
    LPG stove efficiency is assumed to be 60%
    
    Parameters
    ----------
    arg1 : national_LPG_price
        The national LPG price (USD)
    arg2 : travel_time_raster
        Hour to travel between each point and the startpoints as array
    Returns
    ----------
    The cost of LPG in each cell per kg
    
    """
    with rasterio.open(travel_time) as src:
        trav = src.read(1)
        
    transport_cost = (2*14*0.88*trav)/2000
    total_cost = (transport_cost + 1.34)/0.6
    
    return total_cost
