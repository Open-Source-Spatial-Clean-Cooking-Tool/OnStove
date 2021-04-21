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
def get_targets_costs(targets_path, friction_path):
   
    targets = rasterio.open(targets_path)
    affine = targets.transform
    targets = targets.read(1)

    friction = rasterio.open(friction_path)
    costs = costs.read(1)

    target_list = np.argwhere(targets != 0.0)
    start = tuple(target_list[0].tolist())

    targets = targets.astype(np.int8)
    costs = costs.astype(np.float16)

    return targets, costs, start, affine

def optimise(targets, costs, start, jupyter=False, animate=False, affine=None, animate_path=None, silent=False):

    max_i = costs.shape[0]
    max_j = costs.shape[1]

    visited = np.zeros_like(targets, dtype=np.int8)
    dist = np.full_like(costs, np.nan, dtype=np.float32)

    prev = np.full_like(costs, np.nan, dtype=object)

    dist[start] = 0

    # dist, loc
    queue = [[0, start]]
    heapify(queue)

    def zero_and_heap_path(loc):
        if not dist[loc] == 0:
            dist[loc] = 0
            visited[loc] = 1

            heappush(queue, [0, loc])
            prev_loc = prev[loc]

            if type(prev_loc) == tuple:
                zero_and_heap_path(prev_loc)

    counter = 0
    progress = 0
    max_cells = targets.shape[0] * targets.shape[1]
    if jupyter:
        handle = display(Markdown(""), display_id=True)

    while len(queue):
        current = heappop(queue)
        current_loc = current[1]
        current_i = current_loc[0]
        current_j = current_loc[1]
        current_dist = dist[current_loc]

        for x in range(-1, 2):
            for y in range(-1, 2):
                next_i = current_i + x
                next_j = current_j + y
                next_loc = (next_i, next_j)

                # ensure we're within bounds
                if next_i < 0 or next_j < 0 or next_i >= max_i or next_j >= max_j:
                    continue

                # ensure we're not looking at the same spot
                if next_loc == current_loc:
                    continue

                # skip if we've already set dist to 0
                if dist[next_loc] == 0:
                    continue

                # if the location is connected
                if targets[next_loc]:
                    prev[next_loc] = current_loc
                    zero_and_heap_path(next_loc)

                # otherwise it's a normal queue cell
                else:
                    dist_add = costs[next_loc]
                    if x == 0 or y == 0:  # if this cell is  up/down/left/right
                        dist_add *= 1
                    else:  # or if it's diagonal
                        dist_add *= sqrt(2)

                    next_dist = current_dist + dist_add

                    if visited[next_loc]:
                        if next_dist < dist[next_loc]:
                            dist[next_loc] = next_dist
                            prev[next_loc] = current_loc
                            heappush(queue, [next_dist, next_loc])

                    else:
                        heappush(queue, [next_dist, next_loc])
                        visited[next_loc] = 1
                        dist[next_loc] = next_dist
                        prev[next_loc] = current_loc

                        counter += 1
                        progress_new = 100 * counter / max_cells
                        if int(progress_new) > int(progress):
                            progress = progress_new
                            message = f"{progress:.2f} %"
                            if jupyter:
                                handle.update(message)
                            elif not silent:
                                print(message)
                            if animate:
                                i = int(progress)
                                path = os.path.join(animate_path, f"arr{i:03d}.tif")
                                save_raster(path, dist, affine)

    return dist

def friction_start_points(friction, start):
    with rasterio.open(friction) as src:  
    arr = src.read(1)
    for index,row in start.iterrows():
        rows, cols = rasterio.transform.rowcol(src.transform, row["geometry"].x, row["geometry"].y)
        arr[rows][cols] = 0
        
        out_meta = src.meta
        
    with rasterio.open('testar2.tif', 'w', **out_meta) as dst:
        dst.write(arr, indexes = 1)
        dst.close()
        
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
        
        
def rasterize(vector_layer, raster_extent_path, outpul_file, value,
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