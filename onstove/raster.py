import glob
import gzip

import fiona
import numpy as np

import rasterio
import rasterio.mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as enumsResampling


def align_raster(raster_1, raster_2, method='nearest', compression='DEFLATE'):
    raster_1_meta = raster_1.meta
    raster_2_meta = raster_2.meta

    out_meta = raster_1_meta.copy()
    out_meta.update({
        'transform': raster_1_meta['transform'],
        'crs': raster_1_meta['crs'],
        'compress': compression,
        'nodata': raster_2_meta['nodata'],
        'dtype': raster_2_meta['dtype']
    })
    destination = np.full((raster_1_meta['height'], raster_1_meta['width']), raster_2_meta['nodata'])
    reproject(
        source=raster_2.data,
        destination=destination,
        src_transform=raster_2_meta['transform'],
        src_crs=raster_2_meta['crs'],
        src_nodata=raster_2_meta['nodata'],
        dst_transform=raster_1_meta['transform'],
        dst_crs=raster_1_meta['crs'],
        resampling=Resampling[method])
    return destination, out_meta


# def interpolate(raster, max_search_distance=10):
#     with rasterio.open(raster) as src:
#         profile = src.profile
#         arr = src.read(1)
#         arr_filled = fillnodata(arr, mask=src.read_masks(1), max_search_distance=max_search_distance)
#
#     with rasterio.open(raster, 'w', **profile) as dest:
#         dest.write_band(1, arr_filled)


def mask_raster(raster_path, mask_layer, output_file, nodata=0, compression='NONE',
                all_touched=False):
    if isinstance(mask_layer, str):
        with fiona.open(mask_layer, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            crs = 'EPSG:4326'
    else:
        shapes = [mask_layer.dissolve().geometry.loc[0]]
        crs = mask_layer.crs

    if '.gz' in raster_path:
        with gzip.open(raster_path) as gzip_infile:
            with rasterio.open(gzip_infile) as src:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=nodata,
                                                              all_touched=all_touched)
                out_meta = src.meta
    else:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=nodata,
                                                          all_touched=all_touched)
            out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform,
                     'compress': compression,
                     'nodata': nodata,
                     "crs": crs})

    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(out_image)


def reproject_raster(raster_path, dst_crs,
                     cell_width=None, cell_height=None, method='nearest',
                     compression='DEFLATE'):
    """
    Resamples and/or reproject a raster layer.
    """
    with rasterio.open(raster_path) as src:
        # Calculates the new transform, widht and height of 
        # the reprojected layer
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs,
            src.width,
            src.height,
            *src.bounds)
        # If a destination cell width and height was provided, then it 
        # calculates the new boundaries, with, heigh and transform 
        # depending on the new cell size.
        if cell_width and cell_height:
            bounds = rasterio.transform.array_bounds(height, width, transform)
            width = int(width * (transform[0] / cell_width))
            height = int(height * (abs(transform[4]) / cell_height))
            transform = rasterio.transform.from_origin(bounds[0], bounds[3],
                                                       cell_width, cell_height)
        # Updates the metadata
        out_meta = src.meta.copy()
        out_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': compression
        })
        # The layer is then reprojected/resampled
        # if output_file:
        #     # if an output file path was provided, then the layer is saved
        #     with rasterio.open(output_file, 'w', **out_meta) as dst:
        #         for i in range(1, src.count + 1):
        #             reproject(
        #                 source=rasterio.band(src, i),
        #                 destination=rasterio.band(dst, i),
        #                 src_transform=src.transform,
        #                 src_crs=src.crs,
        #                 dst_transform=transform,
        #                 dst_crs=dst_crs,
        #                 resampling=Resampling[method])
        # else:
            # If not outputfile is provided, then a numpy array and the 
            # metadata if returned
        destination = np.full((height, width), src.nodata)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling[method])
        return destination, out_meta


def sample_raster(path, gdf):
    with rasterio.open(path) as src:
        return [float(val) for val in src.sample([(x.coords.xy[0][0],
                                                   x.coords.xy[1][0]) for x in
                                                  gdf['geometry']])]


def merge_rasters(files_path, dst_crs, outpul_file):
    files = glob.glob(files_path)
    src_files_to_mosaic = []

    for fp in files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic[0].shape[0],
                     "width": mosaic[0].shape[1],
                     "transform": out_trans,
                     "crs": dst_crs
                     }
                    )
    with rasterio.open(outpul_file, "w", **out_meta) as dest:
        dest.write(mosaic[0], indexes=1)


def normalize(raster=None, limit=None, output_file=None,
              inverse=False, meta=None, buffer=False):
    if isinstance(raster, str):
        with rasterio.open(raster) as src:
            raster = src.read(1)
            nodata = src.nodata
            meta = src.meta
    else:
        raster = raster.copy()
        nodata = meta['nodata']
        meta = meta
    if callable(limit):
        raster[~limit(raster)] = np.nan

    raster[raster == nodata] = np.nan
    min_value = np.nanmin(raster)
    max_value = np.nanmax(raster)
    raster = (raster - min_value) / (max_value - min_value)
    if inverse:
        if not buffer:
            raster[np.isnan(raster)] = 1
        raster[raster < 0] = np.nan
        raster = 1 - raster
    else:
        if not buffer:
            raster[np.isnan(raster)] = 0
        raster[raster < 0] = np.nan

    meta.update(nodata=np.nan, dtype='float32')

    if output_file:
        with rasterio.open(output_file, "w", **meta) as dest:
            dest.write(raster.astype('float32'), indexes=1)
    else:
        return raster, meta


def resample(raster_path, height, width, method='bilinear'):
    with rasterio.open(raster_path) as src:
        # resample data to target shape
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * (abs(src.transform[4]) / height)),
                int(src.width * (abs(src.transform[0]) / width))
            ),
            resampling=enumsResampling[method]
        )

        # scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
        return data, transform