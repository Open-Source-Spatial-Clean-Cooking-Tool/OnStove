from onstove.layer import VectorLayer, RasterLayer


def raster_setter(raster):
    if isinstance(raster, str):
        return RasterLayer(category='Mini-grids', name='Access', path=raster)
    elif isinstance(raster, RasterLayer):
        return raster
    elif raster is None:
        return None
    else:
        raise ValueError('Raster file type or object not recognized.')
