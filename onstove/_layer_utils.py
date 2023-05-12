from onstove.layer import VectorLayer, RasterLayer
from typing import Union, Optional

def raster_setter(layer: Union[RasterLayer, str, None],
                  category: Optional[str] = None,
                  name: Optional[str] = ''):
    if isinstance(layer, str):
        return RasterLayer(category=category, name=name, path=layer)
    elif isinstance(layer, RasterLayer):
        return layer
    elif layer is None:
        return None
    else:
        raise ValueError('Raster file type or object not recognized.')


def vector_setter(layer: Union[RasterLayer, str, None],
                  category: Optional[str] = None,
                  name: Optional[str] = ''):
    if isinstance(layer, str):
        return VectorLayer(category=category, name=name, path=layer)
    elif isinstance(layer, VectorLayer):
        return layer
    elif layer is None:
        return None
    else:
        raise ValueError('Vector file type or object not recognized.')