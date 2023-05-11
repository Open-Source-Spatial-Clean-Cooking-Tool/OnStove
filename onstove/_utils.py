import pandas as pd
import geopandas as gpd
from typing import Union, Optional, Dict, TypeVar, Any

from onstove.layer import VectorLayer, RasterLayer

KeyType = TypeVar('KeyType')

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

def deep_update(mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]) -> Dict[KeyType, Any]:
    """This source code is originally created by Samuel Colvin and other contributors and has been sourced from the
    `pydantic` library. Hereby we provide attribution to them."""
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


class Processes:
    def __init__(self):
        self.gdf = gpd.GeoDataFrame()

    def normalize(self, column: str, inverse: bool = False) -> pd.Series:
        """Uses the MinMax method to normalize the data from a column of the :attr:`gdf` GeoDataFrame.

        Parameters
        ----------
        column: str
            Name of the column of the :attr:`gdf` to create the normalized data from.
        inverse: bool, default False
            Whether to invert the range in the normalized data.

        Returns
        -------
        pd.Series
            The normalized pd.Series.
        """
        if column in self.gdf.columns:
            data = self.gdf[column]
        elif column in self.__dict__.keys():
            data = self.__dict__[column]
        else:
            raise KeyError(f'Variable "{column}" was not found in the data.')

        if inverse:
            normalized = (data.max() - data) / (data.max() - data.min())
        else:
            normalized = (data - data.min()) / (data.max() - data.min())

        return normalized
