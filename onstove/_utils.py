import pandas as pd
import geopandas as gpd
from typing import Union, Optional

from onstove.layer import VectorLayer, RasterLayer


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


class Processes():
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
