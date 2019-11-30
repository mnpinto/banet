import rasterio
import geopandas as gp
import pandas as pd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
import scipy.io as sio

from .dataset import InOutPath, Region
from .predict import *


__all__ = ['get_fires', 'ba_json']


def get_fires(io_path, url, region, save=True):
    files = io_path.inp.ls(include=['.csv', f'hotspots{region}'])
    frp = [pd.read_csv(f) for f in files]
    frp = pd.concat([*frp, pd.read_csv(url)], axis=0, sort=False).drop_duplicates().reset_index(drop=True)
    if save:
        frp.to_csv(io_path.out/f'hotspots{region}.csv', index=False)
        print(f'hotspots{region}.csv updated')
    else:
        return frp


def ba_json(io_path:InOutPath, region:Region, min_size=1, save=True):
    files = sorted(io_path.inp.ls())
    data = sio.loadmat(files[-1])
    fires = split_mask(data['burned'], thr_obj=min_size)

    gpd = gp.GeoDataFrame()
    for fire in fires:
        dates = data['date'].copy()
        dates[fire==0] = np.nan
        start_date = np.nanmin(dates)
        end_date = np.nanmax(dates)
        geoms = list({'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(fire.astype(np.uint8), transform=region.transform))
        )
        g0 = gp.GeoDataFrame.from_features(geoms, crs={'init' :'epsg:4326'})
        g0['start'] = start_date.astype(int)
        g0['end'] = end_date.astype(int)
        gpd = pd.concat((gpd, g0), axis=0)

    gpd = gpd.loc[gpd.raster_val==1.0].reset_index(drop=True)
    gpd = gp.GeoDataFrame(gpd, crs={'init' :'epsg:4326'})
    gpd['area'] = (gpd.to_crs(epsg=25830).area.values*0.0001).astype(int) # TODO: EPGS for other regions
    gpd = gpd.drop('raster_val', axis=1)
    if save:
        gpd.to_file(io_path.out/'ba.json', driver="GeoJSON")
    else:
        return gpd

