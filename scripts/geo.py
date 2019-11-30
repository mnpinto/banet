import numpy as np
import geopandas as gp
from pathlib import Path
import pandas as pd
import rasterio
import shapely
from rasterio import features
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.coords import BoundingBox
import warnings
from IPython.core.debugger import set_trace

from .util import filter_files

__all__ = ['open_shp', 'open_tif', 'rasterize', 'downsample', 'get_coords', 
           'size_from_bounds', 'bounds_from_shapefile', 'bounds_from_coords', 'crop',
           'polygon_from_bounds', 'is_intersection']

open_shp =  lambda file: gp.read_file(file)

open_tif = lambda file: rasterio.open(file)

def bounds_from_shapefile(x):
    bounds = x.bounds
    return bounds.minx.min(), bounds.miny.min(), bounds.maxx.max(), bounds.maxy.max()

def size_from_bounds(bounds, resolution):
    mlon = np.mean([bounds[2], bounds[0]])
    width = np.ceil((bounds[2]-bounds[0])*(111100/resolution)*np.cos(np.deg2rad(mlon))).astype(int)
    height = np.ceil((bounds[3]-bounds[1])*(111100/resolution)).astype(int)
    return width, height

def size_resolution_assert(size, resolution):
    if size is None and resolution is None: 
        raise Exception('You must define either size or resolution')
    if size is not None and resolution is not None:
        warnings.warn('resolution not used, computed based on size and bounds')
        
def rasterize(x, value_key=None, region=None, merge_alg='replace'):
    if merge_alg == 'replace':
        merge_alg = rasterio.enums.MergeAlg.replace
    elif merge_alg == 'add':
        merge_alg = rasterio.enums.MergeAlg.add
    values = [1]*len(x) if value_key is None else x[value_key]
    shapes = (v for v in zip(x.geometry, values))
    return rasterio.features.rasterize(shapes, out_shape=region.shape, 
            transform=region.transform, merge_alg=merge_alg)
    
def downsample(x, src_tfm=None, dst_tfm=None, dst_shape=None, 
               src_crs={'init': 'EPSG:4326'}, dst_crs={'init': 'EPSG:4326'},
               resampling='average'):
    """
    x is a numpy array
    """
    if resampling == 'average':
        resampling = rasterio.warp.Resampling.average
    elif resampling == 'bilinear':
        resampling = rasterio.warp.Resampling.bilinear
    elif resampling == 'nearest':
        resampling = rasterio.warp.Resampling.nearest
    out = np.zeros(dst_shape)
    rasterio.warp.reproject(x, out, src_transform=src_tfm, dst_transform=dst_tfm, 
                            src_crs=src_crs, dst_crs=dst_crs, resampling=resampling)
    return out

def is_intersection(gdf1, gdf2):
    return len(gp.overlay(gdf1, gdf2, how='intersection')) > 0

def polygon_from_bounds(bounds, to_GeoDataFrame=False, crs={'init': 'EPSG:4326'}):
    b_ind = [[0,1],[2,1],[2,3],[0,3]]
    shape = shapely.geometry.Polygon([(bounds[i],bounds[j]) for i, j in b_ind])
    if to_GeoDataFrame: shape = gp.GeoDataFrame(crs=crs, geometry=[shape])
    return shape

def crop(x, bounds=None, shape=None, crop=True):
    """
    x is a dataset or a list of datasets (rasterio.open). 
    If list then merge with bounds is used.
    else mask is used to crop given bounds or any given shape.
    """
    if len(x) == 1 and isinstance(x, list):
        x = x[0]
    if isinstance(x, list): 
        out, transform = merge(x, bounds)
    else:
        if bounds is not None: shape = polygon_from_bounds(bounds)
        out, transform = mask(x, shapes=[shape], crop=crop)
    return out.squeeze(), transform

def get_coords(x, transform, offset='ul'):
    if isinstance(x, tuple):
        shape = x
    elif hasattr(x, shape):
        shape = x.shape
    else: raise NotImplementedError('x must have a shape attribute or be a shape tuple.')
    rxy = rasterio.transform.xy
    ys, xs = map(range, shape)    
    return (np.array(rxy(transform, [0]*len(xs), xs, offset=offset)[0]), 
            np.array(rxy(transform, ys, [0]*len(ys), offset=offset)[1]))

def bounds_from_coords(x, y):
    return x.min(), y.min(), x.max(), y.max()