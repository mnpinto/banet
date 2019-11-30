import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from shapely.geometry import Point
import geopandas as gpd
from IPython.core.debugger import set_trace

from .geo import rasterize

__all__ = ['BandsFilter', 'BandsRename', 'ActiveFires', 'MirCalc', 'BandsAssertShape',
           'MergeTiles']

class BandsFilter():
    """Remove bands not in to_keep list from the dictionary."""
    def __init__(self, to_keep: list):
        self.to_keep = to_keep if isinstance(to_keep, list) else [to_keep]
        
    def __call__(self, data:dict, *args, **kwargs) -> dict:
        keys = [k for k in data]
        for k in keys:
            if k not in self.to_keep:
                del data[k]
        return data
    

class BandsRename():
    def __init__(self, input_names:list, output_names:list):
        self.input_names = input_names if isinstance(input_names, list) else [input_names]
        self.output_names = output_names if isinstance(output_names, list) else [output_names]

    def __call__(self, data:dict, *args, **kwargs) -> dict:
        for i, o in zip(self.input_names, self.output_names):
            data[o] = data.pop(i)
        return data


class MergeTiles():
    def __init__(self, band:str):
        self.band = band

    def __call__(self, data:dict, *args, **kwargs) -> dict:
        d = np.nanmean(np.array(data[self.band]), axis=(1,2))
        d = np.array(np.array(d).argsort())
        masks = np.array(data[self.band])[d]
        for k in data:
            data_aux = np.zeros_like(data[k][0])*np.nan
            for dband, mask in zip(np.array(data[k])[d], masks):
                I = (np.isnan(data_aux)) & (~np.isnan(mask))
                data_aux[I] = dband[I]
            data[k] = data_aux
        return data


class BandsAssertShape():
    def __call__(self, data:dict, *args, **kwargs) -> dict:
        for k in kwargs['cls'].bands:
            rshape = kwargs['cls'].region.shape
            if isinstance(data[k], list):
                for d in data[k]:
                    shape = d.shape
                    if len(shape) == 3: # first is time
                        shape = shape[1:]
                    if shape != rshape:
                        error = f'{k} shape {shape} does not match region shape {rshape}'
                        raise Exception(error)
            else:
                shape = data[k].shape
                if len(shape) == 3: # first is time
                    shape = shape[1:]
                if shape != rshape:
                    error = f'{k} shape {shape} does not match region shape {rshape}'
                    raise Exception(error)
        return data


class ActiveFires():
    """Get active fires and interpolate to grid"""
    def __init__(self, file):
        self.file = file
        self.lon = None
        self.lat = None
        self.df = self.load_csv()

    def load_csv(self):
        return pd.read_csv(self.file, parse_dates=['acq_date']).set_index('acq_date')
        
    def __call__(self, data, time, *args, **kwargs):
        if self.lon is None or self.lat is None: 
            self.lon, self.lat = kwargs['cls'].region.coords
        frp = self.df[self.df.index == time]

        if len(frp) > 0:
            geometry = [Point(xy) for xy in zip(frp['longitude'], frp['latitude'])]
            frp = gpd.GeoDataFrame(frp, geometry=geometry)
            out = rasterize(frp, 'frp', kwargs['cls'].region, merge_alg='add')
            out[out==0] = np.nan
        else: out = np.zeros(kwargs['cls'].region.shape)*np.nan

        data['FRP'] = out
        return data


class MirCalc():
    def __init__(self, solar_zenith_angle:str, mir_radiance:str, tir_radiance:str,
                 output_name:str='MIR'):
        self.sza = solar_zenith_angle
        self.r_mir = mir_radiance
        self.r_tir = tir_radiance
        self.output_name = output_name

    def __call__(self, data:dict, *args, **kwargs):
        sza = data[self.sza]
        mir = data[self.r_mir]
        tir = data[self.r_tir]
        data[self.output_name] = self.refl_mir_calc(mir, tir, sza, sensor=kwargs['cls'].name)
        return data

    def refl_mir_calc(self, mir, tir, sza, sensor):
        """
        Computes the MIR reflectance from MIR radiance and Longwave IR radiance.
        sensor can be "VIIRS375" or "VIIRS750"
        sza is the solar zenith angle
        for VIIRS375, mir is band I4 and tir band I5
        for VIIRS750, mir is band M12 and tir band M15
        returns a matrix of MIR reflectances with the same shape as mir and tir inputs.
        Missing values are represented by 0.
        """
        lambda_M12= 3.6966 # Programa Central_wave
        lambda_M15=10.7343 # PROGRAMAA CENTRAL_WAVE
        lambda_I4 = 3.7486 # Programa Central_wave
        lambda_I5 = 11.4979 # Programa Central_wave

        c1 = 1.1911e8 # [ W m-2 sr-1 (micrometer -1)-4 ]
        c2 = 1.439e4 # [ K micrometer ]
        E_0_mir_M12 = 11.7881 # M12 newkur_semcab
        E_0_mir_I4= 11.2640 # I4 newkur_semcab

        if sensor=='VIIRS375':
            lambda_mir = lambda_I4
            lambda_tir = lambda_I5
            E_0_mir = E_0_mir_I4
        elif sensor=='VIIRS750':
            lambda_mir = lambda_M12
            lambda_tir = lambda_M15
            E_0_mir = E_0_mir_M12
        else: raise NotImplementedError(
            f'refl_mir_calc not implemented for {sensor}. Available options are VIIRS750 and VIIRS375.')

        miu_0=np.cos((sza*np.pi)/180)

        mir[mir <= 0] = np.nan
        tir[tir <= 0] = np.nan

        # Brighness temperature 
        a1 = (lambda_tir**5)
        a = c1/(a1*tir)
        logaritmo = np.log(a+1)
        divisor = lambda_tir*logaritmo
        T = (c2/divisor)
        del a, logaritmo, divisor

        # Plank function
        divisor2 = (lambda_mir*T)
        exponencial = np.exp(c2/divisor2)
        b = c1*(lambda_mir**-5)
        BT_mir = b/(exponencial-1)
        del divisor2, exponencial, b, T

        # MIR reflectance
        c = (E_0_mir*miu_0)/np.pi
        termo1 = (mir-BT_mir)
        termo2 = (c-BT_mir)
        Refl_mir = termo1/termo2
        Refl_mir[Refl_mir <= 0] = 0
        return Refl_mir