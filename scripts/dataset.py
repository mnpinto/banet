import numpy as np
import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
import re
from calendar import monthlen
from tqdm import tqdm
import scipy.io as sio
from pathlib import Path
from functools import partial
from pyhdf.SD import SD, SDC
from concurrent.futures import ThreadPoolExecutor
from warnings import warn
from IPython.core.debugger import set_trace

from .util import *
from .geo import get_coords, open_tif, open_shp, downsample, crop, rasterize
from .processors import BandsAssertShape

__all__ = ['InOutPath', 'Region', 'BaseDataset', 'Viirs750Dataset', 'MCD64Dataset', 
           'FireCCI51Dataset', 'AusCoverDataset', 'MTBSDataset', 'ICNFDataset', 'Region2Tiles']

class InOutPath():
    def __init__(self, input_path:Path, output_path:Path):
        if isinstance(input_path, str): input_path = Path(input_path)
        if isinstance(output_path, str): output_path = Path(output_path)
        self.inp = input_path
        self.out = output_path
        self.mkoutdir()

    def mkoutdir(self):
        self.out.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return '\n'.join([f'{i}: {o}' for i, o in self.__dict__.items()]) + '\n'

class Region():
    def __init__(self, name:str, bbox:list, pixel_size:float):
        self.name = name
        self.bbox = BoundingBox(*bbox) # left, bottom, right, top
        self.pixel_size = pixel_size
        
    @property
    def width(self):
        return int(np.abs(self.bbox.left-self.bbox.right)/self.pixel_size)

    @property 
    def height(self):
        return int(np.abs(self.bbox.top-self.bbox.bottom)/self.pixel_size)

    @property
    def transform(self):
        return rasterio.transform.from_bounds(*self.bbox, self.width, self.height)

    @property 
    def shape(self):
        return (self.height, self.width)

    @property
    def coords(self):
        return get_coords(self.shape, self.transform)

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            args = json.load(f)
        return cls(args['name'], args['bbox'], args['pixel_size'])

    def export(self, file):
        """Export region information to json file"""
        dict2json(self.__dict__, file)

    def __repr__(self):
        return '\n'.join([f'{i}: {o}' for i, o in self.__dict__.items()]) + '\n'

class BaseDataset():
    def __init__(self, name:str, paths:InOutPath, region:Region, 
                 times:pd.DatetimeIndex=None, bands:list=None):
        self.paths = paths
        self.region = region
        self.name = name
        self.times = times
        self.bands = bands

        if self.times is None:
            self.times = self.find_dates()

    def list_files(self, time:pd.Timestamp) -> list:
        pass

    def find_dates(self):
        pass

    def open(self, files:list) -> dict:
        pass

    def save(self, time:pd.Timestamp, data:dict, do_compression=True):
        tstr = time.strftime('%Y%m%d')
        filename = f'{self.paths.out}/{self.name}{self.region.name}_{tstr}.mat'
        sio.savemat(filename, data, do_compression=do_compression)
   
    def process_one(self, time:pd.Timestamp, proc_funcs:list=[], save=True, **proc_funcs_kwargs):
        tstr = time.strftime('%Y%m%d')
        files = self.list_files(time)
        try:
            if len(files) > 0:
                data = self.open(files)
                proc_funcs = [BandsAssertShape()] + proc_funcs
                kwargs = {'cls': self, **proc_funcs_kwargs}
                for f in proc_funcs:
                    data = f(data, time, **kwargs)
                if save: 
                    self.save(time, data)   
                else: return data
            else: 
                warn(f'No files for {time}. Skipping to the next time.')
        except: warn(f'Unable to process files for {time}. Check if files are corrupted. Skipping to the next time.')
 
    def process_all(self, proc_funcs=[], max_workers=8, **proc_funcs_kwargs):
        process_one = partial(self.process_one, proc_funcs=proc_funcs, **proc_funcs_kwargs)
        with ThreadPoolExecutor(max_workers) as e: 
            list(tqdm(e.map(process_one, self.times), total=len(self.times))) 

    def match_times(self, other, on='month'):
        if on != 'month': 
            raise NotImplementedError('match_times is only implemented on month.')

        ym_other = sorted(set([(t.year, t.month) for t in other.times]))
        out = []
        for t in self.times:
            if (t.year, t.month) in ym_other:
                out.append(t)
        self.times = pd.DatetimeIndex(out)

    def filter_times(self, year):
        if year is not None:
            self.times = self.times[self.times.year == year]

    def __repr__(self):
        return '\n'.join([f'{i}: {o}' for i, o in self.__dict__.items()]) + '\n'


class Viirs750Dataset(BaseDataset):
    def __init__(self, paths:InOutPath, region:Region, 
                 times:pd.DatetimeIndex=None, bands:list=None):
        super().__init__('VIIRS750', paths, region, times, bands)
        self.times = self.check_files()

    def list_files(self, time:pd.Timestamp) -> list:
        if time in self.times:
            dayOfYear = str(time.dayofyear).zfill(3)
            files = self.paths.inp.ls(include=['NPP', f'.A{time.year}{dayOfYear}.'])
        return files

    def check_files(self):
        not_missing = []
        for i, t in tqdm(enumerate(self.times), total=len(self.times)):
            files = self.list_files(t)
            files = ';'.join([f.stem for f in files])
            if sum([s in files for s in self.bands]) != len(self.bands):
                print(f'Missing files for {t}')
            else: not_missing.append(i)
        return self.times[not_missing]
        
    def find_dates(self, first:pd.Timestamp=None, last:pd.Timestamp=None):
        pattern = r'^\w+.A(20[0-9][0-9])([0-3][0-9][0-9])..*$'
        times = []
        for f in self.paths.inp.ls():
            x = re.search(pattern, f.stem)
            if x is not None:
                year, doy = map(x.group, [1,2])
                times.append(pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=int(doy)-1))
        self.times = pd.DatetimeIndex(sorted(set(times)))
        if first is not None:
            self.times = self.times[self.times>=first]
        if last is not None:
            self.times = self.times[self.times<=last]
        return self.times

    def open(self, files:list) -> dict:
        data_dict = {b: [] for b in self.bands}
        for s in self.bands:
            f = sorted([f for f in files if s in f.name])
            if len(f) == 0:
                warn(f'No file for {s} found on {files}')
            for f0 in f:
                hdf_data = SD(str(f0), SDC.READ)
                hdf_file = hdf_data.select(s)
                scale = hdf_attr_check('Scale', hdf_file.attributes(), default=1)
                offset = hdf_attr_check('Offset', hdf_file.attributes(), default=0)
                data = hdf_file[:].astype(float)*scale + offset
                data[data <= -999] = np.nan
                data[data >= 65527] = np.nan
                data_dict[s].append(data)
        return data_dict


class MCD64Dataset(BaseDataset):
    def __init__(self, paths:InOutPath, region:Region, times:pd.DatetimeIndex=None,
                 bands=['bafrac']):
        super().__init__('MCD64A1C6', paths, region, times, bands)

    def list_files(self, time:pd.Timestamp) -> list:
        out = []
        if time in self.times:
            time = pd.Timestamp(f'{time.year}-{time.month}-01')
            time_pattern = f'.A{time.year}{time.dayofyear}.'
            files = self.paths.inp.ls(recursive=True, include=['burndate.tif', time_pattern],
                                exclude=['.xml'])
            # Find windows joint with region bounding box
            for f in files:
                data = open_tif(f)
                if not rasterio.coords.disjoint_bounds(data.bounds, self.region.bbox):
                    out.append(f)
        return out
    
    def find_dates(self, first:pd.Timestamp=None, last:pd.Timestamp=None):
        pattern = r'^\w+.A(20[0-9][0-9])([0-3][0-9][0-9])..*$'
        times = []
        for f in self.paths.inp.ls(recursive=True):
            x = re.search(pattern, f.stem)
            if x is not None:
                year, doy = map(x.group, [1,2])
                times.append(pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=int(doy)-1))
        self.times = pd.DatetimeIndex(sorted(set(times)))
        if first is not None:
            self.times = self.times[self.times>=first]
        if last is not None:
            self.times = self.times[self.times<=last]
        return self.times

    def file_time_range(self, file) -> pd.DatetimeIndex:
        pattern = r'^\w+.A(20[0-9][0-9])([0-3][0-9][0-9])..*$'
        x = re.search(pattern, file.stem)   
        year, doy = map(x.group, [1,2])
        t0 = pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=int(doy)-1)
        return pd.date_range(t0, periods=monthlen(t0.year, t0.month), freq='D')
    
    def open(self, files:list) -> dict:
        times = self.file_time_range(files[0])
        data_dict = {'times': times}
        out = np.zeros((len(times), *self.region.shape))
        data = [open_tif(f) for f in files]
        data, tfm = crop(data, self.region.bbox)
        for i, time in enumerate(times):
            x = (data == time.dayofyear).astype(np.int8)
            out[i] += downsample(x, tfm, self.region.transform, self.region.shape)
        data_dict[self.bands[0]] = out
        return data_dict

    def save(self, time:pd.Timestamp, data:dict, do_compression=True):
        v = self.bands[0]
        for i, t in enumerate(data['times']):
            super().save(t, {v: data[v][i]}, do_compression=do_compression)


class FireCCI51Dataset(BaseDataset):
    def __init__(self, paths:InOutPath, region:Region, times:pd.DatetimeIndex=None,
                 bands=['bafrac']):
        super().__init__('FireCCI51', paths, region, times, bands)

    def list_files(self, time:pd.Timestamp) -> list:
        out = []
        if time in self.times:
            time = pd.Timestamp(f'{time.year}-{time.month}-01')
            time_pattern = time.strftime('%Y%m%d')
            files = self.paths.inp.ls(recursive=True, include=['JD.tif', time_pattern],
                                exclude=['.xml'])
            # Find windows joint with region bounding box
            for f in files:
                data = open_tif(f)
                if not rasterio.coords.disjoint_bounds(data.bounds, self.region.bbox):
                    out.append(f)
        return out
    
    def find_dates(self, first:pd.Timestamp=None, last:pd.Timestamp=None):
        files = self.paths.inp.ls(recursive=True, include=['JD.tif'], exclude=['.xml'])
        self.times = pd.DatetimeIndex(sorted(set([pd.Timestamp(o.name[:8]) for o in files])))
        if first is not None:
            self.times = self.times[self.times>=first]
        if last is not None:
            self.times = self.times[self.times<=last]
        return self.times

    def file_time_range(self, file) -> pd.DatetimeIndex:
        t0 = pd.Timestamp(file.name[:8])
        return pd.date_range(t0, periods=monthlen(t0.year, t0.month), freq='D')
    
    def open(self, files:list) -> dict:
        times = self.file_time_range(files[0])
        data_dict = {'times': times}
        out = np.zeros((len(times), *self.region.shape))
        data = [open_tif(f) for f in files]
        data, tfm = crop(data, self.region.bbox)
        for i, time in enumerate(times):
            x = (data == time.dayofyear).astype(np.int8)
            out[i] += downsample(x, tfm, self.region.transform, self.region.shape)
        data_dict[self.bands[0]] = out
        return data_dict

    def save(self, time:pd.Timestamp, data:dict, do_compression=True):
        v = self.bands[0]
        for i, t in enumerate(data['times']):
            super().save(t, {v: data[v][i]}, do_compression=do_compression)


class AusCoverDataset(BaseDataset):
    def __init__(self, paths:InOutPath, region:Region, times:pd.DatetimeIndex=None,
                bands=['bafrac']):
        super().__init__('AusCover', paths, region, times, bands)

    def list_files(self, time:pd.Timestamp) -> list:
        out = []
        if time.year in self.times.year:
            time = pd.Timestamp(f'{time.year}-01-01')
            time_pattern = time.strftime('_%Y_')
            files = self.paths.inp.ls(recursive=True, include=['.tif', time_pattern],
                                exclude=['.xml'])
        return files
    
    def find_dates(self, first:pd.Timestamp=None, last:pd.Timestamp=None):
        files = self.paths.inp.ls(recursive=True, include=['.tif'], exclude=['.xml'])
        self.times = pd.DatetimeIndex(sorted(set([pd.Timestamp(f'{o.stem[-10:-6]}-01-01') 
                                                  for o in files])))
        if first is not None:
            self.times = self.times[self.times>=first]
        if last is not None:
            self.times = self.times[self.times<=last]
        return self.times

    def file_time_range(self, file) -> pd.DatetimeIndex:
        t0 = pd.Timestamp(f'{file.stem[-10:-6]}-01-01')
        return pd.date_range(t0, periods=12, freq='MS')
    
    def open(self, files:list) -> dict:
        times = self.file_time_range(files[0])
        data_dict = {'times': times}
        out = np.zeros((len(times), *self.region.shape))
        data = [open_tif(f) for f in files]
        data = data[0]
        crs = data.crs
        tfm = data.transform
        data = data.read(1)
        for i, time in enumerate(times):
            x = (data == time.month).astype(np.int8)
            out[i] += downsample(x, tfm, self.region.transform, 
                                 self.region.shape, src_crs=crs)
        data_dict[self.bands[0]] = out
        return data_dict

    def save(self, time:pd.Timestamp, data:dict, do_compression=True):
        v = self.bands[0]
        for i, t in enumerate(data['times']):
            super().save(t, {v: data[v][i]}, do_compression=do_compression)


class MTBSDataset(BaseDataset):
    def __init__(self, paths:InOutPath, region:Region, times:pd.DatetimeIndex=None,
                bands=['bafrac']):
        super().__init__('MTBS', paths, region, times, bands)

    def list_files(self, *args) -> list:
        files = self.paths.inp.ls(recursive=True, include=['.shp'], exclude=['.xml'])
        return files
    
    def find_dates(self, first:pd.Timestamp=None, last:pd.Timestamp=None):
        files = self.list_files()
        df = open_shp(files[0])
        self.times = pd.date_range(f'{df.Year.min()}-01-01', 
                                   f'{df.Year.max()}-12-01', freq='MS')
        if first is not None:
            self.times = self.times[self.times>=first]
        if last is not None:
            self.times = self.times[self.times<=last]
        return self.times
    
    def open(self, files:list) -> dict:
        data_dict = {'times': self.times}
        data = open_shp(files[0]).to_crs({'init': 'EPSG:4326'})
        out = np.zeros((len(self.times), *self.region.shape))
        R = Region(self.region.name, self.region.bbox, pixel_size=0.0003)
        for i, time in enumerate(self.times):
            x = data.loc[(data.Year==time.year) & (data.StartMonth==time.month)].copy()
            x_raster = rasterize(x, region=R)
            out[i] += downsample(x_raster, R.transform, self.region.transform,
                                self.region.shape)
        data_dict[self.bands[0]] = out
        return data_dict

    def save(self, time:pd.Timestamp, data:dict, do_compression=True):
        v = self.bands[0]
        for i, t in enumerate(data['times']):
            super().save(t, {v: data[v][i]}, do_compression=do_compression)
            
    def process_all(self, *args):
        self.process_one(self.times[0])


class ICNFDataset(BaseDataset):
    def __init__(self, paths:InOutPath, region:Region, times:pd.DatetimeIndex=None,
                bands=['bafrac']):
        super().__init__('ICNF', paths, region, times, bands)

    def list_files(self, *args) -> list:
        files = self.paths.inp.ls(recursive=True, include=['.shp'], exclude=['.xml'])
        return files
    
    def find_dates(self, first:pd.Timestamp=None, last:pd.Timestamp=None):
        files = self.list_files()
        df = open_shp(files[0])
        self.times = sorted(set([pd.Timestamp(f'{o[:-2]}01') 
                            for o in df.FIREDATE if o is not None]))
        if first is not None:
            self.times = self.times[self.times>=first]
        if last is not None:
            self.times = self.times[self.times<=last]
        return self.times
    
    def open(self, files:list) -> dict:
        data_dict = {'times': self.times}
        data = open_shp(files[0]).to_crs({'init': 'EPSG:4326'})
        data = data.loc[~data.FIREDATE.isna()]
        times = pd.DatetimeIndex([pd.Timestamp(o) for o in data.FIREDATE])
        data['times'] = times
        out = np.zeros((len(self.times), *self.region.shape))
        R = Region(self.region.name, self.region.bbox, pixel_size=0.0003)
        for i, time in enumerate(self.times):
            x = data.loc[(times.year==time.year) & 
                         (times.month==time.month)].copy()
            x_raster = rasterize(x, region=R)
            out[i] += downsample(x_raster, R.transform, self.region.transform,
                                self.region.shape)
        data_dict[self.bands[0]] = out
        return data_dict

    def save(self, time:pd.Timestamp, data:dict, do_compression=True):
        v = self.bands[0]
        for i, t in enumerate(data['times']):
            super().save(t, {v: data[v][i]}, do_compression=do_compression)
            
    def process_all(self, *args):
        self.process_one(self.times[0])


class Region2Tiles():
    def __init__(self, paths:InOutPath, input_name:str, target_name:str,
                 regions:list=None, bands:list=None, size=128, step=100):
        self.paths = paths
        self.input_name = input_name
        self.target_name = target_name
        self.bands = bands
        self.size = size
        self.step = step
        if regions is None:
            self.regions = [o.name for o in self.paths.inp.ls()]
        else:
            self.regions = regions

        for folder in ['images', 'masks']:
            (self.paths.out/folder).mkdir(exist_ok=True)
    
    def open(self, file, bands):
        f = sio.loadmat(file)
        return np.array([f[k] for k in bands]).transpose(1,2,0)
    
    def process_one(self, file, folder, bands):
        try:
            data = self.open(file, bands)
            rr, cc, _ = data.shape
            for c in range(0, cc-1, self.step):
                for r in range(0, rr-1, self.step):
                    img = self.crop(data, r, c)
                    if np.nansum(~np.isnan(img)) > 0:
                        self.save(img, file, r, c, folder, bands)
        except:
            warn(f'Unable to process {file}.')
    
    def process_all(self, max_workers=8, include=[]):
        for r in self.regions:
            print(f'Creating tiles for {r}')
            for i, s in enumerate([self.input_name, self.target_name]):
                files_list = self.paths.inp.ls(recursive=True, include=[*include, *['.mat', r, s]])
                folder = 'images' if s == self.input_name else 'masks'
                bands = self.bands[i]
                process_one = partial(self.process_one, folder=folder, bands=bands)
                with ThreadPoolExecutor(max_workers) as e: 
                    list(tqdm(e.map(process_one, files_list), total=len(files_list))) 

    def crop(self, im, r, c): 
        '''
        crop image into a square of size sz, 
        '''
        sz = self.size
        out_sz = (sz, sz, im.shape[-1])
        rs,cs,hs = im.shape
        tile = np.zeros(out_sz)
        if (r+sz > rs) and (c+sz > cs):
            tile[:rs-r, :cs-c, :] = im[r:, c:, :]
        elif (r+sz > rs):
            tile[:rs-r, :, :] = im[r:, c:c+sz, :]
        elif (c+sz > cs):
            tile[:, :cs-c, :] = im[r:r+sz ,c:, :]
        else:
            tile[...] = im[r:r+sz, c:c+sz, :]
        return tile

    def save(self, data, file, r, c, folder, bands):            
        sio.savemat(self.paths.out/f'{folder}/{file.stem}_{r}_{c}.mat', 
            {v: data[...,i] for i, v in enumerate(bands)}, do_compression=True)