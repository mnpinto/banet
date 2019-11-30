import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, '../')

from scripts.dataset import *
from scripts.processors import *

bands =  ['Reflectance_M5', 'Reflectance_M7', 'Reflectance_M10', 'Radiance_M12', 
          'Radiance_M15', 'SolarZenithAngle', 'SatelliteZenithAngle']
          
def setup(region, viirs_inpath, mcd64_inpath, cci51_inpath, outpath, act_fires_path, bands, year):
    paths = InOutPath(f'{viirs_inpath}/{region}', f'{outpath}/{region}')
    R = Region.load(f'../data/regions/R_{region}.json')

    # VIIRS750
    print('VIIRS750')
    viirs = Viirs750Dataset(paths, R, bands=bands)
    viirs.filter_times(year)
    merge_tiles = MergeTiles('SatelliteZenithAngle')
    mir_calc = MirCalc('SolarZenithAngle', 'Radiance_M12', 'Radiance_M15')
    rename = BandsRename(['Reflectance_M5', 'Reflectance_M7'], ['Red', 'NIR'])
    bfilter = BandsFilter(['Red', 'NIR', 'MIR'])
    act_fires = ActiveFires(f'{act_fires_path}/hotspots{R.name}.csv')
    viirs.process_all(proc_funcs=[merge_tiles, mir_calc, rename, bfilter, act_fires])

    # MCD64A1C6
    if mcd64_inpath is not None:
        print('MCD64A1C6')
        paths.inp = Path(mcd64_inpath)
        mcd = MCD64Dataset(paths, R)
        mcd.match_times(viirs)
        mcd.process_all()

    # FireCCI51
    if cci51_inpath is not None:
        print('FireCCI51')
        paths.inp = Path(cci51_inpath)
        cci51 = FireCCI51Dataset(paths, R)
        cci51.match_times(viirs)
        cci51.process_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('region', type=str)
    arg('--viirs_inpath', type=str)
    arg('--mcd64_inpath', type=str, default=None)
    arg('--cci51_inpath', type=str, default=None)
    arg('--outpath', type=str)
    arg('--act_fires_path', type=str)
    arg('--bands', type=list, default=bands)
    arg('--year', type=int, default=None)
    args = parser.parse_args()

    # Call getData with args
    print(f'Processing data for {args.region}')
    setup(args.region, args.viirs_inpath, args.mcd64_inpath, args.cci51_inpath,
          args.outpath, args.act_fires_path, args.bands, args.year)