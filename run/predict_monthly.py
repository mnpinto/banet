import sys
sys.path.insert(0, '../')
import argparse
import pandas as pd
import calendar

from scripts.nrt import *
from scripts.predict import *
from scripts.util import *
Path.ls = ls
from scripts.dataset import InOutPath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--region', type=str)
    arg('--year', type=int)
    arg('--input_path', type=str)
    arg('--output_path', type=str)
    args = parser.parse_args()

    weight_files = ['/srv/banet/data/models/banetv0.20-val2017-fold0.pth',
                    '/srv/banet/data/models/banetv0.20-val2017-fold1.pth',
                    '/srv/banet/data/models/banetv0.20-val2017-fold2.pth']
    #weight_files = ['/srv/banet/run/models/banet-val2018-fold99-test.pth']
    iop = InOutPath(args.input_path, f'{args.output_path}/{args.region}')
    year, region = args.year, args.region
    times = pd.DatetimeIndex([pd.Timestamp(o.stem.split('_')[-1]) 
                              for o in (iop.inp/region).ls(include=['.mat'])])
    times = times[times.year == year]
    tstart, tend = times.min(), times.max()
    month_start = (tstart + pd.Timedelta(days=31)).month
    for m in range(month_start, tend.month):
        print(f'Generating maps for {calendar.month_name[m]} {year}:')
        t = pd.Timestamp(f'{year}-{m}-01')
        predict_month(iop, t, weight_files, region)