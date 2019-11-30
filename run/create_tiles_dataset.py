import sys, os
sys.path.insert(0, '../')
import argparse
from pathlib import Path
import numpy as np
import shutil

from scripts.dataset import *
from scripts.processors import *
from scripts.geo import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--region', type=str)
    arg('--year', type=int, default=None)
    arg('--input_path', type=str)
    arg('--output_path', type=str)
    args = parser.parse_args()

    iop = InOutPath(args.input_path, args.output_path)
    td = Region2Tiles(iop, 'VIIRS750', 'MCD64A1C6', regions=[args.region],
                      bands=[['Red', 'NIR', 'MIR', 'FRP'], ['bafrac']])

    if args.year is None:
        td.process_all()
    else:
        td.process_all(include=[f'_{args.year}'])