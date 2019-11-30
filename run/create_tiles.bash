#!/bin/bash -l
python=$HOME/anaconda3/envs/fastai_v1_2/bin/python
inpath=/srv/banet/data/procdata
outpath=/srv/banet/data/tiles/train
region=PI

$python create_tiles_dataset.py --region $region --input_path $inpath --output_path $outpath