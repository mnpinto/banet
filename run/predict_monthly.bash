#!/bin/bash -l
python=$HOME/anaconda3/envs/fastai_v1_2/bin/python
inpath=/srv/banet/data/procdata
outpath=/srv/banet/data/monthly
region=PI
year=2017

$python predict_monthly.py --region=$region --year=$year \
                           --input_path=$inpath --output_path=$outpath