#!/bin/bash -l
python=$HOME/anaconda3/envs/fastai_v1_2/bin/python
inpath=/srv/banet/data/tiles/train

# val_year sets the year used for validation
# r_fold is the number/name of the fold used when saving the model weights

$python ../scripts/train.py --val_year 2018 --r_fold 99 --input_path $inpath

# To finetune using pretrained weights you can use --pretrained_weights arg
# pretrained_weights=/srv/banet/models/banetv0.20-val2017-fold0.pth
# $python ../scripts/train.py --val_year 2018 --r_fold 99 --input_path $inpath --pretrained_weights $pretrained_weights 