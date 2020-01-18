#!/bin/bash -l
val_year=2018
fold_name=99
input_path=/srv/banet/data/tiles/train
output_path=/srv/banet_nbdev/data/models

banet_train_model $val_year $fold_name $input_path $output_path --n_epochs 1