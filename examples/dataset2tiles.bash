#!/bin/bash -l
input_path=/srv/banet_nbdev/data/procdata
output_path=/srv/banet_nbdev/data/tiles/train
region=PI

banet_dataset2tiles $region $input_path $output_path