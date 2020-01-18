#!/bin/bash -l
input_path=/srv/banet_nbdev/data/procdata
output_path=/srv/banet_nbdev/data/monthly
region=PI
year=2017

banet_predict_monthly $region $input_path $output_path $year