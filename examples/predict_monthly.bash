#!/bin/bash -l
input_path=/srv/banet/data/procdata
output_path=/srv/banet/data/monthly
region=PI
year=2017

banet_predict_monthly $region $input_path $output_path $year