#!/bin/bash -l
python=$HOME/anaconda3/envs/web/bin/python # The download scripts requires SOAPpy library that at the present only works with python2
project_path=/srv/banet

email=$EMAIL # Email used for registation on ladsweb site
auth=$AUTH # Authentication obtained on ladsweb site
region=PI # Region must be defined in data/regions/R_PI.json
path=/srv/banet/data/rawdata/$region # Where to save the data
time_start=2018-09-01 
time_end=2018-09-30 # The model runs with 64-day sequences, to predict a single month make sure to give enough days before and after.

$python $project_path/scripts/ladsweb.py $email $auth $path $region $time_start $time_end