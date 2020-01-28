#!/bin/bash -l 
email=""
auth=""
region="PI"
tstart="2017-10-27 00:00:00"
tend='2017-10-27 23:59:59'
path_save="/srv/banet/data/rawdata"

banet_viirs750_download $region "$tstart" "$tend" $email $auth $path_save