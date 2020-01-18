#!/bin/bash -l
viirs_path=/srv/banet/data/rawdata
mcd64_path=/srv/mcd64
cci51_path=/srv/BA_validation/data/FireCCI51
save_path=/srv/banet_nbdev/data/procdata
fires_path=/srv/banet/data/hotspots
region=PI

# Create dataset only for VIIRS
# banet_create_dataset $region $viirs_path $fires_path $save_path --year=2017

# Create dataset for VIIRS and MCD64A1C6
# banet_create_dataset $region $viirs_path $fires_path $save_path \
#                       --mcd64_path $mcd64_path --year=2017

# Create dataset for VIIRS, MCD64A1C6 and FireCCI51 data.
banet_create_dataset $region $viirs_path $fires_path $save_path \
                     --mcd64_path $mcd64_path --cci51_path $cci51_path --year=2017
