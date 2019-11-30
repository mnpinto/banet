#!/bin/bash -l
python=$HOME/anaconda3/envs/fastai_v1_2/bin/python
viirs_inpath=/srv/banet/data/rawdata
mcd64_inpath=/srv/mcd64
cci51_inpath=/srv/BA_validation/data/FireCCI51
outpath=/srv/banet/data/procdata
act_fires_path=/srv/banet/data/hotspots
region=PI

# Create dataset only for VIIRS
# $python proc_dataset.py $region --viirs_inpath $viirs_inpath --outpath $outpath \
#                         --act_fires_path $act_fires_path --year=2017

# Create dataset for VIIRS and MCD64A1C6
$python proc_dataset.py $region --viirs_inpath $viirs_inpath --mcd64_inpath $mcd64_inpath --outpath $outpath \
                        --act_fires_path $act_fires_path --year=2018

# Create dataset for VIIRS, MCD64A1 and FireCCI51 data.
# $python proc_dataset.py $region --viirs_inpath $viirs_inpath \
#                        --mcd64_inpath $mcd64_inpath --cci51_inpath $cci51_inpath \
#                        --outpath $outpath --act_fires_path $act_fires_path 

# Iterate over several regions
#for region in CA PT BR MZ AU; do  
# $python proc_dataset.py $region --viirs_inpath $viirs_inpath \
#                         --mcd64_inpath $mcd64_inpath --cci51_inpath $cci51_inpath \
#                         --outpath $outpath --act_fires_path $act_fires_path 
#done
