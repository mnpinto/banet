# Release notes

<!-- do not remove -->
## 0.6.7

### Minor update
- Added parameter `times` to `RunManager.preprocess_dataset`. This argument is passed to ViirsDataset and can be used to filter specific dates for preprocessing.


## 0.6.6

### Fixed active fire download links for NRT and updated requirements
- The urls for downloading near-real-time viirs active fires changed recently and the function `nrt.RunManager.update_hotspots` was updated accordingly.
- proj, geos, cartopy added as requirements in the `settings.ini`.
- fastai was fixed to version 2.2.7 in the `settings.ini`.
- pyhdf was fixed to version 0.10.2 in the `settings.ini` to address an error occuring with the newest version.

## 0.6.5

### Improved predict_time and get_preds for historical data
- Added argument check_file (default=False) in `historical.RunManager.get_preds`. When True the code will run over existing netcdf output files and check if there is missing data. If missing data is found, the prediction step runs. 

## 0.6.4

### Improved predict_time function
- Monthly output are now saved in netcdf format. Each tile is written directly to the netcdf file instead of holding it in memory until the complete region is mapped.


## 0.6.3

### Bug fixes
- Fix save option in `BaseDataset.process_one` and delete data before calling `gc.collect()`;

### Improved predict_time function
- Compute every tile for each month instead of every month for each tile;
- Save montly outputs and instead of holding all data in memory;
- Monthly outputs are combined at the end;
- Existing months (already computed) will be skipped (to recalculate the output files need to be manually deleted).

## 0.6.2

### Improved pre-processing
- Pre-process by tiles to avoid memory error in large regions or low memory machines: e.g.: `RunManager.preprocess_dataset(max_size=4000)`;
- Pre-process in parallel: When using pyhdf to read the hdf data (now the default instead of netCDF4), the `max_workers` argument can be set to more than 1, e.g.: `RunManager.preprocess_dataset(max_size=4000, max_workers=8)`;
- When max_size is smaller than the region size (tiles are used) the result of MergeTiles may differ in days where several tiles are combined since they are combined based on the average SatelliteZenithAngle. In the FRP band the hotspots may also change position to a neighboring pixel in this case due to small difference in the coordinates during the interpolation of the active fires.
- The smaller the max_size the larger the file size due to less compression. 

### Region class shape
- Region class in geo.py now accepts a shape tuple as argument. When shape=None (default) it will be calculated from the bounding box and the pixel size has before. 