# Release notes

<!-- do not remove -->

## 0.6.2

### Improved pre-processing
- Pre-process by tiles to avoid memory error in large regions or low memory machines: e.g.: `RunManager.preprocess_dataset(max_size=4000)`;
- Pre-process in parallel: When using pyhdf to read the hdf data (now the default instead of netCDF4), the `max_workers` argument can be set to more than 1, e.g.: `RunManager.preprocess_dataset(max_size=4000, max_workers=8)`;
- When max_size is smaller than the region size (tiles are used) the result of MergeTiles may differ in days where several tiles are combined since they are combined based on the average SatelliteZenithAngle. In the FRP band the hotspots may also change position to a neighboring pixel in this case due to small difference in the coordinates during the interpolation of the active fires.
- The smaller the max_size the larger the file size due to less compression. 

### Region class shape
- Region class in geo.py now accepts a shape tuple as argument. When shape=None (default) it will be calculated from the bounding box and the pixel size has before. 