# BA-Net: A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images
> Over the past decades, methods for burned areas mapping and dating from remote sensing imagery have been the object of extensive research. The limitations of current methods, together with the heavy pre-processing of input data they require, make them difficult to improve or apply to different satellite sensors. Here, we explore a deep learning approach based on daily sequences of multi-spectral images, as a promising and flexible technique that can be applicable to observations with various spatial and spectral resolutions. We test the proposed model for five regions around the globe using input data from VIIRS 750 m bands resampled to a 0.01º spatial resolution grid. The derived burned areas are validated against higher resolution reference maps and compared with the MCD64A1 Collection 6 and FireCCI51 global burned area datasets. We show that the proposed methodology achieves competitive results in the task of burned areas mapping, despite using lower spatial resolution observations than the two global datasets. Furthermore, we improve the task of burned areas dating for the considered regions of study when compared with state-of-the-art products. We also show that our model can be used to map burned areas for low burned fraction levels and that it can operate in near-real-time, converging to the final solution in only a few days. The obtained results are a strong indication of the advantage of deep learning approaches for the problem of mapping and dating of burned areas and provide several routes for future research.


![Graphical Abstract](nbs/images/graphical_abstract.jpg)

## Install

libhdf4-dev is required for pyhdf to read .hdf4 files.

`sudo apt install -y libhdf4-dev`

`conda install -c conda-forge pykdtree pyresample`

`pip install banet`

## Setup ladsweb
To create an account and an authentication token visit ladsweb website and save the following config file with your email and key at `~/.ladsweb`.
```bash
{
    "url"   : "https://ladsweb.modaps.eosdis.nasa.gov",
    "key"   : "",
    "email" : ""
}
```

## How to use
Example of near-real-time run for a region centred in Portugal.

```python
from banet.core import *
from banet.geo import *
from banet.nrt import *

region = 'PT'
hotspots_region = 'Europe'
paths = ProjectPath('PT')
weight_files = ['banetv0.20-val2017-fold0.pth']
Region(region, [-10, 36, -6, 44], 0.001).export(paths.config/f'R_{region}.json')
manager = RunManager(paths, region, product='VIIRS375', time='yesterday')

manager.update_hotspots(hotspots_region)
manager.download_viirs()
manager.preprocess_dataset()
manager.get_preds(weight_files, threshold=0.01, max_size=2000)
manager.postprocess(filename=f'ba_{manager.time.strftime("%Y%m%d")}', threshold=0.5, area_epsg=3763)
```

Note that `manager.update_hotspots()` will only download active fires data from the past 7 days. You can download active fires data for any period and region at https://firms.modaps.eosdis.nasa.gov/download/.

After successfully running `manager.preprocess_dataset()` the contents of `paths.ladsweb` can be removed.


## Citation
```bibtex
@article{pinto2020banet,
  title={A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images},
  author={Pinto, Miguel M and Libonati, Renata and Trigo, Ricardo M and Trigo, Isabel F and DaCamara, Carlos C},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={160},
  pages={260--274},
  year={2020},
  publisher={Elsevier}
}
```
