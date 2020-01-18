# BA-Net: A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images
> Over the past decades, methods for burned areas mapping and dating from remote sensing imagery have been the object of extensive research. The limitations of current methods, together with the heavy pre-processing of input data they require, make them difficult to improve or apply to different satellite sensors. Here, we explore a deep learning approach based on daily sequences of multi-spectral images, as a promising and flexible technique that can be applicable to observations with various spatial and spectral resolutions. We test the proposed model for five regions around the globe using input data from VIIRS 750 m bands resampled to a 0.01º spatial resolution grid. The derived burned areas are validated against higher resolution reference maps and compared with the MCD64A1 Collection 6 and FireCCI51 global burned area datasets. We show that the proposed methodology achieves competitive results in the task of burned areas mapping, despite using lower spatial resolution observations than the two global datasets. Furthermore, we improve the task of burned areas dating for the considered regions of study when compared with state-of-the-art products. We also show that our model can be used to map burned areas for low burned fraction levels and that it can operate in near-real-time, converging to the final solution in only a few days. The obtained results are a strong indication of the advantage of deep learning approaches for the problem of mapping and dating of burned areas and provide several routes for future research.


## Install

`pip install banet`

## How to use
