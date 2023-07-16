# BA-Net: A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images
> Over the past decades, methods for burned areas mapping and dating from remote sensing imagery have been the object of extensive research. The limitations of current methods, together with the heavy pre-processing of input data they require, make them difficult to improve or apply to different satellite sensors. Here, we explore a deep learning approach based on daily sequences of multi-spectral images, as a promising and flexible technique that can be applicable to observations with various spatial and spectral resolutions. We test the proposed model for five regions around the globe using input data from VIIRS 750 m bands resampled to a 0.01º spatial resolution grid. The derived burned areas are validated against higher resolution reference maps and compared with the MCD64A1 Collection 6 and FireCCI51 global burned area datasets. We show that the proposed methodology achieves competitive results in the task of burned areas mapping, despite using lower spatial resolution observations than the two global datasets. Furthermore, we improve the task of burned areas dating for the considered regions of study when compared with state-of-the-art products. We also show that our model can be used to map burned areas for low burned fraction levels and that it can operate in near-real-time, converging to the final solution in only a few days. The obtained results are a strong indication of the advantage of deep learning approaches for the problem of mapping and dating of burned areas and provide several routes for future research.


![Graphical Abstract](nbs/images/graphical_abstract.jpg)

## Install

libhdf4-dev is required for pyhdf to read .hdf4 files.

`sudo apt install -y libhdf4-dev`

`pip install banet`

## Dataset
The dataset used to train the model and the pretrained weights are available at https://drive.google.com/drive/folders/1142CCdtyekXHc60gtIgmIYzHdv8lMHqN?usp=sharing. Notice that the size of the full dataset is about 160 GB. You can, however, donwload individual regions in case you want to test with a smaller dataset.

## Generate predictions for a new region in 5 steps
The procedure to generate predictions for a new region or for a different period of the existing regions is straightforward.

* **Step 1.** Define a `.json` file with region name, bounding boxes and spatial resolution. Region definition files are by default on `data/regions` and should be named as `R_{name}.json`, where name is the name you give to the region inside the file. For example for Iberian Peninsula region the file `data/regions/R_PI.json` contains the following: `{"name": "PI", "bbox": [-10, 36, 5, 44], "pixel_size": 0.01}`.

* **Step 2.** To download the reflectance data the command line script `banet_viirs750_download` can be used. However, in order to use it, you need to first register at the website (https://ladsweb.modaps.eosdis.nasa.gov/) and generate an authentication token. 

* **Step 3.** Next you need to download the VIIRS active fire data for the region you seletected. This procedure is manual but you should be able to request data for the entire temporal window in one go. To do that go to https://firms.modaps.eosdis.nasa.gov/download/, select `Create New Request`, select the region based on your region bounding box, for fire data source select `VIIRS`, then select the date range and finally `.csv` for the file format. You should receive an email with the confirmation and later another with the link to download the data. If not go back to the download page enter the email you used for the request and choose `Check Request Status`. If it is completed the download link will appear. Once you have the file place it in `data/hotspots` and name it `hotspots{name}.csv` where name is the name of the region as in the `.json` file defined in Step 1.

* **Step 4.** Now that you have all the data, you can use the command line script `banet_create_dataset`.

* **Step 5.** Finally you can use the `banet_predict_monthly` command line script to generate the model outputs.

**Note:** Some examples of usage for the command line tools are available in the documentation. 


## Train the model from scratch
To train the model you need a dataset of image tiles and the respective targets. The data for the 5 study regions is available for download at https://drive.google.com/drive/folders/1142CCdtyekXHc60gtIgmIYzHdv8lMHqN?usp=sharing. In case you want to train a model on different regions the procedure to collect VIIRS data is described on **Generate predictions for a new region in 5 steps** above. 

* **Step 1.** In case you opted for new regions you need to collect MCD64A1 collection 6 burned areas to use as targets. MCD64A1 collection 6 data can be downloaded from `ftp://ba1.geog.umd.edu/` server (description and credentials available on http://modis-fire.umd.edu/files/MODIS_C6_BA_User_Guide_1.2.pdf section 4). Once you log into the server go to `Collection 6/TIFF` folder and download data for the window or windows covering your region (Figure 2 on the user guide shows the delineation of the windows).

* **Step 2.** Once you have the data you can generate a dataset with `banet_create_dataset` and `banet_dataset2tiles` command line tools. If you downloaded the dataset already in `.mat` files format provided in the url above then you just use the `banet_dataset2tiles`.

* **Step 3.** To train the model the `banet_train_model` command line tool is provided. To use sequences of 64 days with 128x128 size tiles and batch size of 1, you need a 8 GB GPU. You can try reducing the sequence length if your GPU has less memory or increase the batch size otherwise. 

**Note:** Some examples of usage for the command line tools are available in the documentation.

## Fine-tune the model for a specific region or for other data source (transfer learning)
It is possible to fine-tune the trained model weights to a specific region or using a different source for the input data (e.g., data from VIIRS 375m bands or data from another satellite). The easiest way to include a new dataset is to write a `Dataset` class similar to `Viirs750Dataset` or the other dataset classes on `banet.data`. Once you have the new dataset you can follow the **Train the model from scratch** guideline and the only change you need to make is to make sure you load the pretrained weights before starting to train.

## Troubleshooting
If you find any bug or have any question regarding the code or applications of BA-Net you can navigate to `Issues` tab and create a new Issue describing your problem. I will try to answer as soon as possible.

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
