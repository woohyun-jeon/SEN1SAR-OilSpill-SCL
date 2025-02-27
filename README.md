# Oil spill detection with dual-polarimetric Sentinel-1 SAR images
This repository contains the implementation of oil spill detection using dual-polarimetric Sentinel-1 SAR images.

## Prerequisites
* python >= 3.10
* torch >= 2.2.0
* torchvision >= 0.17.0

## Usage
1) Clone the repository and install the required dependencies with the following command:
```
$ git clone https://github.com/woohyun-jeon/SEN1SAR-OilSpill-SCL.git
$ cd SEN1SAR-OilSpill-SCL
$ pip install -r requirements.txt
```
2) Download datasets from here:
https://drive.google.com/drive/folders/1aAYPKh5zh7SP9RD1-c4-1Di44KdWJ-5x?usp=drive_link

The directory structure should be as follows:
```
  image/  
    0000.tif
    0001.tif
    ...
  label/
    0000.tif
    0001.tif
    ... 
  train.txt
  valid.txt
  test.txt
```
* It is important to mention that "sup_path" argument in "configs.yaml" file, denoting the parent directory of image & label path, should be properly adjusted.
* Plus, "out_path" argument, indicating output directory of prediction and model files, should be properly adjusted.

3) Run main.py code with the following command:
```
$ cd src
$ python main.py
```