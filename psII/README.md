# PhytoOracle's PSII Pipeline

#### Outline

Welcome to PhytoOracle's psII pipeline! This pipeline uses the data transformers from the [PhytoOracle group](https://github.com/phytooracle) to extract chlorophyll fluorescence data. The pipeline is avaiable for either HPC (High Performance Computing) systems or cloud based systems.

#### Transformers used

PSII currently uses 3 different transformers for data conversion:

|Order|Transformer|Process
|:-:|:-:|:-:|
1|[bin2tif](https://github.com/phytooracle/psii_bin_to_tif)|Converts bin files to GeoTIFFs|
2|[plotclip](https://github.com/phytooracle/rgb_flir_plot_clip_geojson)|Clips GeoTIFFs to agricultural plot boundaries|
3|[fluorescence segmentation](https://github.com/phytooracle/psii_segmentation)|Segments pixels given a validated set of thresholds|
4|[fluorescence aggregation](https://github.com/phytooracle/psii_fluorescence_aggregation)|Aggregates segmentation data for each image and calculates F0, Fm, Fv, and Fv/Fm|

#### Data overview

PhytoOracle's psII pipeline requires a metadata file (`<metadata>.json`) for every compressed image file (`<image>.bin`). Each folder (one scan) contains one metadata file and 2 compressed images, one taken from a left camera and one taken from a right camera. We provide publicly-available data in the [CyVerse DataStore](https://datacommons.cyverse.org/browse/iplant/home/shared/terraref/ua-mac/raw_tars).

#### Setup Guide

- Change directory to psII:
```
cd ~/PhytoOracle/psII/
```

#### Running PhytoOracle on Atmosphere VM
##### Launch workers
- Open a new terminal window and run:
```
./worker_scripts/po_worker.sh
```

##### Process data
- Run the pipeline:
```
./run.sh <date>
```
