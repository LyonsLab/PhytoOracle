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
4|[fluorescence aggregation](https://github.com/phytooracle/psii_fluorescence_aggregation)|aggregates segmentation data for each image and calculates F0, Fm, Fv, and Fv/Fm|

#### Data overview

PhytoOracle's psII pipeline requires a metadata file (`<metadata>.json`) for every compressed image file (`<image>.bin`). Each folder (one scan) contains one metadata file and 2 compressed images, one taken from a left camera and one taken from a right camera. We provide publicly-available data in the [CyVerse DataStore](https://datacommons.cyverse.org/browse/iplant/home/shared/terraref/ua-mac/raw_tars).

#### Setup Guide

- Download [CCTools](http://ccl.cse.nd.edu/software/downloadfiles.php) and extract it's contents within your HPC home path:
```
cd ~

wget http://ccl.cse.nd.edu/software/files/cctools-7.1.12-x86_64-centos7.tar.gz

tar -xvf cctools-7.1.12-x86_64-centos7.tar.gz
```       

- Clone the PhytoOracle repository within your HPC's storage space such as /xdisk:
```
git clone https://github.com/LyonsLab/PhytoOracle.git
```

- Change directory to psII:
```
cd PhytoOracle/psII/
```

#### Running on the HPC systems
##### Launch workers
- If using PBS: 
```
qsub worker_scripts/po_work_ocelote.pbs
```
- If using SLURM:
```
sbatch worker_scripts/po_work_puma.sh
```

##### Pipeline staging
- Download raw data:
```
iget -N 0 -KVPT /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/ps2Top/ps2Top-<day>.tar
```

- Download the required files:
```
iget -N 0 -PVT /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/season10_multi_latlon_geno.geojson
```

> **_NOTE:_** Replace <day> with any day you want to process. 

- Extract file contents and move the folder to the root directory:
```
tar -xvf ps2Top-<date>.tar
mv ps2Top/<date> .
```

- Run the pipeline interactively:
```
./manager_scripts/gpu_init_puma.sh

./run.sh <date>
```

- Submit the pipeline as a job (HPC only):
```
sbatch po_slurm_submit.sh <date> . <num_workers>
```

