Data Types & Descriptions
-------------------------

.. list-table::
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Gantry Sensor
     - Input Data Type
     - Input Data Size / Day
     - Output Data Type
     - Output Data Size / Day
   * - StereoTopRGB
     - L,R-Binary images & metadata json 
     - 143 GB (9355 datasets)
     - Soil-mask Tifs & canopy-cover.csv
     - 23 GB 
   * - Scanner3D
     - Ply files & metadata json
     - 108 GB (238 datasets)
     - Output Data Type
     - Output Data Size / Day
   * - PSII 
     - Binary images & metadata json
     - 70-90GB / Night
     - Fluorescence-aggregates.csv 
     - 30-35MB / Night


**Data Availability**

Raw Input data is avaialable via CyVerse Data Store under the following folder and can be accessed using iRODS:

.. code::

   /iplant/home/elyons/ACIC/2019-final-gantry-data:
   C- /iplant/home/elyons/ACIC/2019-final-gantry-data/ps2Top
   C- /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop
   C- /iplant/home/elyons/ACIC/2019-final-gantry-data/stereoTop

.. code::

   /iplant/home/shared/terraref/ua-mac/raw_tars/season_10:
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/EnvironmentLogger
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/SWIR
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/co2Sensor
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/cropCircle
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/flirIrCamera
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/ndviSensor
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/priSensor
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/ps2Top
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/scanner3DTop
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/stereoTop

   
**Compressed tar files for Data Staging**
  
- Raw data needed for the workflow can be downloaded onto a virtual machine using iRODS. 
- A compressed tar file for each of the sensors is available from the links below: 

  - StereoRGB: `/iplant/home/elyons/ACIC/2019-final-gantry-data/stereoTop/stereoTop-2018-05-15.tar.gz`
  
  - Scanner3D: `/iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/scanner3DTop-raw.tar.gz`
  
  - PSII: `/iplant/home/elyons/ACIC/2019-final-gantry-data/ps2Top/PS2-2018-02-12.tar.gz`
  
  
Extractor Containers
--------------------

Click `here <https://github.com/uacic/PhytoOracle/blob/master/docs/containers.md>`_ for a list of available extractor containers.
