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
     - -
   * - PSII 
     - Binary images & metadata json
     - 70-90GB / Night
     - Fluorescence-aggregates.csv 
     - 30-35MB / Night
   * - FlirIr 
     - Binary image & metadata json
     - 5-6GB (9270 datasets)
     - meantemp.csv 
     - ~7MB csv / scan

**Data Availability**

Raw Input data is avaialable via CyVerse Data Store under the following folder and can be accessed using iRODS:

.. code::

   /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020:
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/flirIrCamera
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/ps2Top
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/scanner3DTop
   C- /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTop
  
Transformer Containers
--------------------

Click `here <https://github.com/uacic/PhytoOracle/blob/master/docs/containers.md>`_ for a list of available extractor containers.
