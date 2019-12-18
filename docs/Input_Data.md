Data Types & Descriptions
-------------------------

| Gantry Sensor | Input Data Type | Input Data Size / Day |
| ------------- | --------------- | --------------------- |
| stereoRGB | - Left Binary Image   | 143GB (9355 Datasets) |
| | - Right Binary Image  |           |
| | - Metadata json       |           |   

Data Availability 
-----------------

Raw Input data is avaialable via CyVerse Data Store under the following folder and can be accessed using iRODS:
```
/iplant/home/elyons/ACIC/2019-final-gantry-data:
  C- /iplant/home/elyons/ACIC/2019-final-gantry-data/ps2Top
  C- /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop
  C- /iplant/home/elyons/ACIC/2019-final-gantry-data/stereoTop
  ```
  
Data Staging
------------
  
- Raw data needed for the workflow can be downloaded onto a virtual machine using iRODS. 
- A compressed tar file for each of the sensors is available from the links below: 
  - StereoRGB: `/iplant/home/elyons/ACIC/2019-final-gantry-data/stereoTop/stereoTop-2018-05-15.tar.gz`
  - Scanner3D: `/iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/scanner3DTop-raw.tar.gz`
  - PSII: `/iplant/home/elyons/ACIC/2019-final-gantry-data/ps2Top/PS2-2018-02-12.tar.gz`
