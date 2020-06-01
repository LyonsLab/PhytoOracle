*****************************************************
Running the StereoTopRGB Pipeline for Plant Area Data
*****************************************************

StereoTopRGB: This pipeline extracts plant area data from image files using data transformers.

Pipeline Overview
=================

StereoTopRGB currently uses 4 different programs for the analytical pipeline:

.. list-table::
   :header-rows: 1
   
   * - Program
     - Purpose
     - Input
     - Output
   * - `cleanmetadata <https://github.com/AgPipeline/moving-transformer-cleanmetadata>`_
     - Cleans gantry generated metadata
     - :code:`metadata.json`
     - :code:`metadata_cleaned.json`
   * - `bin2tif <https://github.com/AgPipeline/moving-transformer-bin2tif>`_
     - Converts bin compressed files to tif
     - :code:`left_image.bin`, :code:`right_image.bin`
     - :code:`left_image.tif`, :code:`right_image.tif`
   * - `gistools <https://github.com/uacic/docker-builds/tree/master/gistools>`_
     - Corrects GPS coordinates
     - :code:`coordinates_CORRECTED_<date>.csv`
     - :code:`corrected_coordinates_left_image.tif`, :code:`corrected_coordinates_right_image.tif`
   * - `plotclip <https://github.com/AgPipeline/transformer-plotclip>`_ 
     - Clips GeoTIFF or LAS files according to plots
     - :code:`corrected_coordinates_left_image.tif`, :code:`corrected_coordinates_right_image.tif`
     - :code:`clipped_left_image.tif`, :code:`clipped_right_image.tif`

Running the Pipeline 
====================

The pipeline runs in the following manner:

1. Request an interactive node on the HPC
2. Request worker nodes
3. Clone the Git within the HPC
4. Retrieve data from desired scandate
5. Edit scripts and run the pipeline 

.. note::
   
   At this point we assume that the interactive and worker nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   Otherwise follow the guide `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

**4. Retrieve data from desired scandate**

Download the data from the CyVerse DataStore with iRODS commands using:

.. code::

   iget -rKVP /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTopRGB/<scan_date>.tar


Replace :code:`<scan_date>` with any day you want to process. Un-tar and move the folder to the stereoTopRGB directory.

.. code::

   tar -xvf <scan_date>.tar
   mv ./stereoTopRGB/<scan_date> ./

Dowload the coordiate correction :code:`.csv` file.

.. note::
   
   This file will be changing to the shared directory

.. code::

   iget -rKVP /iplant/home/emmanuelgonzalez/Ariyan_ortho_attempt_4/2020-01-08_coordinates_CORRECTED_4-16-2020.csv

   
**5. Edit scripts and run the pipeline**

1. Copy your current working directory (:code:`pwd`) and edit the :code:`HPC_PATH="<pwd>"` on line 14 in the :code:`process_one_set.sh` file.
2. Edit your :code:`entrypoint.sh` on line 4 to reflect the :code:`<scan_date>` folder you want to process.
3. Also in :code:`entrypoint.sh` ensure that on lines 7 and 11 the :code:`path` to CCTools is correct.
4. Once everything is edited, run the pipeline with :code:`./entrypoint.sh`.

Demo Data
=========

Demo data for the StereoTopRGB pipeline can be downloaded through iRODS from the CyVerse DataStore with:

.. code::

   /iplant/home/shared/terraref/ua-mac/raw_tars/demo_data/Lettuce/StereoTopRGB_demo.tar