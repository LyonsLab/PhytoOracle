*****************************************************
Running the StereoTopRGB Pipeline for Plant Area Data
*****************************************************

StereoTopRGB: This pipeline extracts plant area data from image files using data transformers.

Transformers Used
=================

StereoTopRGB currently uses 3 different transformers for data conversion:

.. list-table::
   :header-rows: 1
   
   * - Transformer
     - Process
   * - `cleanmetadata <https://github.com/AgPipeline/moving-transformer-cleanmetadata>`_
     - Cleans gantry generated metadata
   * - `bin2tif <https://github.com/AgPipeline/moving-transformer-bin2tif>`_
     - Converts bin compressed files to tif
   * - `gistools <https://github.com/uacic/docker-builds/tree/master/gistools>`_
     - Corrects GPS coordinates
   * - `plotclip <https://github.com/AgPipeline/transformer-plotclip>`_ 
     - Clips GeoTIFF or LAS files according to plots


Data Overview
=============

Each scan folder should contain 3 files: 2 compressed images(:code:`.bin`) for the left and right cameras, and their corresponding metadata(:code:`.json`). Data pulled from the `CyVerse DataStore <https://cyverse.org/data-store>`_ should be organized in this manner.

Running the Pipeline 
====================

The pipeline runs in the following manner:

1. Request an interactive node on the HPC
2. Request worker nodes
3. Clone the Git within the HPC
4. Retrieve data from desired scandate
5. Edit scripts and run the pipeline 

.. note::
   To launch on the HPC (steps 1-3) follow the guide `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_. This page will continue from step 4 beyond.

**4. Retrieve data from desired scandate**

At this point your worker nodes should already be running and you should be in your StereoTopRGB directory within your interactive node. Download the data that you need using:

.. code::

   iget -rKVP /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTopRGB/<scan_date>.tar


Replace :code:`<scan_date>` with any day you want to process. Un-tar and move the folder to the stereoTopRGB directory.

.. code::

   tar -xvf <scan_date>.tar
   mv ./stereoTopRGB/<scan_date> ./

Dowload the coordiate correction :code:`.csv` file.

.. code::

   iget -rKVP /iplant/home/emmanuelgonzalez/Ariyan_ortho_attempt_4/2020-01-08_coordinates_CORRECTED_4-16-2020.csv

**5. Edit scripts and run the pipeline**

1. Copy your current working directory (:code:`pwd`) and edit the :code:`HPC_PATH="<pwd>"` on line 14 in the :code:`process_one_set.sh` file.
2. Edit your :code:`entrypoint.sh` on line 4 to reflect the :code:`<scan_date>` folder you want to process.
3. Also in :code:`entrypoint.sh` ensure that on lines 7 and 11 the :code:`path` to CCTools is correct.
4. Once everything is edited, run the pipeline with :code:`./entrypoint.sh`.