*****************************************************
Running the StereoTopRGB Pipeline for Plant Area Data
*****************************************************

This pipeline extracts plant area data from image files. This guide provides demo data you can use follow along with and ensure the pipeline is functional.

Pipeline Overview
=================

StereoTopRGB currently uses 4 different programs for the analytical pipeline:

.. list-table::
   :header-rows: 1
   
   * - Program
     - Function
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

.. note::
   
   At this point, we assume that the interactive "manager" and "worker" nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

**Retrieve data**

Navigate to your RGB directory, download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/StereoTopRGB
   iget -rKVP /iplant/home/shared/terraref/ua-mac/raw_tars/demo_data/Lettuce/StereoTopRGB_demo.tar
   tar -xvf StereoTopRGB_demo.tar

Data from the Gantry can be found within :code:`/iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTopRGB/<scan_date>.tar`

**Retrieve correction file**

Dowload the coordiate correction :code:`.csv` file:

.. code::

   iget -rKVP /iplant/home/emmanuelgonzalez/Ariyan_ortho_attempt_4/2020-01-08_coordinates_CORRECTED_4-16-2020.csv

.. note::
   
   The :code:`.csv` file will soon be moved to a shared directory; this will change accordingly.
   
**Edit scripts**

+ :code:`process_one_set.sh`

  Find your current working directory using the command :code:`pwd`
  Open :code:`process_one_set.sh` and paste the output from :code:`pwd` into line 14. It should look something like this:

  .. code:: 

    HPC_PATH="xdisk/group_folder/personal_folder/PhytoOracle/StereoTopRGB/"

+ :code:`entrypoint.sh`

  + In line 1, specify the :code:`<scan_date>` folder you want to process. For our purposes this will look like:

    .. code:: 

      phython3 gen_files_list.py StereoTopRGB_demo > raw_data_files.json

  + In lines 7 and 11, specify the location of CCTools:

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/jx2json

    and

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/makeflow

**Run pipeline**

Begin processing using:

.. code::

  ./entrypoint.sh

.. note::
   
   This may return a notice with a "FATAL" error. This happens as the pipeline waits for a connection to DockerHub, which takes some time. Usually, the system will fail quickly if there is an issue.

   If the pipeline fails, check to make sure you have a "/" concluding line 14 of :code:`process_one_set.sh`. This is one of the most common errors and is necessary to connect the program scripts to the HPC.
