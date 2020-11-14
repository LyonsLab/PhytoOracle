***************************************************************
Running the StereoTopRGB Pipeline for Detecting Plant Area Data
***************************************************************

This pipeline extracts plant area data from image files. This guide provides demo data you can use follow along with and ensure the pipeline is functional. Before starting, change to :code:`alpha` branch with :code:`git checkout alpha`.

Pipeline Overview
=================

StereoTopRGB currently uses 8 different programs for the analytical pipeline:

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
     - Converts bin compressed files to geotiff
     - :code:`image.bin`
     - :code:`image.tif`
   * - `collect_gps <https://github.com/emmanuelgonz/collect_gps>`_
     - Collects GPS coordinates from all geotiff files
     - :code:`image.tif`
     - :code:`collected_coordinates.csv`
   * - MEGASTITCH (Zarei, unpublished)
     - Finds best possible coordinates of all geotiffs
     - :code:`collected_coordinates.csv`
     - :code:`corrected_coordinates.csv`
   * - `replace_gps <https://github.com/emmanuelgonz/edit_gps>`_ 
     - Applies corrected GPS coordinates to images
     - :code:`corrected_coordinates.csv`, :code:`image.tif`
     - :code:`corrected_image.tif`
   * - `plotclip <https://github.com/emmanuelgonz/plotclip_shp>`_ 
     - Clips geotiffs to the plot
     - :code:`corrected_image.tif`, :code:`shapefile.geojson`
     - :code:`plot.tif`
   * - `stitch_plots <https://github.com/phytooracle/stitch_plots>`_ 
     - Stitch plots together to form a full field orthomosaic
     - :code:`plot.tif`
     - :code:`orthomosaic.tif`
   * - Plant area extractor (unpublished, 2020) 
     - Extracts plant area for each single plant
     - :code:`plot.tif`
     - ::code:`plant_area.csv`

Running the Pipeline 
====================

.. note::
   
   At this point, we assume that the interactive "manager" and "worker" nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

Retrieve data
^^^^^^^^^^^^^

Navigate to your RGB directory, download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/StereoTopRGB
   iget -rKVP /iplant/home/shared/terraref/ua-mac/raw_tars/demo_data/Lettuce/StereoTopRGB_demo.tar
   tar -xvf StereoTopRGB_demo.tar

Data from the Gantry can be found within :code:`/iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTopRGB/<scan_date>.tar`

Retrieve correction file
^^^^^^^^^^^^^^^^^^^^^^^^

Dowload the coordiate correction :code:`.csv` file:

.. code::

   iget -rKVP /iplant/home/emmanuelgonzalez/Ariyan_ortho_attempt_4/2020-01-08_coordinates_CORRECTED_4-16-2020.csv

.. note::
   
   The :code:`.csv` file will soon be moved to a shared directory; this will change accordingly.
   
Edit scripts
^^^^^^^^^^^^

+ :code:`process_one_set.sh`, :code:`process_one_set2.sh`

  Find your current working directory using the command :code:`pwd`.
  Open :code:`process_one_set.sh` and paste the output from :code:`pwd` into line 14 (line 12 in :code:`process_one_set2.sh`). It should look something like this:

  .. code:: 

    HPC_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/StereoTopRGB/"
  
  Set your :code:`.simg` folder path in line 15 (line 13 in :code:`process_one_set2.sh`).

  .. code:: 

    SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  
  
+ :code:`run.sh`

  Open :code:`run.sh` and paste the output from :code:`pwd` into line 7. It should look something like this:

    .. code:: 

      PIPE_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/StereoTopRGB/"
    
    Set your :code:`.simg` folder path in line 8.

    .. code:: 

      SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  

  + In line 1, specify the :code:`<scan_date>` folder you want to process. For our purposes this will look like:

    .. code:: 

      phython3 gen_files_list.py StereoTopRGB_demo > raw_data_files.json

+ :code:`entrypoint.sh`, :code:`entrypoint-2.sh`

  In lines 7 and 11, specify the location of CCTools:

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/jx2json

    and

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/makeflow

Run pipeline
^^^^^^^^^^^^

Begin processing using:

.. code::

  ./run.sh <folder_to_process>

.. note::
   
   This may return a notice with a "FATAL" error. This happens as the pipeline waits for a connection to DockerHub, which takes some time. Usually, the system will fail quickly if there is an issue.

   If the pipeline fails, check to make sure you have a "/" concluding line 14 of :code:`process_one_set.sh`. This is one of the most common errors and is necessary to connect the program scripts to the HPC.
