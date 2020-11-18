*********************************************
Running the FlirIr Pipeline for Infrared Data
*********************************************

This pipeline extracts temperature data from image files. This guide provides demo data you can use follow along with and ensure the pipeline is functional. Before starting, change to :code:`master` branch with :code:`git checkout master`.

Pipeline Overview
=================

FlirIr currently uses 8 different programs for data conversion:

.. list-table::
   :header-rows: 1
   
   * - Program
     - Function
     - Input
     - Output
   * - `flir2tif <https://github.com/phytooracle/flir_bin_to_tif_s10>`_
     - Temperature calibrated transformer that converts bin compressed files to tif 
     - :code:`image.bin`, :code:`metadata.json`
     - :code:`image.tif`
   * - `collect_gps <https://github.com/phytooracle/rgb_flir_collect_gps>`_
     - Collects GPS coordinates from all geotiff files
     - :code:`image.tif`
     - :code:`collected_coordinates.csv`
   * - MEGASTITCH (Zarei, unpublished)
     - Finds best possible coordinates of all geotiffs
     - :code:`collected_coordinates.csv`
     - :code:`corrected_coordinates.csv`
   * - `replace_gps <https://github.com/phytooracle/rgb_flir_edit_gps>`_ 
     - Applies corrected GPS coordinates to images
     - :code:`corrected_coordinates.csv`, :code:`image.tif`
     - :code:`corrected_image.tif`
   * - `flir_field_stitch <https://github.com/phytooracle/flir_field_stitch>`_
     - GDAL based transformer that combines all immages into a single orthomosaic
     - Directory of all converted :code:`image.tif`
     - :code:`ortho.tif`
   * - `plotclip_geo <https://github.com/phytooracle/rgb_flir_plot_clip_geojson>`_
     - Clips plots from orthomosaic
     - :code:`coordinatefile.geojson`, :code:`ortho.tif`
     - :code:`clipped_plots.tif`
   * - `stitch_plots <https://github.com/phytooracle/stitch_plots>`_
     - Renames and stitches plots
     - Directory of all :code:`clipped_plots.tif`
     - :code:`stitched_plots.tif`
   * - `flir_meantemp <https://github.com/phytooracle/flir_meantemp>`_ 
     - Extracts temperature using from detected biomass
     - :code:`coordinatefile.geojson`, Directory of all :code:`stitched_plots.tif`
     - :code:`meantemp.csv`

Running the Pipeline 
====================

.. note::
   
   At this point, we assume that the interactive "foreman" and "worker" nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

Retrieve data
^^^^^^^^^^^^^

Navigate to your directory containing FlirIr, and download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/FlirIr
   iget -rKVP /iplant/home/shared/terraref/ua-mac/raw_tars/demo_data/Lettuce/FlirIr_demo.tar
   tar -xvf FlirIr_demo.tar

Data from the Gantry can be found within :code:`/iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/flirIrCamera/<scan_date>.tar`
   
Edit scripts
^^^^^^^^^^^^

+ :code:`process_one_set.sh`

  Find your current working directory using the command :code:`pwd`
  Open :code:`process_one_set.sh` and copy the output from :code:`pwd` into line 14. It should look something like this:

  .. code:: 

    HPC_PATH="xdisk/group_folder/personal_folder/PhytoOracle/FlirIr/"

  Set your :code:`.simg` folder path in line 8.

  .. code:: 

    SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  

+ :code:`run.sh`

  + Paste the output from :code:`pwd` into line 7. It should look something like this:

    .. code:: 

      PIPE_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/FlirIr/"

  + Set your :code:`.simg` folder path in line 8.

    .. code:: 

      SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  

  + In line 4, specify the :code:`<scan_date>` folder you want to process. For our purposes, this will look like:

    .. code:: 

      DATE="FlirIr_demo"

  + In lines 25 and 28, specify the location of CCTools:

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/jx2json

    and

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/makeflow

Run pipeline
^^^^^^^^^^^^

Begin processing using:

.. code::

  ./run.sh

.. note::

   This may return a notice with a "FATAL" error. This happens as the pipeline waits for a connection to DockerHub, which takes some time. Usually, the system will fail quickly if there is an issue.

   If the pipeline fails, check to make sure you have a "/" concluding line 14 of :code:`process_one_set.sh`. This is one of the most common errors and is necessary to connect the program scripts to the HPC.
