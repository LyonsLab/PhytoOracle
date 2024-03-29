***************************************************************
Running the StereoTopRGB Pipeline for Detecting Plant Area Data
***************************************************************

This pipeline extracts plant area data from image files. This guide provides demo data you can use follow along with and ensure the pipeline is functional. Before starting, change to :code:`alpha` branch with :code:`git checkout alpha`.

Pipeline Overview
=================

StereoTopRGB currently uses 7 different programs for the analytical pipeline:

.. list-table::
   :header-rows: 1
   
   * - Program
     - Function
     - Input
     - Output
   * - `bin2tif <https://github.com/phytooracle/rgb_bin_to_tif>`_
     - Converts bin compressed files to geotiff
     - :code:`image.bin`, :code:`metadata.json`
     - :code:`image.tif`
   * - `collect_gps <https://github.com/phytooracle/rgb_flir_collect_gps>`_
     - Collects GPS coordinates from all geotiff files
     - :code:`image.tif`
     - :code:`collected_coordinates.csv`
   * - `Orthomosaicing <https://github.com/ariyanzri/Lettuce_Image_Stitching>`_
     - Finds best possible coordinates of all geotiffs
     - :code:`collected_coordinates.csv`
     - :code:`corrected_coordinates.csv`
   * - `replace_gps <https://github.com/phytooracle/rgb_flir_edit_gps>`_ 
     - Applies corrected GPS coordinates to images
     - :code:`corrected_coordinates.csv`, :code:`image.tif`
     - :code:`corrected_image.tif`
   * - `plotclip <https://github.com/phytooracle/rgb_flir_plot_clip_geojson>`_ 
     - Clips geotiffs to the plot
     - :code:`corrected_image.tif`, :code:`shapefile.geojson`
     - :code:`plot.tif`
   * - `Plant detection <https://github.com/phytooracle/rgb_flir_plant_detection>`_
     - Detects plants over days
     - :code:`plot.tif`
     - ::code:`genotype.csv`
   * - `Plant clustering <https://github.com/phytooracle/rgb_flir_plant_clustering>`_
     - Tracks plants over days
     - :code:`genotype.csv`
     - ::code:`pointmatching.csv`


Running the Pipeline 
====================

.. note::
   
   At this point, we assume that the interactive "foreman" and "worker" nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

Retrieve data
^^^^^^^^^^^^^

Navigate to your RGB directory, download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/StereoTopRGB
   iget -rKVP /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/stereoTop/<stereoTop-date.tar>
   tar -xvf <stereoTop-date.tar>

.. note::

   For a full list of available unprocessed data navigate to https://datacommons.cyverse.org/browse/iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/StereoTopRGB/

Retrieve vector and ML model files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dowload the coordiate correction :code:`.csv` file:

.. code::

  iget -N 0 -PVT /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/season10_multi_latlon_geno.geojson

  iget -N 0 -PVT /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/necessary_files/gcp_season_10.txt

  iget -N 0 -PVT /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/necessary_files/model_weights.pth
   
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

  +Open :code:`run.sh` and paste the output from :code:`pwd` into line 7. It should look something like this:

    .. code:: 

      PIPE_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/StereoTopRGB/"
    
  +Set your :code:`.simg` folder path in line 8.

    .. code:: 

      SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  

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


Troubleshooting and Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

If problems arise with this pipeline, please refer to the `tutorial on GitHub specific to the RGB pileline <https://github.com/LyonsLab/PhytoOracle/tree/master/stereoTopRGB>`_. If problems persist, raise an issue.