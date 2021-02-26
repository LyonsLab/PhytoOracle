***********************************************************
Running the PSII Pipeline for Photosynthetic Potential Data
***********************************************************

This pipeline uses the data transformers to extract chlorophyll fluorescence data from image files. Before starting, change to :code:`alpha` branch with :code:`git checkout alpha`.

Pipeline Overview
=================

PSII currently uses 6 different programs for the analytical pipeline:

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
   * - `bin2tif <https://github.com/phytooracle/psii_bin_to_tif>`_
     - Converts bin compressed files to geotiff
     - :code:`image.bin`
     - :code:`image.tif`
   * - `resizetif <https://github.com/phytooracle/psii_resize_tif>`_
     - Resized original geotiffs to correct 
     - :code:`image.tif`
     - :code:`resized_image.tif`
   * - `plotclip <https://github.com/phytooracle/rgb_flir_plot_clip_geojson>`_ 
     - Clips geotiffs to the plot
     - :code:`resized_image.tif`, :code:`shapefile.geojson`
     - :code:`plot.tif`
   * - `psii_segmentation <https://github.com/phytooracle/psii_segmentation>`_ 
     - Segments images given a validated set of thresholds
     - :code:`plot.tif`
     - :code:`segment.csv`
   * - `psii_fluorescence_aggregation <https://github.com/phytooracle/psii_fluorescence_aggregation>`_
     - Aggregates segmentation data for each image and calculates F0, Fm, Fv, and Fv/Fm
     - :code:`segment.csv`, :code:`multitresh.json`
     - ::code:`fluorescence_agg.csv`

Running the Pipeline 
====================

.. note::
   
   At this point, we assume that the interactive "foreman" and "worker" nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

Retrieve data
^^^^^^^^^^^^^

Navigate to your PSII directory, download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/FlirIr
   iget -rKVP /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/ps2Top/<ps2Top-date.tar>
   tar -xvf <ps2Top-date.tar>

.. note::

   For a full list of available unprocessed data navigate to https://datacommons.cyverse.org/browse/iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/ps2Top/
   
Edit scripts
^^^^^^^^^^^^

+ :code:`process_one_set.sh`, :code:`process_one_set2.sh`

  Find your current working directory using the command :code:`pwd`.
  Open :code:`process_one_set.sh` and paste the output from :code:`pwd` into line 15. It should look something like this:

  .. code:: 

    HPC_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/PSII/"
  
  Set your :code:`.simg` folder path in line 16.

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
