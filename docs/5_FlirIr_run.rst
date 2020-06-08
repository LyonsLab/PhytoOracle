*********************************************
Running the FlirIr Pipeline for Infrared Data
*********************************************

.. note::
   
   The FlirIr pipeline will be updated shortly. If this documentation is not accurate at time of use, please open an issue on the PhytoOracle GitHub.

This pipeline extracts temperature data from image files. This guide provides demo data you can use follow along with and ensure the pipeline is functional. 

Pipeline Overview
=================

FlirIr currently uses 4 different programs for data conversion:

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
   * - `flir2tif <https://github.com/AgPipeline/moving-transformer-flir2tif>`_
     - Converts bin compressed files to tif 
     - :code:'image.flir'
     - :code:'image.tif'
   * - `flir_clip <https://github.com/AgPipeline/moving-transformer->`_
     - Matches temperature data to plot
     - :code:'image.tif'
     - :code:'clipped_image.tif'
   * - `plot_meantemp <https://github.com/AgPipeline/moving-transformer-meantemp>`_ 
     - Extracts temperature from detected biomass
     - :code:'image.bin', :code:'clipped_image.tif'
     - :code:'meantemp.csv'

Running the Pipeline 
====================

.. note::
   
   At this point we assume that the interactive Master and Worker nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

**Retrieve data**

Navigate to your directory containing FlirIr, and download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/FlirIr
   iget -rKVP /iplant/home/shared/terraref/ua-mac/raw_tars/demo_data/Lettuce/FlirIr_demo.tar
   tar -xvf FlirIr_demo.tar

Data from the Gantry can be found within :code:'/iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/flirIrCamera/<scan_date>.tar'
   
**Edit scripts**

+ :code:'process_one_set.sh'
  Find your current working directory using the command :code:'pwd'
  Open :code:'process_one_set.sh' and copy the output from :code:'pwd' into line 14. It should look something like this:

  .. code:: 
    HPC_PATH="xdisk/group_folder/personal_folder/PhytoOracle/FlirIr/"

+ :code:`entrypoint.sh`
  + In line 1, specify the :code:`<scan_date>` folder you want to process. For our purposes, this will look like:

    .. code:: 
      phython3 gen_files_list.py FlirIr_demo > raw_data_files.json

  + In lines 7 and 11, specify the location of CCTools:

    .. code:: 
      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/jx2json

    and

    .. code:: 
      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/makeflow

**Run pipeline**

Begin computations using:

.. code::
  ./entrypoint.sh

.. note::
   
   It will return a notice with a "FATAL" error. This happens as the pipeline waits for a connection to Docker. It should take some time and will fail quickly if there is an issue.
   If the pipeline fails, ENSURE THEIR THE LAST SLASH IN HPC_PATH VARIABLE IN :code:'process_one_set.sh'. This is the most common error.
