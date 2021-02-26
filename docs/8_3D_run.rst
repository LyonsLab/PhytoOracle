****************************************************************
Running the Scanner3DTop Pipeline for Generating 3D Point Clouds
****************************************************************

This pipeline combines left and right 3D point clouds to create a single, merged 3D point cloud per range. Before starting, change to :code:`master` branch with :code:`git checkout master`.

Pipeline Overview
=================

Scanner3DTop currently uses only a single distributed program:

.. list-table::
   :header-rows: 1
   
   * - Program
     - Function
     - Input
     - Output
   * - `3D MergePly <https://github.com/phytooracle/3d_merge_ply>`_
     - Merges PLY files into a single 3D point cloud
     - :code:`left.ply`, :code:`right.ply`
     - :code:`merged.ply`

Running the Pipeline 
====================

.. note::
   
   At this point, we assume that the interactive "foreman" and "worker" nodes have already been setup and are running, and the pipelines have been cloned from GitHub. 
   If this is not the case, start `here <https://phytooracle.readthedocs.io/en/latest/2_HPC_install.html>`_.

Retrieve data
^^^^^^^^^^^^^

Navigate to your directory containing Scanner3DTop, and download the data from the CyVerse DataStore with iRODS commands and untar:

.. code::

   cd /<personal_folder>/PhytoOracle/FlirIr
   iget -rKVP /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/Scanner3DTop/<Scanner3DTop-date.tar>
   tar -xvf <Scanner3DTop-date.tar>

.. note::

   For a full list of available unprocessed data navigate to https://datacommons.cyverse.org/browse/iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/Scanner3DTop/

Edit scripts
^^^^^^^^^^^^

+ :code:`process_one_set.sh`

  Find your current working directory using the command :code:`pwd`
  Open :code:`process_one_set.sh` and copy the output from :code:`pwd` into line 12. It should look something like this:

  .. code:: 

    HPC_PATH="xdisk/group_folder/personal_folder/PhytoOracle/Scanner3DTop/"

  Set your :code:`.simg` folder path in line 13.

  .. code:: 

    SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  

+ :code:`run.sh.main`

  + Paste the output from :code:`pwd` into line 7. It should look something like this:

    .. code:: 

      PIPE_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/Scanner3DTop/"

  + Set your :code:`.simg` folder path in line 8.

    .. code:: 

      SIMG_PATH="/xdisk/group_folder/personal_folder/PhytoOracle/singularity_images/"  

  + In line 4, specify the :code:`<scan_date>` folder you want to process. For our purposes, this will look like:

    .. code:: 

      DATE="<scan_date>"

  + In lines 16 and 19, specify the location of CCTools:

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/jx2json

    and

    .. code:: 

      /home/<u_num>/<username>/cctools-<version>-x86_64-centos7/bin/makeflow

Run pipeline
^^^^^^^^^^^^

Begin processing using:

.. code::

  ./run.sh.main

.. note::

   This may return a notice with a "FATAL" error. This happens as the pipeline waits for a connection to DockerHub, which takes some time. Usually, the system will fail quickly if there is an issue.

   If the pipeline fails, check to make sure you have a "/" concluding line 14 of :code:`process_one_set.sh`. This is one of the most common errors and is necessary to connect the program scripts to the HPC.
