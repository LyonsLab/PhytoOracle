*************************************
Obtaining Pipeline Related Containers
*************************************

All the code used throughout the pipeline is containerized through `Docker <https://www.docker.com/>`_ and hosted on `DockerHub <https://hub.docker.com/>`_.

We use `Singularity <https://sylabs.io/docs/>`_ to execute containers on the HPC system.

Each container is first downloaded and stored in :code:`.simg` format to maximise time efficiency. 

We suggest creating a folder containing all containers in :code:`.simg` format close to your root directory and adding the path to the folder to the :code:`process_one_set.sh` pipeline scripts.

To create a :code:`.simg` file you will require Singularity to be installed and executable, then do:

.. code::

   singularity build <name_of_container>.simg docker://<dockeruser>/<container>:<version>

For Example:

.. code::

   singularity build stitch_plots.simg docker://phytooracle/stitch_plots:latest

Full list of containers
=======================

Pipeline
GitHub Link
DockerHub Link

`cleanmetadata <https://github.com/AgPipeline/moving-transformer-cleanmetadata>`_
`bin2tif <https://github.com/AgPipeline/moving-transformer-bin2tif>`_
`collect_gps <https://github.com/emmanuelgonz/collect_gps>`_
MEGASTITCH (Zarei, unpublished)
`replace_gps <https://github.com/emmanuelgonz/edit_gps>`_ 
`plotclip <https://github.com/emmanuelgonz/plotclip_shp>`_ 
`stitch_plots <https://github.com/phytooracle/stitch_plots>`_ 
Plant area extractor (unpublished) 

StereoTopRGB
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Container
     - DockerHub Repo
     - GitHub Link
   * - cleanmetadata 
     - :code:`docker://AgPipeline/moving-transformer-cleanmetadata:latest`
     - https://github.com/AgPipeline/moving-transformer-cleanmetadata
   * - bin2tif
     - :code:`docker://AgPipeline/moving-transformer-bin2tif:latest`
     - https://github.com/AgPipeline/moving-transformer-bin2tif
   * - collect_gps 
     - :code:`docker://emmanuelgonz/collect_gps:latest`
     - https://github.com/emmanuelgonz/collect_gps
   * - MEGASTITCH (Zarei, unpublished)
     - unpublished
     - unpublished
   * - replace_gps
     - :code:`docker://emmanuelgonz/plotclip_shp:latest`
     - https://github.com/emmanuelgonz/edit_gps
   * - plotclip
     - :code:`docker://emmanuelgonz/plotclip_shp:latest`
     - https://github.com/emmanuelgonz/plotclip_shp
   * - stitch_plots
     - :code:`docker://phytooracle/stitch_plots:latest`
     - https://github.com/phytooracle/stitch_plots
   * - Plant area extractor (unpublished, 2020) 
     - unpublished
     - unpublished

FlirIr
^^^^^^

PSII
^^^^

Stereop3DTop
^^^^^^^^^^^^