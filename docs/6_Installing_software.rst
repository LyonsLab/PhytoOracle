*************************************
Obtaining Pipeline Related Containers
*************************************

All the code used throughout the pipeline is containerized through `Docker <https://www.docker.com/>`_ and hosted on `DockerHub <https://hub.docker.com/>`_. We use `Singularity <https://sylabs.io/docs/>`_ to execute containers on the HPC system.

Each container is first downloaded and stored in :code:`.simg` format to maximise time efficiency. We suggest creating a folder containing all containers in :code:`.simg` format close to your root directory and adding the path to the folder to the :code:`process_one_set.sh` pipeline scripts.

To create a :code:`.simg` file you will require Singularity to be installed and executable, then do:

.. code::

   singularity build <name_of_container>.simg docker://<dockeruser>/<container>:<version>

For Example:

.. code::

   singularity build stitch_plots.simg docker://phytooracle/stitch_plots:latest

Full list of containers
=======================

StereoTopRGB
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Container
     - DockerHub Repo
     - GitHub Link
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

.. list-table::
   :header-rows: 1

   * - Container
     - DockerHub Repo
     - GitHub Link
   * - flir2tif
     - :code:`docker://cosimichele/po_flir2tif_s10:latest`
     - https://github.com/CosiMichele/Containers/tree/master/po_flir2tif_s10
   * - collect_gps 
     - :code:`docker://emmanuelgonz/collect_gps:latest`
     - https://github.com/emmanuelgonz/collect_gps
   * - MEGASTITCH (Zarei, unpublished)
     - unpublished
     - unpublished
   * - replace_gps
     - :code:`docker://emmanuelgonz/plotclip_shp:latest`
     - https://github.com/emmanuelgonz/edit_gps
   * - flirfieldplot
     - :code:`docker://cosimichele/flirfieldplot:latest`
     - https://github.com/CosiMichele/Containers/tree/master/flirfieldplot
   * - plotclip_geo
     - :code:`docker://emmanuelgonz/plotclip_shp:latest`
     - https://github.com/emmanuelgonz/plotclip_shp
   * - stitch_plots
     - :code:`docker://phytooracle/stitch_plots:latest`
     - https://github.com/phytooracle/stitch_plots
   * - po_temp_cv2stats 
     - :code:`docker://cosimichele/po_temp_cv2stats:latest`
     - https://github.com/CosiMichele/Containers/tree/master/po_meantemp_comb

PSII
^^^^

Stereop3DTop
^^^^^^^^^^^^