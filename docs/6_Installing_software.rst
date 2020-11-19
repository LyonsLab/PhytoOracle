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
     - :code:`docker://phytooracle/rgb_bin_to_tif:latest`
     - https://github.com/phytooracle/rgb_bin_to_tif
   * - collect_gps 
     - :code:`docker://phytooracle/rgb_flir_collect_gps:latest`
     - https://github.com/phytooracle/rgb_flir_collect_gps
   * - Orthomosaicing
     - 
     - https://github.com/ariyanzri/Lettuce_Image_Stitching
   * - replace_gps
     - :code:`docker://phytooracle/rgb_flir_edit_gps:latest`
     - https://github.com/phytooracle/rgb_flir_edit_gps
   * - plotclip
     - :code:`docker://phytooracle/rgb_flir_plot_clip_geojson:latest`
     - https://github.com/phytooracle/rgb_flir_plot_clip_geojson
   * - Plant detection
     - :code:`docker://phytooracle/rgb_flir_plant_detection`
     - https://github.com/phytooracle/rgb_flir_plant_detection
   * - Plant clustering
     - :code:`docker://phytooracle/rgb_flir_plant_clustering:latest`
     - https://github.com/phytooracle/rgb_flir_plant_clustering


FlirIr
^^^^^^

.. list-table::
   :header-rows: 1

   * - Container
     - DockerHub Repo
     - GitHub Link
   * - flir2tif
     - :code:`docker://phytooracle/flir_bin_to_tif_s10:latest`
     - https://github.com/phytooracle/flir_bin_to_tif_s10
   * - collect_gps 
     - :code:`docker://phytooracle/rgb_flir_collect_gps:latest`
     - https://github.com/phytooracle/rgb_flir_collect_gps
   * - Orthomosaicing
     - 
     - https://github.com/ariyanzri/Lettuce_Image_Stitching
   * - replace_gps
     - :code:`docker://phytooracle/rgb_flir_edit_gps:latest`
     - https://github.com/phytooracle/rgb_flir_edit_gps
   * - flir_field_stitch
     - :code:`docker://phytooracle/flir_field_stitch:latest`
     - https://github.com/phytooracle/flir_field_stitch
   * - plotclip
     - :code:`docker://phytooracle/rgb_flir_plot_clip_geojson:latest`
     - https://github.com/phytooracle/rgb_flir_plot_clip_geojson
   * - flir_meantemp 
     - :code:`docker://phytooracle/flir_meantemp:latest`
     - https://github.com/phytooracle/flir_meantemp

PSII
^^^^

.. list-table::
   :header-rows: 1

   * - Container
     - DockerHub Repo
     - GitHub Link
   * - cleanmetadata
     - :code:`docker://AgPipeline/moving-transformer-cleanmetadata:latest`
     - https://github.com/AgPipeline/moving-transformer-cleanmetadata
   * - bin2tif 
     - :code:`docker://phytooracle/psii_bin_to_tif:latest`
     - https://github.com/phytooracle/psii_bin_to_tif
   * - resizetif
     - :code:`docker://phytooracle/psii_resize_tif:latest`
     - https://github.com/phytooracle/psii_resize_tif
   * - flir_field_stitch
     - :code:`docker://phytooracle/flir_field_stitch:latest`
     - https://github.com/phytooracle/flir_field_stitch
   * - plotclip
     - :code:`docker://phytooracle/rgb_flir_plot_clip_geojson:latest`
     - https://github.com/phytooracle/rgb_flir_plot_clip_geojson
   * - psii_segmentation
     - :code:`docker://phytooracle/psii_segmentation:latest`
     - https://github.com/phytooracle/psii_segmentation
   * - psii_fluorescence_aggregation
     - :code:`docker://phytooracle/psii_fluorescence_aggregation:latest`
     - https://github.com/phytooracle/psii_fluorescence_aggregation

Scanner3DTop
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Container
     - DockerHub Repo
     - GitHub Link
   * - 3D MergePly
     - :code:`docker://phytooracle/3d_merge_ply:latest`
     - https://github.com/phytooracle/3d_merge_ply