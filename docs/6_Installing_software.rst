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

