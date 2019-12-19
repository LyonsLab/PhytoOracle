
**PhytoOracle Workflows**
-------------------------

PhytoOracle is designed for distributed scaling on cloud platforms and High-Performance Computers. The minimum requirements being:

        - One MASTER instance with the required data staged that will broadcast  and distribute jobs
        - One or more WORKER instances (Cloud/HPC) that will connect to the Master and execute jobs

|general_concept_map|_

Workflows for the following sensors are available via Github. Click on the links below for step-by-step instructions

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Workflow
     - Link
   * - StereotopRGB
     - `Click here <https://github.com/uacic/PhytoOracle/blob/master/stereoTop/README.md>`_
   * - Scanner3D
     - Under construction
   * - PSII Flourescence
     - Under construction



Description of datasets
-----------------------

See |Input_Data| section for more details on input data types and descriptions.

Describe scaling methods
------------------------

PhytoOracle workflows can technically be run on any cloud platform (public & private) and also WORKERS can be connected from a High-Performance Computer. See Advanced documentation `here <>`_ for sample PBS scripts.  

Workflow modification
---------------------

See our `Advanced Secion <>`_ for more details on workflow modifications and container swapping. 


-----

.. |general_concept_map| image:: ./pics/general_concept_map.png
    :width: 650
    :height: 450
.. _general_concept_map: 

.. |general_concept_map| image:: ./pics/general_concept_map.jpeg
    :width: 500
    :height: 100
.. _general_concept_map:   
