
PhytoOracle Workflows
=====================

PhytoOracle is designed for distributed scaling on cloud platforms and High-Performance Computers. The minimum requirements being:

        - One MASTER instance with the required data staged that will broadcast and distribute jobs
        - One or more WORKER instances (Cloud/HPC) that will connect to the Master and execute jobs

.. image:: ../pics/general_concept_map.png
   :width: 600

Workflows for the following sensors are available via `PhytoOracle Github <https://github.com/uacic/PhytoOracle>`_. Click on the links below for step-by-step instructions

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Workflow
     - Link
   * - StereotopRGB
     - `Click here <https://github.com/uacic/PhytoOracle/tree/alpha/stereoTopRGB>`_
   * - Scanner3D
     - Under construction
   * - PSII Flourescence
     - Under construction
   * - FlirIr
     - `Click here <https://github.com/uacic/PhytoOracle/tree/alpha/FlirIr>`_


**Description of datasets**

See `Data types & description <https://phytooracle.readthedocs.io/en/latest/Input_data.html>`_ section for more details on input data types and descriptions.

**Workflow Scaling**

- PhytoOracle workflows can be run on any cloud platform (public & private) and also WORKERS can be connected from a High-Performance Computer. 
- It is adviced that the worker instances have the recommended software dependencies installed 

See Advanced documentation `here <https://phytooracle.readthedocs.io/en/latest/advanced.html>`_ for sample PBS scripts.  

**Workflow modification**

See our `Advanced Secion <https://phytooracle.readthedocs.io/en/latest/advanced.html>`_ for more details on workflow modifications and container swapping. 


-----
