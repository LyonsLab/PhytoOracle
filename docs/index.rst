.. PhytoOracle documentation master file, created by
   sphinx-quickstart on Thu May 21 12:03:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************
Welcome to PhytoOracle!
***********************

PhytoOracle is a scalable, distributed workflow manager for analyzing highthroughput phenotyping data.  
It is designed to process data from the `UA Gantry <https://uanews.arizona.edu/story/world-s-largest-robotic-field-scanner-now-place,>`_, but can be adapted to work on data coming from other platforms.  
PhytoOracle uses a master-worker framework for distributed computing and can run jobs on nearly all unix-like environment. 

Supported Sensors & Pipelines
=============================

.. list-table::
   :header-rows: 1

   * - Sensor
     - Data Description
   * - `StereoTopRGB <https://phytooracle.readthedocs.io/en/latest/4_StereoTopRGB_run.html>`_
     - Data from RGB images is used for measuring the plant's canopy cover
   * - `FlirIr <https://phytooracle.readthedocs.io/en/latest/5_FlirIr_run.html>`_
     - Infrared images are used to extract information about mean temperature between plants
   * - PSII
     - Measures the photosynthetic activity
   * - Stereotop3D
     - Used to measure the plant's physical structure
   * - Hyperspectral
     - Collects and processes information from across the electromagnetic spectrum

