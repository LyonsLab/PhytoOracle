.. PhytoOracle documentation master file, created by
   sphinx-quickstart on Thu May 21 12:03:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************
Welcome to PhytoOracle!
***********************

PhytoOracle is a scalable, distributed workflow manager for analyzing highthroughput phenotyping data.  
It is designed to process data from the `UA Gantry <https://uanews.arizona.edu/story/world-s-largest-robotic-field-scanner-now-place,>`_, but can be adapted to work on data coming from other platforms.  
PhytoOracle uses a master-worker framework for distributed computing (HPC, Cloud, etc.) and can run jobs on nearly all unix-like environments. 
Access our Github `here <https://github.com/uacic/PhytoOracle/>`_.

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
     - Collects and processes information from across the electromagnetic spectrum for a wide variety of data (e.g. vegetation indices)

Pipeline Structure
==================

All of the pipelines follow the same structure that allows for accessiblility and modularity.

1. Setting up the Master interactive node and Worker nodes on the HPC;
2. Cloning the pipeline of choice;
3. Staging the data;
4. Editing the scripts;
5. Launching the pipeline.

Acknowledgements
================

This project partially built on code initially developed by the `TERRA-REF project <https://www.terraref.org/>`_ and `Ag-Pipeline <https://github.com/AgPipeline/>`_ team. We thank the University of Arizona Advanced Cyberinfrastrcture Concept class of 2019 for additional work.
