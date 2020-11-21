.. PhytoOracle documentation master file, created by
   sphinx-quickstart on Thu May 21 12:03:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************
Welcome to PhytoOracle!
***********************

|PhytoOracle_logo|_

PhytoOracle is a scalable, distributed workflow manager for analyzing highthroughput phenotyping data.  
It is designed to process data from the `UA Gantry <https://uanews.arizona.edu/story/world-s-largest-robotic-field-scanner-now-place,>`_, but can be adapted to work on data coming from other platforms.  
PhytoOracle uses a master-worker framework for distributed computing (HPC, Cloud, etc.) and can run jobs on nearly all unix-like environments. 
Access our Github `here <https://github.com/uacic/PhytoOracle/>`_.

Supported Sensors & Pipelines
=============================

.. list-table::
   :header-rows: 1

   * - Sensor
     - Sensor Name
     - Data Description
   * - `StereoTopRGB <https://phytooracle.readthedocs.io/en/latest/4_StereoTopRGB_run.html>`_
     - Prosilica GT3300C
     - Stereo RGB images.  Identifies plants; measuring plant area
   * - `FlirIr <https://phytooracle.readthedocs.io/en/latest/5_FlirIr_run.html>`_
     - FLIR A615, 45°
     - Infrared images. Measures temperature of plants
   * - `PSII <https://phytooracle.readthedocs.io/en/latest/7_PSII_run.html>`_
     - LemnaTec custom based of an Allied Vision Manta camera
     - Fluorescence images. Measures chlorophyll fluorescence for calculating plant photosynthetic potential.
   * - `Scanner3DTop <https://phytooracle.readthedocs.io/en/latest/8_3D_run.html>`_
     - Custom 3D Fraunhofer
     - Laser scanning images.  Generates a point cloud for measuring physical structure of plants.
   * - Hyperspectral (VNIR/SWIR)
     - Custom Headwall Photonics
     - Hyperspectral images.  Collects and processes information from across the electromagnetic spectrum for a wide variety of phenotypes (e.g., vegetation indices)

Pipeline Structure
==================

All of the pipelines follow the same structure that allows for accessiblility, scalability, and modularity. The steps are:

1. Setting up the Master interactive node and Worker nodes on the HPC
2. Cloning the pipeline of choice
3. Staging the data
4. Editing the scripts
5. Launching the pipeline

Acknowledgements
================

This project partially built on code initially developed by the `TERRA-REF project <https://www.terraref.org/>`_ and `Ag-Pipeline <https://github.com/AgPipeline/>`_ team. We thank the University of Arizona Advanced Cyberinfrastrcture Concept class of 2019 for additional work. Logo credit: Christian Gonzalez.

Issues and Questions
====================

If you have questions, raise an issue on the `GitHub <https://github.com/LyonsLab/PhytoOracle/>`_ page.

**For specific workflows and adapting a pipeline for your own work contact:**

* Emmanuel Gonzalez: emmanuelgonzalez [at] email.arizona.edu
* Michele Cosi: cosi [at] email.arizona.edu

**For plant detection and plant clustering:**

* Travis Simmons: travis.simmons [at] ccga.edu 

**For the orthomosaicing algorithm:**

* Ariyan Zarei: ariyanzarei [at] email.arizona.edu 

.. |PhytoOracle_logo| image:: ../pics/PhytoOracle_logo.PNG
    :width: 50
.. _PhytoOracle_logo: