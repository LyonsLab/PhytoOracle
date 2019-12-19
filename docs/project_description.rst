About TERRA-REF
---------------

Modern agriculture has made great progress in reducing hunger and poverty and improving food security and nutrition but still faces tremendous challenges in the coming decades. In order to accelerate plant breeding, we need novel high-throughput phenotyping (HTP) approaches to advance the understanding of genotype-to-phenotype. The Transportation Energy Resources from Renewable Agriculture Phenotyping Reference Platform (TERRA-REF) is one such program that aims to transform plant breeding by using remote sensing to quantify plant traits. The TERRA-REF project provides a data and computation pipeline responsible for collecting, transferring, processing and distributing large volumes of crop sensing and genomic data. Visit official `

|gantry|_


**Gantry Sensors**

|gantry_sensors|_


The Lemnatec Scanalyzer Field System is a high-throughput phenotyping field-scanning robot that autonomously moves and continuously collects images of the crops it hovers. Attached to the 30-ton steel gantry of the field-scanning robot are sensors and cameras that collect different sets of data. The diverse array of sensors allow researchers to collect significant sets of data that can be used to leverage biological insight into how environments affect phenotypes and the overall relationship between genotypes (gene) and phenotypes (characteristic). Below are three field scanning imaging sensors:
 
  
**StereoRGB**
  
  	The Stereo RGB camera captures images from above which enables researchers to determine canopy cover (spread of plants), the  amount of crops, etc.
	
**Scanner3D**
  
  	A 3D scanner that captures the architecture of plants, such as leaf angles and shapes.
	
**PSII Fluorescence**
  
  	A fluorescence camera that allows researchers to measure plant's photosynthetic efficiency.
	

**TERRA-REF Data Volume**

- The gantry system outputs 1 TB of data per day from various sensors. For individual sensor data types and sizes for a day see Data Types & Descriptions `here <https://phytooracle.readthedocs.io/en/latest/Input_data.html>`_

	
**Need for data analysis workflows**

- Existing cyber-infrastructure for the TERRA-REF platform lacks workflow features and has complex dependencies which make it difficult to distribute compute across compute platforms.
- More and more researchers are planning studies of various diverse plant phenotypes using the TERRA-REF platform hence, the need for data-analysis workflows that are reporoducible and easy to deploy.


**PhytoOracle**
---------------

PhytoOracle is a scalable, modular data pipeline for phenomics research. It uses `CCTools <http://ccl.cse.nd.edu/>`_ `Makeflow <http://ccl.cse.nd.edu/software/makeflow/>`_ workflow system for executing large complex workflows on clusters, clouds, and grids. PhytoOracle aims to significantly reduce data-processing times while providing reliable workflows for easy deployment.

**PhytoOracle Solution:**
	- Modular - uses containers for easy deployment and workflow-redesigns.
	- Scalable - uses distributed-computing where possible to reduce processing times.
	- Customizable - uses JX workflow language that allows for custom workflow designs.
	- Fault tolerant - Makeflow is highly fault tolerant: it can crash or be killed, and upon resuming, will reconnect to running jobs and continue where it left off.
	- Leverages open-source solutions.

|general_concept_map|_





----

.. |general_concept_map| image:: ../pics/general_concept_map.png
    :width: 600
.. _general_concept_map: 
.. |gantry| image:: ../pics/gantry.png
    :width: 600
.. _gantry: 
.. |gantry_sensors| image:: ../pics/gantry-sensors.png
    :width: 600
.. _gantry_sensors: 
