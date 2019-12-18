**About Terra-Ref**
-------------------

Modern agriculture has made great progress in reducing hunger and poverty and improving food security and nutrition but still faces tremendous challenges in the coming decades. In order to accelerate plant breeding, we need novel high-throughput phenotyping (HTP) approaches to advance the understanding of genotype-to-phenotype. The Transportation Energy Resources from Renewable Agriculture Phenotyping Reference Platform (TERRA-REF) is one such program that aims to transform plant breeding by using remote sensing to quantify plant traits. The TERRA-REF project provides a data and computation pipeline responsible for collecting, transferring, processing and distributing large volumes of crop sensing and genomic data.

**Gantry Sensors**
------------------

The Lemnatec Scanalyzer Field System is a high-throughput phenotyping field-scanning robot that autonomously moves and continuously collects images of the crops it hovers. Attached to the 30-ton steel gantry of the field-scanning robot are sensors and cameras that collect different sets of data. The diverse array of sensors allow researchers to collect significant sets of data that can be used to leverage biological insight into how environments affect phenotypes and the overall relationship between genotypes (gene) and phenotypes (characteristic). Below are three sensors specific to this project:

  
  **Field Scanning Imaging Sensors**
  
  **INSERT IMAGE HERE**
  
  *StereoRGB*
  
  	The Stereo RGB camera is a camera that captures images from above which enables researchers to determine canopy cover (spread of plants), the  amount of crops, etc.
	
  *Scanner3D*
  
  	A 3D scanner that captures the architecture of plants, such as leaf angles and shapes.
	
  *PSII Fluorescence*
  
  	A camera that allows researchers to understand how efficient plants are at photosynthesizing.
	


**TERRA-REF Data Volume**
--------------------------

- The gantry system outputs 1 TB of data per day from various sensors

	
**Need for data analysis workflows**
------------------------------------

- Existing cyber-infrastructure for the TERRA-REF platform lacks workflow features and has complex dependencies which make it difficult to deploy on 
- The TERRA-REF platform has primarily been used to characterize a single crop (Sorghum) and as more researchers look to using the scanalyzer for various crops, it is pertinent to have a data-analysis workflows that suit their needs and capacity.

**Computing:**

- Requires large computation of data
- Each step in the Terra-ref pipeline requires interaction with a database 
- RabbitMQ lacks workflow features
- Complex dependencies

**Development:**

- Monitoring and reprocessing is time intensive
- Difficult to add new algorithms
- Not clear how to reuse and adapt components


**PhytoOracle Solution**
------------------------


	- Modular
	- Scalable
	- Customizable
	- Fault tolerant
	- 



**Concept Maps**
----------------

