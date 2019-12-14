**Introduction to PhytoOracle**
===============================
PhytoOracle is a scalable, modular data pipeline that aims to improve data processing for phenomics research. It refines the TERRA-REF AgPipeline by using `CCTools\' <http://ccl.cse.nd.edu/software/>`_ Makeflow and WorkQueue frameworks for distributed task management. The program is available as Docker/Singularity containers that can run on either local, cloud, or high-performance computing (HPC) platforms. PhytoOracle makes it easy to swap between available extractors **(link)** or to integrate new extractors depending on the research needs. As a result, PhytoOracle significantly reduces data processing time, thereby enabling faster leveraging of biological insight from the data.

**About Terra-Ref**
-------------------

Modern agriculture has made great progress in reducing hunger and poverty and improving food security and nutrition but still faces tremendous challenges in the coming decades. In order to accelerate plant breeding, we need novel high-throughput phenotyping (HTP) approaches to advance the understanding of genotype-to-phenotype. The Transportation Energy Resources from Renewable Agriculture Phenotyping Reference Platform (TERRA-REF) is one such program that aims to transform plant breeding by using remote sensing to quantify plant traits. The TERRA-REF project provides a data and computation pipeline responsible for collecting, transferring, processing and distributing large volumes of crop sensing and genomic data.

**Terra-ref Sensors**
---------------------

The Lemnatec Scanalyzer Field System is a high-throughput phenotyping field-scanning robot that autonomously moves and continuously collects images of the crops it hovers. Attached to the 30-ton steel gantry of the field-scanning robot are sensors and cameras that collect different sets of data. The diverse array of sensors allow researchers to collect significant sets of data that can be used to leverage biological insight into how environments affect phenotypes and the overall relationship between genotypes (gene) and phenotypes (characteristic). Below are three sensors specific to this project:

  **Field Scanning Imaging Sensors**
  
   *Stereo RGB Camera*
    A camera that captures images from above which enables researchers to determine canopy cover (spread of plants), the  amount of crops, etc.

   *3D Laser Scanner (LIDAR)*
    A 3D scanner that captures the architecture of plants, such as leaf angles and shapes.
    
   *PSII Fluorescence Response Camera*
    A camera that allows researchers to understand how efficient plants are at photosynthesizing.

	
**Pain points of Terra-ref pipeline**
-------------------------------------

**Computing:**

- Requires large computation of data
- Each step in the Terra-ref pipeline requires interaction with a database 
- RabbitMQ lacks workflow features
- Complex dependencies

**Development:**

- Monitoring and reprocessing is time intensive
- Difficult to add new algorithms
- Not clear how to reuse and adapt components

**Solution**
------------

Establish a generalized workflow that includes a template extractor, which will enable a lower barrier for contributors and reduce the effort for developers.

**Intro to CC tools**
---------------------

The Cooperating Computing Tools `(CCTools) <http://ccl.cse.nd.edu/software/>`_ help design and deploy scalable applications that run on hundreds or thousands of machines at once. Work Queue within CCTools is a framework for building large master-worker applications that span thousands of machines drawn from clusters, clouds, and grids. 

CCTool's `ReadtheDocs 
<https://www.cctools.readthedocs.io/en/latest/about/>`_

**Concept Maps**
----------------

