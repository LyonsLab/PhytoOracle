# PhytoOracle

## Overview

PhytoOracle is a collection of scalable and modular data pipelines that aim to improve data processing for phenomics research. PhytoOracle is based on the Master-Worker Framework enabled by the CCTools’ Makeflow and WorkQueue. This framework allows for job distribution on High Performance Computers (HPC) or Cloud systems. PhytoOracle's Master is designed to deploy on HPC Interactive Nodes but is also possible to deploy on Cloud Virtual Machines (VMs). 

## Pipelines and Data

+ Canopy cover data through the StereoTopRGB pipeline
+ Infrared data through the FlirIr pipeline
+ Photosyntetic activity through the PSII pipeline (beta)
+ Field and plant structure data through the 3D pipeline (alpha)
+ Hyperspectral data (TBA)

## System Requirements

+ [CCTools 7.0.21](http://ccl.cse.nd.edu/software/downloadfiles.php)
+ [Singularity 3.5.1]()
+ [iRODS Client]()

## Getting Started

+ See detailed documentation [here](https://phytooracle.readthedocs.io/en/latest/contents.html) for instruction manuals.

## Sensor Modules available

+ stereoTop-RGB

### Sensor Modules under-construction
+ scanner3D
+ PSII Flourescence

## Distributed Scaling

- PhytoOracle employs distributed-scaling capabilities from [CCTools](https://cctools.readthedocs.io/en/latest/) suite. 

- Read more about how to scale in the workflow scaling section [here](https://phytooracle.readthedocs.io/en/latest/workflows.html)

## License 

Licensed under the **MIT License**

## Author Description

PhytoOracle is a class project undertaken by under-graduate and graduate students taking the “Applied Concepts in Cyberinfrastructure” course, 2019 at the University of Arizona taught by Dr. Nirav Merchant and Dr. Eric Lyons.

## Where to get additional help

+ Need help? Found a bug? Raise an issue on our github page [here](https://github.com/uacic/PhytoOracle/issues)
+ E-mail [Maintainer@PhytoOracle](sateeshp@email.arizona.edu)

## Acknowledgements

This material based upon work supported by Cyverse & CCTools. Cyverse is based upon work supported by the National Science Foundation under Grant Numbers: DBI-0735191, DBI-1265383, DBI-1743442. CCTools is based upon work supported by the National Science Foundation under Grant Numbers: CCF-0621434 and CNS-0643229.
