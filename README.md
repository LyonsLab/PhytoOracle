# PhytoOracle

## Overview

PhytoOracle is a collection of scalable and modular data pipelines that aim to improve data processing for phenomics research. PhytoOracle is based on the Master-Worker Framework enabled by the CCTools’ Makeflow and WorkQueue. This framework allows for job distribution on High Performance Computers (HPC) or Cloud systems. PhytoOracle's Master is designed to deploy on HPC Interactive Nodes but is also possible to deploy on Cloud Virtual Machines (VMs). 

## Pipelines and Data

+ Canopy cover data through the [StereoTopRGB](https://github.com/uacic/PhytoOracle/tree/alpha/stereoTopRGB) pipeline
+ Infrared data through the [FlirIr](https://github.com/uacic/PhytoOracle/tree/alpha/FlirIr) pipeline
+ Photosyntetic activity through the [PSII](https://github.com/uacic/PhytoOracle/tree/alpha/psII) pipeline (beta)
+ Field and plant structure data through the [Scanner3DTop](https://github.com/uacic/PhytoOracle/tree/alpha/scanner3DTop) pipeline (alpha)
+ Hyperspectral data (TBA)

## Getting Started

+ Read on [HPC deployment here](https://github.com/uacic/PhytoOracle/blob/alpha/HPC_Install.md) or [Cloud with HPC support deployment here](https://github.com/uacic/PhytoOracle/blob/alpha/CloudHPC_installation.md). 
+ See detailed documentation [here](https://phytooracle.readthedocs.io/en/latest/contents.html) for instruction manuals (to be updated).

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
