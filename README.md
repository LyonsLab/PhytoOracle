# PhytoOracle

## Overview

PhytoOracle is a scalable, modular data pipeline that aims to improve data processing for phenomics research. It refines the TERRA-REF AgPipeline by using CCTools’ Makeflow and WorkQueue frameworks for distributed task management. The program is available as Docker/Singularity containers that can run on either local, cloud, or high-performance computing (HPC) platforms

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
