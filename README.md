# PhytoOracle
<p align="center">
    <img src="pics/PhytoOracle_logo.PNG" width="300" height="300" />
<p>
PhytoOracle is a scalable, distributed workflow manager for analyzing highthroughput phenotyping data. It is designed to process data from the [UA Gantry](https://www.youtube.com/watch?v=da2gKRdMeXY), but can be adapted to work on data coming from other platforms.  
PhytoOracle uses a master-worker framework for distributed computing (HPC, Cloud, etc.) and can run jobs on nearly all unix-like environments. 

## Documentation

See detailed documentation [here](https://phytooracle.readthedocs.io) for instruction manuals. 

## Supported Sensors and Pipelines

+ [StereoTopRGB](https://phytooracle.readthedocs.io/en/latest/4_StereoTopRGB_run.html)
+ [FlirIr](https://phytooracle.readthedocs.io/en/latest/5_FlirIr_run.html)
+ [PSII](https://phytooracle.readthedocs.io/en/latest/7_PSII_run.html)
+ [StereoTop3D](https://phytooracle.readthedocs.io/en/latest/8_3D_run.html)
+ Hyperspectral (VNIR/SWIR)

**For more information on types and description of each camera used, access the documentation above.**

## Resources

+ [Code availability (GitHub)](https://github.com/phytooracle)
+ [Containers repository (DockerHub)](https://hub.docker.com/u/phytooracle)

## License 

PhytoOracle is licensed under the **MIT License**.

## Issues and Questions

Need help? Found a bug? Raise an issue on our github page [here](https://github.com/LyonsLab/PhytoOracle/issues).

**For specific workflows and adapting a pipeline for your own work contact:**

+ Emmanuel Gonzalez: emmanuelgonzalez [at] email.arizona.edu

+ Michele Cosi: cosi [at] email.arizona.edu

**For plant detection and plant clustering:**

+ Travis Simmons: travis.simmons [at] ccga.edu

**For the orthomosaicing algorithm:**

+ Ariyan Zarei: ariyanzarei [at] email.arizona.edu

## Acknowledgements

This project partially built on code initially developed by the [TERRA-REF](https://www.terraref.org/) project and [AgPipeline](https://github.com/AgPipeline/) team. We thank the University of Arizona Advanced Cyberinfrastrcture Concept class of 2019 for additional work.

This material based upon work supported by Cyverse & CCTools. Cyverse is based upon work supported by the National Science Foundation under Grant Numbers: DBI-0735191, DBI-1265383, DBI-1743442. CCTools is based upon work supported by the National Science Foundation under Grant Numbers: CCF-0621434 and CNS-0643229. 
