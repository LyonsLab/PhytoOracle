***********************************************
Using PhytoOracle on the Cloud with HPC Support
***********************************************

Overview
========

This guide will walk you through the necessary steps required to launch PhytoOracle's pipelines onto a Clould system with HPC support. 
With this setup, the Cloud based Virtual Machine (VM) will act like the Master, whilst the HPC will act like the Worker, in a master-worker distributive framework.

This setup was successfully tested on CyVerse's Atmosphere and is now being tested on NSF's JetStream Cloud computing platform. 
In principle, this setup should be working on all commercial Cloud services. 
PhytoOracle's repositories will be downloaded on the Master VM, and jobs will be distributed to the Worker HPC nodes using pbs scripts initialized from within the HPC login node (tested with University of Arizona's HPC system).


Master VM Hardware and Software Requirements
============================================

Before installing software, make sure that your VM has at least 500 GB of Hard Disk space. 
This is because the raw data can meansure upward of 120 GB depending on what pipeline you use (e.g. FlirIr raw data is only 5.8 GB whilst RGB raw data is approximately 128 GB). 
You will need the extra working space due to the output and temporary files. 

Software wise, on your Master VM you should have installed:

- Python 3 (tested with python v 3.8)
- Singularity (tested with singularity v 3.5.3)
- `CCTools <https://ccl.cse.nd.edu/software/downloadfiles.php>`_ (tested with CCTools v 7.0.19)
- iRODS (tested with iRODS v 4.2.7)
- Git (tested with Git v 1.7.1)
- Apache2 (tested with Apache2 v 2.4.29)
- Nginx (tested with nginx v 1.14.0)

CCTools is avaialbe to install and run without root permissions. Dowload and store CCTools in your `home` directory.

Upon installing all required software onto the Master VM, clone the PhytoOracle repository with

.. code::
   git clone https://github.com/uacic/PhytoOracle


:code:`cd` (change directory) to your required pipeline, and you're done!

Before proceeding, note your Master VM :code:`IP` address, which will be required when launching worker nodes on the HPC.

HPC Software Requirements
=========================

Look at documentation from your HPC provider, ensure that the HPC is running CentOS 7 and has these software installed:

- Python 3 (tested with python v 3.8)
- Singularity (tested with singularity v 3.5.3)
- `CCTools <https://ccl.cse.nd.edu/software/downloadfiles.php>`_ (tested with CCTools v 7.0.19)
- iRODS (tested with iRODS v 4.2.7)
- Git (tested with Git v 1.7.1)

CCTools is avaialbe to install and run without root permissions. Dowload and store CCTools in your :code:`home` directory; if Python3, Singularity, iRODS, Git are not installed, please contact your HPC provider.

Launching Workers
=================
To launch workers, go to your HPC and use a `.pbs` script to request for resources for your jobs. Using your preferred editor, create a `.pbs` script and paste the following lines:

.. code::

   #!/bin/bash
   #PBS -W group_list=<your_group_list>
   #PBS -q <priority>
   #PBS -l select=<#_nodes>:ncpus=<CPU_#>:mem=<RAM_#>gb
   #PBS -l place=pack:shared
   #PBS -l walltime=<max_hour_#>:00:00  
   #PBS -l cput=<max_compute_#>:00:00
   module load singularity 

   export CCTOOLS_HOME=/home/<u_#>/<username>/cctools-<version>
   export PATH=${CCTOOLS_HOME}/bin:$PATH

   cd /home/<u_#>/<username>

   # Repeat the following line with as many transformers requried
   singularity pull docker://agpipeline/<transformer>

   /home/u12/cosi/cctools-7.0.19-x86_64-centos7/bin/resource_monitor -O log-flirIr-makeflow -i 2 -- work_queue_factory -T local <MASTER_VM_IP_ADDRESSS> 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 900

Change the highlighted :code:`<fields>` to preferred settings. Save your changes and submit with 

.. code::

   qsub <name>.pbs

Depending on the traffic on the HPC and on the set priorities, wait for workers to become avaialbe before launch.

A working example on the University of Arizona's HPC runinng the FliIr pipeline is

.. code::

   #!/bin/bash
   #PBS -W group_list=<group_list>
   #PBS -q standard
   #PBS -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True
   #PBS -l place=pack:shared
   #PBS -l walltime=24:00:00  
   #PBS -l cput=384:00:00
   module load singularity

   export CCTOOLS_HOME=/home/u12/cosi/cctools-7.0.19-x86_64-centos7
   export PATH=${CCTOOLS_HOME}/bin:$PATH

   cd /home/u12/cosi/

   singularity pull docker://agpipeline/cleanmetadata:2.0
   singularity pull docker://agpipeline/flir2tif:2.2
   singularity pull docker://agpipeline/meantemp:3.0
   singularity pull docker://agpipeline/bin2tif:2.0

   /home/u12/cosi/cctools-7.0.19-x86_64-centos7/bin/resource_monitor -O log-flirIr-makeflow -i 2 -- work_queue_factory -T local 128.196.142.26 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 9000


**Your setup on the Cloud with HPC support is now complete! Please go to the pipeline of your choice to continue with running and processing.**