****************************
Using PhytoOracle on the HPC
****************************

Overview
========

This guide will walk you through the necessary steps required to launch PhytoOracle's pipelines onto a High Performance Computer system using interactive nodes (as tested on the University of Arizona's HPC).

PhytoOracle's repositories will be downloaded on the interactive node which will act as the Master, and jobs will be distributed to the Workers nodes using :code:`pbs` scripts.

Software Requirements
=====================

Look at documentation from your HPC provider, ensure that the HPC is running CentOS 7 and has these software installed:

+ Python 3 (tested with python v 3.8)
+ Singularity (tested with singularity v 3.5.3)
+ `CCTools <https://ccl.cse.nd.edu/software/downloadfiles.php>`_ (tested with CCTools v 7.0.19)
+ iRODS (tested with iRODS v 4.2.7)
+ Git (tested with Git v 1.7.1)


Launching an interactive node and accessing PhytoOracle
=======================================================

To launch an interactive node,:

.. code::
   
   qsub -I -N phytooracle -W group_list=<your_group_list> -q <priority> -l select=1:ncpus=<CPU_N>:mem=<RAM_N>gb:np100s=1:os7=True -l walltime=<max_hour_N>:0:0

replace :code:`<your_group_list>, <priority>, <CPU_N>, <RAM_N>, <max_hour_N>` with your preferred settings.

When the interactive node is done loading, clone the PhytoOracle repository with:

.. code::

   git clone https://github.com/uacic/PhytoOracle


:code:`cd` (change directory) to your required pipeline, and you're done!

Before proceeding, note the IP address of the interactive node. You can find the IP address with :config:`ifconfig`.

The IP address will be needed for configuring workers.

Launching Workers
=================

To launch workers, in a PBS Pro environment 

Using your preferred editor, create a :code:`.pbs` script and paste the following lines:

.. code::

   #!/bin/bash
   #PBS -W group_list=<your_group_list>
   #PBS -q <priority>
   #PBS -l select=<N_nodes>:ncpus=<CPU_N>:mem=<RAM_N>gb
   #PBS -l place=pack:shared
   #PBS -l walltime=<max_hour_N>:00:00  
   #PBS -l cput=<max_compute_N>:00:00
   module load singularity 

   export CCTOOLS_HOME=/home/<u_N>/<username>/cctools-<version>
   export PATH=${CCTOOLS_HOME}/bin:$PATH

   # This might change according with your RJMS system
   cd /home/<u_N>/<username>

   # Repeat the following line with as many transformers requried
   singularity pull docker://agpipeline/<transformer>

   /home/u12/cosi/cctools-7.0.19-x86_64-centos7/bin/resource_monitor -O log-flirIr-makeflow -i 2 -- work_queue_factory -T local <INTERACTIVE_NODE_ADDRESS>.<HPC_SYSTEM> 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 900

As before, change the highlighted :code:`<fields>` to preferred settings. Save your changes and submit with 

.. code::

   qsub <name>.pbs

Depending on the utilization of the HPC system it might take some time before the workers launch.

A working example on the University of Arizona's HPC running the FliIr pipeline is

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

   /home/u12/cosi/cctools-7.0.19-x86_64-centos7/bin/resource_monitor -O log-flirIr-makeflow -i 2 -- work_queue_factory -T local i18n9.ocelote.hpc.arizona.edu 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 900


**Your setup on the HPC is now complete. You have the Master and Worker running, you can now run the pipeline of your choice:**

+ `StereoTopRGB <https://phytooracle.readthedocs.io/en/latest/4_StereoTopRGB_run.html>`_
+ `flirIr <https://phytooracle.readthedocs.io/en/latest/5_FlirIr_run.html>`_
+ PSII
+ Stereo3DTop
+ Hyperspectral