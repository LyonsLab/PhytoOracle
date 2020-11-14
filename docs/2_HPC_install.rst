****************************
Using PhytoOracle on the HPC
****************************

Overview
========

This guide will walk you through the necessary steps required to launch PhytoOracle's pipelines onto a High Performance Computer system using interactive nodes (as tested on the University of Arizona's HPC running the PBS Pro and SLURM RJMS - Resource and Job Management System).

The interactive node functions as the "foreman" within the time-saving "foreman-worker" framework. The interactive node distributes the computational load, connecting to the "workers" through its IP address and job management scripts from the PhytoOracle repository. 

Software Requirements
=====================

Ensure that your HPC is running CentOS 7 and has these software installed:

+ `Python 3 <https://www.python.org/downloads/>`_ (tested with python v 3.8)
+ `Singularity <https://sylabs.io/docs/>`_ (tested with singularity v 3.5.3)
+ `CCTools <https://ccl.cse.nd.edu/software/downloadfiles.php>`_ (tested with CCTools v 7.0.19)
+ `iRODS <https://docs.irods.org/4.2.8/>`_ (tested with iRODS v 4.2.7)
+ `Git <https://git-scm.com/>`_ (tested with Git v 1.7.1)

Launching Interactive Node
===========================

To launch an interactive node:

.. code::
   
   qsub -I -N phytooracle -W group_list=<your_group_list> -q <priority> -l select=1:ncpus=<CPU_N>:mem=<RAM_N>gb:np100s=1:os7=True -l walltime=<max_hour_N>:0:0

or

.. code::

   srun --nodes=1 --mem=<RAM_N> --ntasks=1 --cpus-per-task=<CPU_N> --time=<max_hour_N> --job-name=po_mcomp --account=<your_group_list> --partition=<priority> --mpi=pmi2 --pty bash -i

replace :code:`<your_group_list>, <priority>, <CPU_N>, <RAM_N>, <max_hour_N>` with your preferred settings.

An example on the UA HPC is:

.. code:: 
   
   qsub -I -N phytooracle -W group_list=lyons_lab -q standard -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True -l walltime=12:0:0

or

.. code::

   srun --nodes=1 --mem=470GB --ntasks=1 --cpus-per-task=94 --time=24:00:00 --job-name=po_mcomp --account=lyons-lab --partition=standard --mpi=pmi2 --pty bash -i

Once the interactive node is running, clone the PhytoOracle repository:

.. code::

   git clone https://github.com/uacic/PhytoOracle

Before proceeding, note the IP address of the interactive node. You can find the IP address with :code:`ifconfig`. This will be used for connecting the manager to the workers.

:code:`cd` (change directory) into the desired pipeline and continue.

Launching Workers
=================

Create an executable script according to your HPC's RJMS system. If using PBS Pro, use your preferred editor to create a :code:`.pbs` script, if using SLURM create a :code:`.sh` using the following templates:

PBS Pro:

.. code::

   #!/bin/bash
   #PBS -W group_list=<your_group_list>
   #PBS -q <priority>
   #PBS -l select=<N_nodes>:ncpus=<CPU_N>:mem=<RAM_N>gb
   #PBS -l place=pack:shared
   #PBS -l walltime=<max_hour_N>:00:00  
   #PBS -l cput=<max_compute_N>:00:00
   module load singularity 

   export CCTOOLS_HOME=/home/<u_num>/<username>/cctools-<version>
   export PATH=${CCTOOLS_HOME}/bin:$PATH

   /home/<U_ID>/<USERNAME>/cctools-<version>/bin/resource_monitor -O log-flirIr-makeflow -i 2 -- work_queue_factory -T local <INTERACTIVE_NODE_ADDRESS>.<HPC_SYSTEM> 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 900

SLURM:

.. code::

   #!/bin/bash 
   #SBATCH --account=lyons-lab --partition=standard
   #SBATCH --job-name="phytooracle"
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --ntasks-per-node=94
   #SBATCH --time=24:00:00
   #module load singularity
   module load python/3.8

   export CCTOOLS_HOME=/home/<u_num>/<username>/cctools-<version>
   export PATH=${CCTOOLS_HOME}/bin:$PATH

   /home/<U_ID>/<USERNAME>/cctools-<version>/bin/work_queue_worker -M PhytoOracle_FLIR -t 900

As before, change the highlighted :code:`<fields>` to preferred settings. 

Here are examples on the UA HPC system, using "u1" as the user number and "hpcuser" as the username, looks like:

PBS Pro:

.. code:: 

   #!/bin/bash
   #PBS -q standard
   #PBS -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True
   #PBS -W group_list=lyons-lab
   #PBS -l place=pack:shared
   #PBS -l walltime=5:00:00
   #PBS -l cput=140:00:00
   #module load unsupported
   #module load ferng/glibc
   module load singularity

   export CCTOOLS_HOME=/home/u1/hpcuser/cctools-7.1.5-x86_64-centos7
   export PATH=${CCTOOLS_HOME}/bin:$PATH
   cd /home/u1/hpcuser/data_output_folder

   #RGB
   #singularity pull docker://agpipeline/cleanmetadata:2.2
   #singularity pull docker://agpipeline/bin2tif:2.0
   #singularity pull docker://zhxu73/gistools:latest
   #singularity pull docker://emmanuelgonzalez/plotclip_shp:latest

   #FlirIR
   #singularity pull docker://agpipeline/cleanmetadata:2.2
   #singularity pull docker://agpipeline/flir2tif:2.2
   #singularity pull docker://agpipeline/meantemp:3.0
   #singularity pull docker://agpipeline/bin2tif:2.0

   /home/u1/hpcuser/cctools-7.1.5-x86_64-centos7/bin/work_queue_factory -T local <commander_IP_address>.ocelote.hpc.arizona.edu 9123 -w 24 -W 26 --workers-per-cycle 10 --cores=1 -t 900

It is important to note that lines 12, 14, and 27 will have to be personalized, and the commander IP address must be specified in line 27.

SLURM:

.. code::

   #!/bin/bash 
   #SBATCH --account=windfall --partition=windfall
   #SBATCH --job-name="phytooracle"
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --ntasks-per-node=94
   #SBATCH --time=24:00:00
   #module load singularity
   module load python/3.8

   export CCTOOLS_HOME=/home/u12/cosi/cctools-7.1.6-x86_64-centos7
   export PATH=${CCTOOLS_HOME}/bin:$PATH

   /home/u1/hpcuser/cctools-7.1.6-x86_64-centos7/bin/work_queue_worker -M PhytoOracle_FLIR -t 900

Save your changes and submit with: 

PBS Pro

.. code::

   qsub <filename>.pbs

SLURM

.. code::

   sbatch <filename.pbs>

Depending on the traffic to the HPC system, this may take some time. You can search for your submitted job using:

PBS Pro

.. code:: 

   qstat -u username

SLURM

.. code::

   squeue -u username

**The HPC setup is now complete. Navigate to the pipeline of your choice to continue:**

+ `StereoTopRGB <https://phytooracle.readthedocs.io/en/latest/4_StereoTopRGB_run.html>`_
+ `FlirIr <https://phytooracle.readthedocs.io/en/latest/5_FlirIr_run.html>`_
+ PSII
+ Stereo3DTop
+ Hyperspectral