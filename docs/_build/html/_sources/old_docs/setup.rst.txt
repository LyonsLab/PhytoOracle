PhytoOracle HPC Setup Guide
------------

This guide will walk you through the necessary steps required to launch PhytoOracle's pipelines onto a High Performance Computer system using interactive nodes (as tested on the University of Arizona's HPC).

PhytoOracle's repositories will be downloaded on the interactive node which will act as the Master, and jobs will be distributed to the Workers nodes using pbs scripts.

**Required Software**

Look at documentation from your HPC provider, ensure that the HPC is running CentOS 7 and has these software installed:

- Python 3 (tested with python v 3.8)
- Singularity (tested with singularity v 3.5.3)
- [CCTools](https://ccl.cse.nd.edu/software/downloadfiles.php) (tested with CCTools v 7.0.19)
- iRODS (tested with iRODS v 4.2.7)
- Git (tested with Git v 1.7.1)

CCTools is avaialbe to install and run without root permissions. Dowload and store CCTools in your `home` directory; if Python3, Singularity, iRODS, Git are not installed, please contact your HPC provider.

**Launching an interactive node and accessing PhytoOracle**

To launch an interactive node, do:

.. code::
   qsub -I -N phytooracle -W group_list=<your_group_list> -q <priority> -l select=1:ncpus=<CPU_#>:mem=<RAM_#>gb:np100s=1:os7=True -l walltime=<max_hour_#>:0:0

replace `<your_group_list>, <priority>, <CPU_#>, <RAM_#>, <max_hour_#>` with your preferred settings.

When the interactive node is done loading, clone the PhytoOracle repository with

.. code::
   git clone https://github.com/uacic/PhytoOracle

`cd` (change directory) to your required pipeline, and you're done!

Before proceeding, notice that your interactive node has an address: is the number before the `$`; it will be required when launching workers. 

**Launching Workers**

To launch workers, you use a `.pbs` script. Using your preferred editor, create a `.pbs` script and paste the following lines:

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

   /home/u12/cosi/cctools-7.0.19-x86_64-centos7/bin/resource_monitor -O log-flirIr-makeflow -i 2 -- work_queue_factory -T local <INTERACTIVE_NODE_ADDRESS>.<HPC_SYSTEM> 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 900

As before, change the highlighted `<fields>` to preferred settings. Save your changes and submit with 

.. code::
   qsub <name>.pbs

Depending on the traffic on the HPC and on the set priorities, wait for workers to become avaialbe before launch.

A working example on the University of Arizona's HPC runinng the FliIr pipeline is

.. code::
   bash
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

Your setup on the *HPC* is now complete! Please go to the pipeline of your choice to continue with running and processing.

**Manual Installation**

Below are the instructions for installation of CCTools and Singularity on Jetsream or other cloud platforms.

**CCTools (7.0.21)**

- You can install the dependency for compile from source (Ubuntu 18.04 LTS) [here](https://jxuzy.blogspot.com/2019/11/install-cctools-ubuntu-1804lts.html):

- These commands will compile and install cctools (version 7.0.21) to `/usr/bin`, so that they are in the `$PATH`.

.. code:: 

    wget http://ccl.cse.nd.edu/software/files/cctools-7.0.21-source.tar.gz
    tar -xvf cctools-7.0.21-source.tar.gz
    cd cctools-release-7.0.21
    ./configure --prefix /usr
    make -j$(nproc)
    sudo make install


**Singularity 3.5.1** (recommended)

- Install dependencies for singularity

.. code::

    sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup
    
    wget https://dl.google.com/go/go1.13.5.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.13.5.linux-amd64.tar.gz
    echo "export PATH=\$PATH:/usr/local/go/bin" | sudo tee -a /etc/profile
    export PATH=$PATH:/usr/local/go/bin

- Build singularity

.. code::

    wget https://github.com/sylabs/singularity/releases/download/v3.5.1/singularity-3.5.1.tar.gz
    tar -xvf singularity-3.5.1.tar.gz
    cd singularity
    ./mconfig && \
    make -C builddir && \
    sudo make -C builddir install

**Connecting to CyVerse Data Store**

.. code::

   iinit    # initialize irods with your account 
   
   "irods_host": "data.cyverse.org",
   "irods_port": 1247,
   "irods_user_name": "username",
   "irods_zone_name": "iplant"
   
Read more about `Using icommands <https://wiki.cyverse.org/wiki/display/DS/Using+iCommands>`_ here.   


**Known Issues & Caveats**

.. list-table:: Known Issues
   :widths: 25 25 25
   :header-rows: 1

   * - Issue
     - Description
     - Fix
   * - Corrupted Singularity image
     - Happens when multiple worker try to pull into the same default location
     - Pull Singularity containers manually onto each worker before execution
   * - Kernel too old  
     - happens due to incompatibility to run singularity containers
     - Rebuild container using an older base image
   * - Disk space
     - VM need to have space stage the whole data sets
     - 
   * - Workers not connecting to Masters
     - too frequent network IO
     - HTTP server move away the transfer of input files from Makeflow
   * - 
     - 
     - Bundling jobs reduces the number/frequency of network IO for returning result
   * - Makeflow -T local as a job/rule does not work
     - "could" be missing implict dependency
     - 
