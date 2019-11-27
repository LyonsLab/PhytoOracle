|CyVerse logo|_

|Home_Icon|_
`Learning Center Home <http://learning.cyverse.org/>`_

starBlast
=========

.. contents::

What is starBlast?
------------------

starBlast is a distributed scaling of BLAST using sequence-server and cctools.

.. note::
   
   starBlast is a project undertaken by under-graduate and graduate students taking the "Applied Concepts in Cyberinfrastructure" course, 2019 at University of Arizona taught by Dr. Nirav Merchant and Dr. Eric Lyons.

starBlast-VICE
~~~~~~~~~~~~~~

starBlast-VICE is a Visual and Interactive Computing Environment in CyVerse’s Discovery Environment (DE) for running interactive apps such as sequence-server. 


starBlast-Dockers
~~~~~~~~~~~~~~~~~~~~~~~~~~

starBlast-Docker containers can be deployed on any cloud such as Atmosphere for distributed scaling of BLAST searches.

starBlast-HPC
~~~~~~~~~~~~~

starBlast-HPC is a solution to distribute large number of queries on a High Performance Computer.


What is sequenceServer?
-----------------------

SequenceServer lets you rapidly set up a BLAST+ server for individual use or for sharing datasets with your colleagues and your community. Read about SequenceServer's user-centric design and sustainable software development philosophy `here <https://doi.org/10.1093/molbev/msz185>`_.

What is Work_Queue?
-------------------

Work Queue is a system and library for creating and managing scalable master-worker style programs that scale up to thousands of machines on clusters, clouds, and grids.


Platform(s)
-----------

*We will use the following CyVerse platform(s):*

.. list-table::
    :header-rows: 1

    * - Solution
      - Platform
      - Capacity
      - Link
      - Platform Documentation
      - Learning Center Docs
    * - starBlast VICE
      - Discovery Environment
      - 5-25 Students
      - `Discovery Environment <https://de.cyverse.org/de/>`_
      - `DE Manual <https://wiki.cyverse.org/wiki/display/DEmanual/Table+of+Contents>`_
      - `Guide <https://learning.cyverse.org/projects/discovery-environment-guide/en/latest/>`__
    * - starBlast Atmosphere
      - Atmosphere Cloud / Docker
      - 25-50 Students
      - `Atmosphere <https://atmo.cyverse.org/de/>`_
      - `Atmosphere Manual <https://wiki.cyverse.org/wiki/display/DEmanual/Table+of+Contents>`_
      - `Guide <>`__
    * - starBlast HPC
      - HPC & Atmosphere Cloud
      - 50+ Students
      - `cctools <https://atmo.cyverse.org/de/>`_
      - `PBS on HPC  <https://public.confluence.arizona.edu/display/UAHPC>`_
      - `Workqueue <https://cctools.readthedocs.io/en/latest/work_queue/>`__

----

starBlast-VICE Setup
--------------------

.. note::

   Requires access to Cyverse `Discovery Environment <https://de.cyverse.org/de/>`_
   
1. Click on the following button to quick-launch sequence-server in CyVerse Discovery Environment with two blast databases (Human_GRCh38_p12 & Mouse_GRCm38_p4)

	|sequenceServer|_
	
2. Click [Launch Analysis]
3. Check the notifications Bell Icon for a link to your sequenceserver instance
4. Start BLASTING

To set up a custom database on the VICE platform, see the appendix section.

----

starBlast-Docker on Atmosphere Cloud
------------------------------------

To deploy starBlast setup on CyVerse Atmosphere cloud, you will need access to `Atmosphere <https://atmo.cyverse.org/application/images>`_. Request access to Atmosphere from your `CyVerse user account <https://user.cyverse.org>`_.

You will need to launch a Master instance that will host sequenceServer and one or more Worker instances as needed to distribute the blast jobs. 

Both the Master and Worker Virtual Machine instances use Docker containers to run sequenceServer and connect Workers. 

Launching Master & Worker Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Go to https://atmo.cyverse.org and log in with your Cyverse Username and Password
2. Launch a Master (medium1) instance which will broadcast as a Master using `this <https://atmo.cyverse.org/application/images/1759>`_ image with docker preinstalled.
3. Launch a Worker (XLarge1) instance which will connect to the Master using `this <https://atmo.cyverse.org/application/images/1759>`_ image with docker preinstalled.
4. When the instances are ready showing Active (with a green dot), login to your Master and Worker instances using ssh <CYVERSE_USERNAME>@<MASTER_VM_IP_ADDRESS> and enter your cyverse password.


Setting Up Master Docker
~~~~~~~~~~~~~~~~~~~~~~~~

Copy and pase the following code in the Master instance to launch sequenceServer with two databases (Human_GRCh38_p12 & Mouse_GRCm38_p4) ready to distribute BLAST queries to workers

.. code:: 

   docker run -ti -p 80:3000 -p 9123:9123 -e PROJECT_NAME=starBlast -e WORKQUEUE_PASSWORD= -e BLAST_NUM_THREADS=4 zhxu73/sequenceserver-scale
   
.. note::
	
   It might take 2-5min to download the databases from CyVerse data store	
   
Setting Up Worker Docker
~~~~~~~~~~~~~~~~~~~~~~~~

Copy and paste the following code into your Worker instance to connect the Worker docker to the Master docker. The Worker knows where to find the master by the environmental variable PROJECT_NAME set as above. 

.. code:: 

   docker run -ti --net=host -e PROJECT_NAME=starBlast -e WORKQUEUE_PASSWORD= -e BLAST_NUM_THREADS=4 -e NUM_WORKER=2 zhxu73/sequenceserver-scale-worker
   
Start Blasting
~~~~~~~~~~~~~~

Now, anyone can open a web-browser and go to <MASTER_VM_IP_ADDRESS> to access sequence-Server front-end and start BLASTING!

.. code::

   <MASTER_VM_IP_ADDRESS>

----

starBlast-HPC Setup
-------------------

The starBlast-HPC Setup  was conceived for groups that wish a larger quantity of power.  

In order to achieve a successful setup of the starBlast HPC system, a small amount of command line knowledge is required.

Similar to the starBlast-Atmosphere Cloud,  the starBlast HPC system has a Master-Worker set up: a dockerized atmosphere VM machine acts as the Master, and the HPC acts as the Worker. It is suggested that the Worker is set up well ahead of time.

Setting Up the Worker HPC
~~~~~~~~~~~~~~~~~~~~~~~~

It is important that the following software are installed on the HPC:

- glibc version 2.14 or newer, 

- ncbi-blast+ version 2.6.0 or newer (ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.9.0+-src.tar.gz)

- CCTools (cctools-7.0.21-x86_64-centos7.tar.gz)

Put both ncbi-blast+ and CCTools in your home directory.
Databases need to be downloaded in a personal directory in the home folder.

.. code::

   /home/<U_NUMBER>/<USER>/Database
   
The HPC uses a .pbs and qsub system to submit jobs.

Create a .pbs file that contains the following code and change the <VARIABLES> to preferred options:

.. code::

   #!/bin/bash
   #PBS -W group_list=<GROUP_NAME>
   #PBS -q <QUEUE_TYPE>
   #PBS -l select=<NUMBER_OF_NODES>:ncpus=<NUMBER_OF_CPUS_PER_NODE>:mem=<NUMBER_OF_RAM_PER_NODE>gb
   #PBS -l place=pack:shared
   #PBS -l walltime=<WALLTIME_REQUIRED>
   #PBS -l cput=<WALLTIME_REQUIRED>
   module load unsupported
   module load ferng/glibc
   export CCTOOLS_HOME=/home/<U_NUMBER>/<USER>/cctools-7.0.19-x86_64-centos7
   export PATH=${CCTOOLS_HOME}/bin:$PATH
   export PATH=$PATH:/home/<U_NUMBER>/<USER>/ncbi-blast-2.9.0+/bin
   /home/<U_NUMBER>/<USER>/cctools-7.0.19-x86_64-centos7/bin/work_queue_factory -M starBLAST -T local -w <NUMBER_OF_WORKERS>

An example of a .pbs file running on the University of Arizona HPC:

.. code::

   #!/bin/bash
   #PBS -W group_list=ericlyons
   #PBS -q windfall
   #PBS -l select=2:ncpus=6:mem=24gb
   #PBS -l place=pack:shared
   #PBS -l walltime=02:00:00
   #PBS -l cput=02:00:00
   module load unsupported
   module load ferng/glibc
   module load blast
   export CCTOOLS_HOME=/home/u12/cosi/cctools-7.0.19-x86_64-centos7
   export PATH=${CCTOOLS_HOME}/bin:$PATH
   cd /home/u12/cosi/cosi-workers
   /home/u12/cosi/cctools-7.0.19-x86_64-centos7/bin/work_queue_factory -M starBLAST -T local -w 2

In the example above, the user already has blast installed (calls it using “module load blast“). The script will submit to the HPC nodes a total of 2 workers.

Submit the .pbs script with 

.. code::
    
   qsub <NAME_OF_PBS>.pbs
   
Setting Up Master Docker
~~~~~~~~~~~~~~~~~~~~~~~~

Copy and paste the following code in the Master instance to launch sequenceServer with two databases (Human_GRCh38_p12 & Mouse_GRCm38_p4) ready to distribute BLAST queries to workers

IMPORTANT: THE PATH TO THE DATABASE ON THE MASTER NEED TO BE THE SAME AS THE ONE ON THE WORKER

.. code:: 

   docker run --rm -ti -p 80:3000 -p 9123:9123 -e PROJECT_NAME=starBLAST = -e BLAST_NUM_THREADS=4 -e SEQSERVER_DB_PATH=/home/<U_NUMBER>/<USER>/Database zhxu73/sequenceserver-scale
   
An example is:

.. code:: 

   docker run --rm -ti -p 80:3000 -p 9123:9123 -e PROJECT_NAME=starBLAST = -e BLAST_NUM_THREADS=4 -e SEQSERVER_DB_PATH=/home/u12/cosi/Data zhxu73/sequenceserver-scale
   
In case the user does not have access to iRODS please use:

.. code::

   docker run --rm -ti -p 80:3000 -p 9123:9123 -e PROJECT_NAME=starBLAST -e WORKQUEUE_PASSWORD= -e BLAST_NUM_THREADS=4 -e /home/<U_NUMBER>/<USER>/Database -v $HOME/blastdb:/<U_NUMBER>/<USER>/Database zhxu73/sequenceserver-scale:no-irods
   
.. note::

   The custom Database folder on the Master needs to have read and write permissions
Start BLASTING! Enter the <MASTER_VM_IP_ADDRESS> in your browser using the actual Master IP address.

.. code::

   <MASTER_VM_IP_ADDRESS>
   
----

Appendix A
----------

starBlast concept map
~~~~~~~~~~~~~~~~~~~~~

|concept_map|_

starBlast-Atmosphere Using iRods for Custom Databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the PATH to custom databases on CyVerse Data Store by using the custom IRODS_SYNC_PATH variable 

.. code:: 
   
   -e IRODS_SYNC_PATH=/PATH/TO/Databases

starBlast-VICE Using Custom Databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See documentation and a demo tutorial on launching the sequenceserver VICE app with custom databases `here <https://cyverse-sequenceserver.readthedocs-hosted.com/en/latest/>`_.

starBlast-Docker on any Cloud Using Custom Databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

starBlast (no-irods) docker containers can be run on any cloud platform/s you have access to by supplying the local path to blast databases as follows:

Master/Web Docker

.. code::
   
   docker run -ti -p 80:3000 -p 9123:9123 -e PROJECT_NAME=starBlast -e WORKQUEUE_PASSWORD= -e BLAST_NUM_THREADS=4 --volume=/local_db_path:/var/www/sequenceserver/db zhxu73/sequenceserver-scale:no-irods

Worker Docker

.. code::

   docker run -ti --net=host -e PROJECT_NAME=starBlast -e WORKQUEUE_PASSWORD= -e BLAST_NUM_THREADS=4 -e NUM_WORKER=2 --volume=/local_db_path:/var/www/sequenceserver/db zhxu73/sequenceserver-scale-worker:no-irods
   
.. note::

   Here are some links to private and public cloud service providers:
   
   `XSEDE Jetstream <https://use.jetstream-cloud.org/application/images>`_
   
   `Digital Ocean Cloud <https://www.digitalocean.com/>`_. See more documentation `here <https://github.com/sateeshperi/starBlast/blob/master/DO.rst>`_.
   
   `Google Cloud Platform <https://cloud.google.com/>`_


Making Custom Databases using ncbi_blast_docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read more here at `ncbi docker wiki <https://github.com/ncbi/docker/wiki/Getting-BLAST-databases>`_

----

**Fix or improve this documentation**

- On Github: `Repo link <https://github.com/sateeshperi/starBlast/>`_
- Send feedback: `Tutorials@CyVerse.org <Tutorials@CyVerse.org>`_

----

|Home_Icon|_
`Learning Center Home`_

.. |sequenceServer| image:: https://de.cyverse.org/Powered-By-CyVerse-blue.svg
.. _sequenceServer: https://de.cyverse.org/de/?type=quick-launch&quick-launch-id=0ade6455-4876-49cc-9b37-a29129d9558a&app-id=ab404686-ff20-11e9-a09c-008cfa5ae621

.. |concept_map| image:: ./img/concept_map.png
    :width: 500
    :height: 500
.. _concept_map: http://learning.cyverse.org/

.. |CyVerse logo| image:: ./img/cyverse_rgb.png
    :width: 500
    :height: 100
.. _CyVerse logo: http://learning.cyverse.org/
.. |Home_Icon| image:: ./img/homeicon.png
    :width: 25
    :height: 25
.. _Home_Icon: http://learning.cyverse.org/
.. |discovery_enviornment| raw:: html

    <a href="https://de.cyverse.org/de/" target="_blank">Discovery Environment</a>
