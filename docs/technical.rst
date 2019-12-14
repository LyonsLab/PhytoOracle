Technical User Manual
=====================

Integrating new extractors
--------------------------
The Makeflow file allows for easy integration of new extractors.


Preliminary Benchmark Results:
------------------------------

+ How long it took to run the full dataset: 15:40:11
+ Software installation: CCTools, Singularity
+ Time to Stage Data: < 1hr for 9360 folders
+ Average Tasks/minute: 10.25 
+ # Worker Factories connected: 4 xxlarge Jetstream VMs (CPU: 44, MEM: 128GB, Disk: 480GB)
+ Results deposition: Results transferred to CyVerse Data Store using iRODS

Stereotop Benchmarking Workflow Process
-------------------------------
* Get Jetstream account 
* Launch Base Ubuntu image on jetstream
* Setup

.. code::
   
   sudo apt-get update #update references to packages to install latest
   sudo apt-get upgrade -y #update all currently installed packages
   sudo apt-get install -y sysstat git ruby ruby-dev wget python-dev swig zlib1g-dev build-essential perl libperl-dev singularity-  container #Install all required dependencies for cctool and what we need

* Download data
.. code::

   iinit    # initialize irods with your account 
   
   "irods_host": "data.cyverse.org",
   "irods_port": 1247,
   "irods_user_name": "username",
   "irods_zone_name": "iplant"

* Stage data

.. code::

   sudo mkdir /Data #make dir for the data
   sudo chown username /Data #Makes your user own directory 
   irsync -vs i:/iplant/home/raptorslab/stereoTop.tar.gz /Data/stereoTop.tar.gz ##sync data from cyverse to local machine 
   cd /Data
   tar xvzf stereoTop.tar.gz  #extract directory should have stereotop.tar.gz and dir 2018-05-18
   chown -R username 2018-05-15 # Makes your user own directory 
   chmod -R 775 /Data # Change permissions
   (rm -f stereoTop.tar.gz)   # maybe if you want #optional remove tarball to save time don’t need to run this


* Install cctools

.. code::

   git clone git://github.com/cooperative-computing-lab/cctools.git cctools-github-src
   cd cctools-github-src #get latest cctools from github
   ./configure --prefix /opt/cctools #if no errors, you're good
   make 
   sudo make install (copies binaries to desired location)
   sudo cp /opt/cctools/bin/* /usr/local/bin/ (copy binaries to desired location, make sure this works)

* Run Workqueue
.. code::
   
   nohup bash checkFinal.bsh & #will run workqueue (can now close your laptop)
   old? (nohup work_queue_factory 129.114.104.40 9123 -t 9999999 --cores=1 -w 45 &)

* Killing workqueue processes (after it’s run) ##

.. code::

   pkill work_queue_factory
   ps ax | grep work
   kill -9 (pids from previous command)


* Benchmark Script

https://ua-acic.slack.com/files/UMS6Z7FEC/FR4U0FVNX/checkfinal.bsh

This creates 3 output files that we can aggregate and use GNUplot to display in the final presentation.

* MVP

Benchmark each extractor individually


* Launch cctools image (as large as possible 44core last one) on jetstream (or atmosphere?)
https://github.com/uacic/starTerra/tree/master/stereoTop
https://jxuzy.blogspot.com/2019/11/install-cctools-ubuntu-1804lts.html



Running Benchmarks:
* Run this in /opt/src/starTerra-php-template/stereoTop
* Assume you have the setup Tanner lead up through dec 11th.
* Makes the raw data files with number given for example here (2)
python gen_files_list.py 2018-05-15/ 2 > raw_data_files.json

* remove the , at the end of the raw_data_files.json file
.. code::
   php main_wf.php > main_wf.jx
   jx2json main_wf.jx > main_workflow.json
   nohup bash entrypoint.bsh -r 0 &

* Save the following output files: 

+ sysUsage.txt
+ cpuUsage.txt
+ memUsage.txt
+ nohup.out

* clears the old stuff
.. code::
   bash entrypoint.bsh -c
   rm nohup.out

* Run these tests upto 40

Benchmarking Results

Stereotop:



Scanner3DTop:
**TODO decide if and how we are attempting to benchmark this one. 

