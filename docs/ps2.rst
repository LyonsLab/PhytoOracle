Instruction manual for PSII Fluorescence HPC workflow
-----------------------------------------------------

**Staging Data on HPC xdisks**

- Login to UA-HPC OCELOTE and create expendable storage disk as below, 

.. code::

    xdisk -c create -h user_name -m 500
    xdisk -c query 

- Check and size the partion as needed using the `xdisk -c size -m 1000` command. 

- Download test data (tarball) using irods (Note: irods is not avaialble on elgato)

.. code::

   iinit # Enter your iRODS credentials
   cd /xdisk/user_name/
   iget -K /iplant/home/shared/terraref/ua-mac/raw_tars/season_10/ps2Top/ps2Top-2020-01-25.tar

- Logout of OCELOTE and login to ELGATO and change directories to `xdisk/user_name/` for execution.

- Git clone PhytoOracle PSII workflow

.. code::

   git clone https://github.com/uacic/PhytoOracle.git
   cd PhytoOracle/psII/
   mv /xdisk/user_name/ps2Top-2020-01-25.tar PhytoOracle/psII/

- Alter the `arg_example.json` and `submit_elagato_ps2.pbs` as needed as submit job as follows

.. code::

   qsub submit_elgato_ps2.pbs
   qstat -u user_name


**Resource Usage Log**

- The workflow comes with CCTools `Resource Monitor <https://cctools.readthedocs.io/en/latest/resource_monitor/>`_ enabled which, is a tool to monitor the computational resources used by the process created by the workflow and all its descendants.
- The resource monitor log outputs the following information though note that the monitor works indirectly, that is, by observing how the environment changed while a process was running, therefore all the information reported should be considered just as an estimate.

.. list-table:: Output Format
   :widths: 25 25
   :header-rows: 1
   
   * - Field
     - Description
   * - command
     - the command line given as an argument
   * - start
     - time at start of execution,since the epoch
   * - end
     - time at end of execution,since the epoch
   * - exit_type
     - one of normal ignal or limit (a string)
   * - signal
     - number of the signal that terminated the process
   * - cores
     - maximum number of cores used
   * - cores_avg
     - number of cores as cpu_time/wall_time
   * - exit_status
     - final status of the parent process
   * - max_concurrent_processes
     - the maximum number of processes running concurrently
   * - total_processes
     - count of all of the processes created
   * - wall_time
     - duration of execution, end - start
   * - cpu_time
     - user+system time of the execution 
   * - virtual_memory
     - maximum virtual memory across all processes
   * - memory
     - maximum resident size across all processes
   * - swap_memory
     - maximum swap usage across all processes 
   * - bytes_read
     - amount of data read from disk
   * - bytes_written
     - amount of data written to disk
   * - bytes_received
     - amount of data read from network interfaces
   * - bytes_sent
     - amount of data written to network interfaces
   * - bandwidth
     - maximum bandwidth used
   * - total_files
     - total maximum number of files and directories of all the working directories in the tree
   * - disk
     - size of all working directories in the tree
   * - limits_exceeded
     - resources over the limit with -l, -L options (JSON object)
   * - peak_times
     - seconds from start when a maximum occured (JSON object)
   * - snapshots
     -  List of intermediate measurements, identified by snapshot_name (JSON object)


Season-10 PSII workflow benchmarking
------------------------------------

- Click `here <https://docs.google.com/spreadsheets/d/1aoz9bixQsi4DKEDw8TkUyXmd6FJ5BdynTXdzsHuvAXw/edit?usp=sharing>`_ for ps2 workflow benchmarking on UA HPC. 
