Instruction manual for StereoTopRGB workflow
--------------------------------------------

.. image:: ../pics/rgb_concept_map.png
    :width: 400

**Staging Data on Master Instance**

- Git Clone the PhytoOracle github repository.

.. code::

    git clone https://github.com/uacic/PhytoOracle
    cd PhytoOracle
    git checkout dev


- Download test data (tarball), and decompress it

.. code::

   iinit # Enter your iRODS credentials
   cd stereoTop
   iget -K /iplant/home/shared/iplantcollaborative/example_data/starTerra/2018-05-15_5sets.tar
   tar -xvf 2018-05-15_5sets.tar

.. note::

   you can also get the data via other methods, as along as the data is in this directory (`PhytoOracle/stereoTop`), and follows the same folder structure.

- Hosting data on a HTTP Server (Nginx)

.. code::

   sudo apt-get install nginx apache2-utils
   wget https://raw.githubusercontent.com/uacic/PhytoOracle/dev/phyto_oracle.conf
   sudo mv phyto_oracle.conf /etc/nginx/sites-available/phyto_oracle.conf
   sudo ln -s /etc/nginx/sites-available/phyto_oracle.conf /etc/nginx/sites-enabled/phyto_oracle.conf
   sudo rm /etc/nginx/sites-enabled/default
   sudo nginx -s reload

- Set username and password for the HTTP file server

.. code::

   sudo htpasswd -c /etc/apache2/.htpasswd YOUR_USERNAME # Set password

- In the file `/etc/nginx/sites-available/phyto_oracle.conf`, change the line (~line 21) to the destination path to where the data is to be decompressed, e.g. `/home/uacic/PhytoOracle/stereoTop`

.. code::

   root /scratch/www;


- Change permissions of the data to allow serving by the HTTP server

.. code::

   sudo chmod -R +r 2018-05-15/
   sudo chmod +x 2018-05-15/*

- Change URL inside `main_wf.php` (~line 30) to the IP address or URL of the Master VM instance with HTTP server

.. note::

    **URL needs to have slash at the end**

.. code::

   $DATA_BASE_URL = "http://vm142-80.cyverse.org/";

- Change username and password inside `process_one_set.sh` (~line 27) to the ones that you set above

.. code::

   HTTP_USER="YOUR_USERNAME"
   HTTP_PASSWORD="PhytoOracle"

**Generating workflow `json` on Master instance**

- Generate a list of the input raw-data files `raw_data_files.jx` from a local path as below

.. code::

   python3 gen_files_list.py 2018-05-15/ >  raw_data_files.json

- Generate a `json` workflow using the `main_wf.php` script. The `main_wf.php` scripts parses the `raw_data_files.json` file created above.

.. code::

   sudo apt-get install php-cli
   php main_wf_phase1.php > main_wf_phase1.jx
   jx2json main_wf_phase1.jx > main_workflow_phase1.json

**Run the workflow on Master**

+ Run the workflow using the following entrypoint bash script

.. code::

   chmod 755 entrypoint.sh
   ./entrypoint.sh

- At this point, the Master will broadcast jobs on a catalog server and wait for Workers to connect. **Note the IP ADDRESS of the VM and the PORT number on which makeflow is listening, mostly `9123`**. We will need it to tell the workers where to find our Master.


**Connecting Worker Factories to Master**

- Launch one or more large instances with CCTools and Singularity installed as instructed above.

- Connect a Worker Factory using the command as below

.. code::

   work_queue_factory -T local IP_ADDRESS 9123 -w 40 -W 44 --workers-per-cycle 10  -E "-b 20 --wall-time=3600" --cores=1      --memory=2000 --disk 10000 -dall -t 900

.. list-table::
   :widths: 20 20
   :header-rows: 1

   * - Argument
     - Description
   * - -T local
     - specifies the mode of execution for the factory
   * - -w
     - minimum number of workers 
   * - -W
     - maximum number of workers

- Once the workers are spawned from the factories,you will see message as below

.. code::

   connected to master

- Makeflow Monitor on your Master VM

.. code::

   makeflow_monitor main_wf_phase1.jx.makeflowlog 


- Work_Queue Status to see how many workers are currently connected to the Master

.. code::

   work_queue_status

- Makeflow Clean up output and logs

.. code::

   ./entrypoint.sh -c
   rm -f makeflow.jx.args.*


**Connect Workers from HPC**

- Here is a pbs script to connect worker factories from UArizona HPC. Modify the following to add the IP_ADDRESS of your Master VM.

.. code::

    #!/bin/bash
    #PBS -W group_list=
    #PBS -q windfall
    #PBS -l select=1:ncpus=16:mem=24gb
    #PBS -l place=pack:shared
    #PBS -l walltime=02:00:00
    #PBS -l cput=02:00:00
    module load unsupported
    module load ferng/glibc
    module load singularity
    export CCTOOLS_HOME=/home/u15/sateeshp/cctools
    export PATH=${CCTOOLS_HOME}/bin:$PATH
    cd /home/u15/sateeshp/
    /home/u15/sateeshp/cctools/bin/work_queue_factory -T local IP_ADDRESS 9123 -w 12 -W 16 --workers-per-cycle 10 --cores=1 -t 900


--------
