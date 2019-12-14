# PythoOracle-StereoTop-RGB


## System Requirements

+ PhytoOracle is designed for distributed scaling on Cloud platforms and High-Performance Computers. The minimum requirements being:
	+ One Master instance with the required data staged that will broadcast jobs
	+ One or more instances that will launch Worker_Factories that will connect to the Master

+ **Required Software**

+ [CCTools 7.0.21](http://ccl.cse.nd.edu/software/downloadfiles.php)
+ [Singularity]()
+ [iRODS Client]()

#### CyVerse Atmosphere Image

+ Click [here](https://atmo.cyverse.org/application/images/1764) for Atmosphere image that comes with recommended CCTools (7.0.21) and Singularity (7.0.21) version installed.

#### Manual Installation 

Here are instructions for installation on Jetsream and other clouds.

##### CCTools (7.0.21)

+ You can install the dependency for compile from source (Ubuntu 18.04 LTS) [here](https://jxuzy.blogspot.com/2019/11/install-cctools-ubuntu-1804lts.html):

+ These commands will compile and install cctools (version 7.0.21) to `/usr/bin`, so that they are in the `$PATH`.
```bash
wget http://ccl.cse.nd.edu/software/files/cctools-7.0.21-source.tar.gz
tar -xvf cctools-7.0.21-source.tar.gz
cd cctools-release-7.0.21
./configure --prefix /usr
make -j$(nproc)
sudo make install
```

##### Singularity 3.5.1 (recommended)

+ Install dependencies for singularity
```bash
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
```
+ Build singularity
```bash
wget https://github.com/sylabs/singularity/releases/download/v3.5.1/singularity-3.5.1.tar.gz
tar -xvf singularity-3.5.1.tar.gz
cd singularity
./mconfig && \
    make -C builddir && \
    sudo make -C builddir install
```

### Staging Data on Master Instance

+ Git Clone the PhytoOracle github repository.
```bash
git clone https://github.com/uacic/PhytoOracle
cd PhytoOracle
git checkout dev
```

+ Download test data (tarball), and decompress it
```bash
iinit # Enter your iRODS credentials
cd stereoTop
iget -K /iplant/home/shared/iplantcollaborative/example_data/starTerra/2018-05-15_5sets.tar
tar -xvf 2018-05-15_5sets.tar
```

> **Note: you can also get the data via other methods, as along as the data is in this directory (`PhytoOracle/stereoTop`), and follows the same folder structure.**

+ Hosting data on a HTTP Server (Nginx)

Why host this server? :
```bash
sudo apt-get install nginx apache2-utils
wget https://raw.githubusercontent.com/uacic/PhytoOracle/dev/phyto_oracle.conf
sudo mv phyto_oracle.conf /etc/nginx/sites-available/phyto_oracle.conf
sudo ln -s /etc/nginx/sites-available/phyto_oracle.conf /etc/nginx/sites-enabled/phyto_oracle.conf
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -s reload
```

+ Set username and password for the HTTP file server
```bash
sudo htpasswd -c /etc/apache2/.htpasswd YOUR_USERNAME # Set password
```

+ In the file `/etc/nginx/sites-available/phyto_oracle.conf`, change the line (~line 21) to the destination path to where the data is to be decompressed, e.g. `/home/uacic/PhytoOracle/stereoTop`
```
	root /scratch/www;
```

+ Change permissions of the data to allow serving by the HTTP server
```bash
sudo chmod -R +r 2018-05-15/
sudo chmod +x 2018-05-15/*
```

+ Change URL inside `main_wf.php` (~line 30) to the IP address or URL of the Master VM instance with HTTP server
> **URL needs to have slash at the end**

```bash
  $DATA_BASE_URL = "http://vm142-80.cyverse.org/";
```

+ Change username and password inside `process_one_set.sh` (~line 27) to the ones that you set above
```bash
HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
```

###### Generating workflow `json` on Master

+ Generate a list of the input raw-data files `raw_data_files.jx` from a local path as below
```bash
python3 gen_files_list.py 2018-05-15/ >  raw_data_files.json
```

+ Generate a `json` workflow using the `main_wf.php` script. The `main_wf.php` scripts parses the `raw_data_files.json` file created above.
```bash
sudo apt-get install php-cli
php main_wf_phase1.php > main_wf_phase1.jx
jx2json main_wf_phase1.jx > main_workflow_phase1.json
```

###### Run the workflow on Master

+ `-r 0` for 0 retry attempts if failed (**it is for testing purposes only**). 
```bash
chmod 755 entrypoint.sh
./entrypoint.sh -r 0
```

At this point, the Master will broadcast jobs on a catalog server and wait for Workers to connect. **Note the IP ADDRESS of the VM and the PORT number on which makeflow is listening, mostly `9123`**. We will need it to tell the workers where to find our Master.

##### Connecting Worker Factories to Master

+ Launch one or more large instances with CCTools and Singularity installed as instructed above.

+ Connect a Worker Factory using the command as below

```bash
work_queue_factory -T local IP_ADDRESS 9123 -w 40 -W 44 --workers-per-cycle 10  -E "-b 20 --wall-time=3600" --cores=1 --memory=2000 --disk 10000 -dall -t 900
```
|argument|description|
|--------|-----------|
| -T local | this species the mode of execution for the factory |
| -w | min number of workers |
| -W | max number of workers | 

Once the workers are spawned from the factories,you will see message as below
```
connected to master
```

+ Makeflow Monitor on your Master VM
```bash
makeflow_monitor main_wf_phase1.jx.makeflowlog 
```

+ Work_Queue Status to see how many workers are currently connected to the Master
```
work_queue_status
```

+ Makeflow Clean up output and logs
```bash
./entrypoint.sh -c
rm -f makeflow.jx.args.*
```

## Connect Workers from HPC

+ Here is a pbs script to connect worker factories from UArizona HPC. Modify the following to add the IP_ADDRESS of your Master VM.

```bash
#!/bin/bash
#PBS -W group_list=ericlyons
#PBS -q windfall
#PBS -l select=2:ncpus=6:mem=24gb
#PBS -l place=pack:shared
#PBS -l walltime=02:00:00
#PBS -l cput=02:00:00
module load unsupported
module load ferng/glibc
module load singularity
export CCTOOLS_HOME=/home/u15/sateeshp/cctools
export PATH=${CCTOOLS_HOME}/bin:$PATH
cd /home/u15/sateeshp/
/home/u15/sateeshp/cctools/bin/work_queue_factory -T local IP_ADDRESS 9123 -w 80 -W 200 --workers-per-cycle 10  -E "-b 20 --wall-time=3600" --cores=1 -t 900
```


