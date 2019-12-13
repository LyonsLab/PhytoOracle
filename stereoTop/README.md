# terra-ref-makeflow


* cctools on Atmosphere

If you are on atmosphere this image will have the recommanded cctools version (version 7.0.21)

https://atmo.cyverse.org/application/images/1762

* cctools on Jetstream on other

If you are on Jetstream or other platform, you can compile cctools from source

You can install the dependency for compile form source (Ubuntu 18.04 LTS) [here](https://jxuzy.blogspot.com/2019/11/install-cctools-ubuntu-1804lts.html):

Those commands will compile and install cctools (version 7.0.21) to /usr/bin, so that they are in the $PATH
```bash
wget http://ccl.cse.nd.edu/software/files/cctools-7.0.21-source.tar.gz
tar -xvf cctools-7.0.21-source.tar.gz
cd cctools-release-7.0.21
./configure --prefix /usr
make -j$(nproc)
sudo make install
```

* Install singularity 3.5.1 (recommand using this version)

Install dependency for singularity
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
Build singularity
```bash
wget https://github.com/sylabs/singularity/releases/download/v3.5.1/singularity-3.5.1.tar.gz
tar -xvf singularity-3.5.1.tar.gz
cd singularity
./mconfig && \
    make -C builddir && \
    sudo make -C builddir install
```

* Pull workflow from github repository, `php-template` branch

```bash
git clone https://github.com/uacic/starTerra.git
cd starTerra
git checkout php-template
```

* Download test data (tarball), and decompressed it
```bash
iinit # Enter your iRODS credentials
cd stereoTop
iget -K /iplant/home/shared/iplantcollaborative/example_data/starTerra/2018-05-15_5sets.tar
tar -xvf 2018-05-15_5sets.tar
```

> Note: you can also get the data via other methods, as along as the data is in this directory (`PhytoOracle/stereoTop`), and follows the same folder structure.

* Hosting data in a HTTP Server (Nginx)
```bash
sudo apt-get install nginx apache2-utils
wget https://raw.githubusercontent.com/uacic/PhytoOracle/dev/phyto_oracle.conf
sudo mv phyto_oracle.conf /etc/nginx/sites-available/phyto_oracle.conf
sudo ln -s /etc/nginx/sites-available/phyto_oracle.conf /etc/nginx/sites-enabled/phyto_oracle.conf
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -s reload
```

Set username and password for the HTTP file server
```bash
sudo htpasswd -c /etc/apache2/.htpasswd YOUR_USERNAME # Set password
```

In file `/etc/nginx/sites-available/phyto_oracle.conf`, change the line (~line 21) below to where the data is decompressed, e.g. `/home/uacic/PhytoOracle/stereoTop`
```
	root /scratch/www;
```

Change permission of the data to allow serving by the HTTP server
```bash
sudo chmod -R +r 2018-05-15/
sudo chmod +x 2018-05-15/*
```

Change URL inside `main_wf.php` (~line 30) to the IP address or URL of the VM instance with HTTP server
> URL needs to have slash at the end
```bash
  $DATA_BASE_URL = "http://vm142-80.cyverse.org/";
```

Change username and password inside `process_one_set.sh` (~line 27) to the ones that you set above
```bash
HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
```

* To generate the list of input raw data files `raw_data_files.jx` from an local path
```bash
python3 gen_files_list.py 2018-05-15/ >  raw_data_files.json
```

* To generate json workflow, (`main_wf.php` will read `raw_data_files.json`)

Install php runtime
```bash
sudo apt-get install php-cli
```
Generate workflow
```bash
php main_wf_phase1.php > main_wf_phase1.jx
jx2json main_wf_phase1.jx > main_workflow_phase1.json
```

* Run the workflow

`-r 0` for 0 retry attempts if failed, it is for testing purpose only
```bash
chmod 755 entrypoint.sh
./entrypoint.sh -r 0
```

* Clean up output and logs
```bash
./entrypoint.sh -c
rm -f makeflow.jx.args.*
```

* Makeflow Monitor 
```bash
makeflow_monitor main_workflow.jx.makeflowlog 
makeflow_monitor sub_workflow.jx.makeflowlog 
```

