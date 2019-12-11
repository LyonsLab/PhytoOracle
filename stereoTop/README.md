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
echo "export PATH=$PATH:/usr/local/go/bin" | sudo tee -a /etc/profile
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
tar -xvf 2018-5-15_5sets.tar
```

> Note: you can also get the data via other methods, as along as the data is in this directory (`starTerra/stereoTop`), and follows the same folder structure.

* To generate the list of input raw data files `raw_data_files.jx` from an local path
```bash
python gen_files_list.py 2018-05-15_5 >  raw_data_files.json
```

* To generate json workflow, (`main_wf.php` will read `raw_data_files.json`)

Install php runtime
```bash
sudo apt-get install php-cli
```
Generate workflow
```bash
php main_wf.php > main_wf.jx
jx2json main_wf.jx > main_workflow.json
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

