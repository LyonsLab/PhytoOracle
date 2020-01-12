
# PythoOracle-Scanner3DTop-Laser

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
### Data staging
```bash
iget -rK /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/small_test_set
iget -rK /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/small_test_set_metadata small_test_set/PNG
```

### Run workflow for phase 1 (`cleanmetadata`, `ply2las`, `plotclip`)

+ Create file list
```bash
python3 gen_files_list.py small_test_set/PNG LEVEL_0_PATH _metadata.json small_test_set/PLY LEVEL_1_PATH __Top-heading-west_0.ply > raw_data_files.json
```

+ Create bundle, `1` for 1 data set per bundle
```bash
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 1
```

+ Split `bundle_list.json` into each individual bundle file
```bash
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
```

+ Start makeflow
```bash
./entrypoint_p1.sh
```

+ Cleanup
```bash
./entrypoint_p1.sh -c
```

### Run workflow for phase 2 (`plotmerge`, `canopy_height`)

+ Untar all the tarball (from phase1, result of `plotclip`), the plotclip result will be untar into `plotclip_out` folder
```bash
ls *.tar | xargs -I {} tar -xvf {}
```

+ Generate plot list off the result from untar
```bash
python3 gen_plot_list.py plotclip_out/ cleanmetadata_out/ LAS_FILES .las > plot_list.json
```

+ Start makeflow
> currently, the parsing of the phase 2 jx file takes around 40 min
```bash
./entrypoint_p2.sh
```

+ Cleanup
```bash
./entrypoint_p2.sh -c
```
