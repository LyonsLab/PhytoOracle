System Setup
------------

**System Requirements**

- PhytoOracle is designed for distributed scaling on Cloud platforms and High-Performance Computers. The minimum requirements being:
  - One Master instance with the required data staged that will broadcast jobs
  - One or more instances that will launch Worker_Factories that will connect to the Master

**Required Software**

+ `CCTools 7.0.21 <http://ccl.cse.nd.edu/software/downloadfiles.php>`_
+ `Singularity 3.5.1 <https://github.com/sylabs/singularity/releases/tag/v3.5.1>`_
+ `iRODS Client <https://github.com/cyverse/irods-icommands-installers>`_

**CyVerse Atmosphere Image**

- Click [here](https://atmo.cyverse.org/application/images/1764) for Atmosphere image that comes with recommended CCTools (7.0.21) and Singularity (7.0.21) version installed.

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

