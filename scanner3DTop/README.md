
# PhytoOracle-Scanner3DTop-Laser

## System Setup

See [this](https://github.com/uacic/PhytoOracle/blob/master/docs/setup.rst) page for links to Atmosphere images and manual installation instructions.

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
cd scanner3DTop
iget -rK /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/small_test_set
iget -rK /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/small_test_set_metadata small_test_set/PNG
```

> **Note: you can also get the data via other methods, as along as the data is in this directory (`PhytoOracle/scanner3DTop`), and follows the same folder structure.**

+ Hosting data on a HTTP Server (Nginx)

> Why host this server?
> Hosting data on an HTTP server is to bypass the connection limit of iRODs, and reduce network load on makeflow
> You can get the data to worker instance via other method as well.
```bash
sudo apt-get install nginx apache2-utils
wget https://raw.githubusercontent.com/uacic/PhytoOracle/dev/phyto_oracle.conf
sudo mv phyto_oracle.conf /etc/nginx/sites-available/phyto_oracle.conf
sudo ln -s /etc/nginx/sites-available/phyto_oracle.conf /etc/nginx/sites-enabled/phyto_oracle.conf
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -s reload
```

+ Set username and password for the HTTP file server
> Setting up password is to merely prevent random trespasser, not as a security measure,
> if security is a concern, additional actions need to be taken
```bash
sudo htpasswd -c /etc/nginx/.htpasswd YOUR_USERNAME # Set password
```

+ In the file `/etc/nginx/sites-available/phyto_oracle.conf`, change the line (~line 21) to the destination path to where the data is to be decompressed, e.g. `/home/uacic/PhytoOracle/scanner3DTop`
```
	root /scratch/www;
```

+ Change permissions of the data to allow serving by the HTTP server
```bash
sudo chmod -R +r small_test_set
sudo chmod +x small_test_set
sudo chmod +x small_test_set/*
sudo chmod +x small_test_set/PNG/*
sudo chmod +x small_test_set/PLY/*
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
