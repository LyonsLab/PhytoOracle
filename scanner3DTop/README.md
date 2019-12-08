
## Scanner3DTop module workflow

* Dependency / Prequsite
Atmosphere Image version 1.1
run `sudo /opt/cctools-get-master` to get the current master branch of cctools

* Data staging
```bash
iget -rK /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/small_test_set
iget -rK /iplant/home/elyons/ACIC/2019-final-gantry-data/scanner3DTop/small_test_set_metadata small_test_set/PNG
```

* Genereate file list
```bash
python gen_files_list.py small_test_set/PNG > raw_data_files.jx
```

* Run workflow

`-r 0` is for 0 retry attempt when job fails. It is for testing purpose only

```bash
chmod 755 entrypoint.sh
./entrypoint.sh -r 0
```

* Cleanup
```bash
sudo ./entrypoint.sh -c
```
