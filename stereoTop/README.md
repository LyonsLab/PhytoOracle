# terra-ref-makeflow

As of cctools 7.0.21, the support for the sub-workflow feature in JX is unclear, you need to pull the current master branch from cctools github repo and compile from source.

If you are on atmosphere this image should have all the dependency to compile cctools from source

https://atmo.cyverse.org/application/images/1762

Or install the dependency yourself (Ubuntu 18.04 LTS)

https://jxuzy.blogspot.com/2019/11/install-cctools-ubuntu-1804lts.html

> Note: you will also need docker runtime to run this workflow

* Those commands will compile and install cctools to your home directory, `makeflow` will be at `$HOME/cctools/bin/makeflow`, which is the path that `entrypoint.sh` uses.
```bash
git clone git://github.com/cooperative-computing-lab/cctools.git cctools-github-src
cd cctools-github-src
./configure --prefix $HOME/cctools
make
make install
```

* Run the workflow, `-r 0` for 0 retry attempts if failed
```bash
cd terra-ref-makeflow
chmod 755 entrypoint.sh
./entrypoint.sh -r 0
```

* Clean up output and logs
```bash
./entrypoint.sh -c
rm -f makeflow.jx.args.*
```
* Modify `main_env.jx` to run on other data sets

Just append/change the full iRODS path in the `IRODS_DIR_PATH_LIST` array (path needs to be in double quotes)

And append/change the UUID (part of the filename, e.g `5716a146-8d3d-4d80-99b9-6cbf95cfedfb_left.bin` has a UUID of `5716a146-8d3d-4d80-99b9-6cbf95cfedfb`) in the `UUID_LIST` array

`NUM_SET` is the number of data sets, basically the number of elements in the `IRODS_DIR_PATH_LIST` array

> Note: the elements in the `IRODS_DIR_PATH_LIST` array and `UUID_LIST` array needs to be 1-to-1 corresponded, meaning data file with `UUID` of 0th elements in `UUID_LIST` are stored at the path represented by the 0th element in the `IRODS_DIR_PATH_LIST` array
