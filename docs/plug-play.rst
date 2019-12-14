Plug-Play User Manual
=====================
*This manual is for users who will be downloading and using the pipeline as-is with the supported extractors.*

Setting up Makeflow
-------------------
As of cctools 7.0.21, the support for the sub-workflow feature in JX is unclear, you need to pull the current master branch from cctools github repo and compile from source.

If you are on atmosphere this image should have all the dependency to compile cctools from source:

https://atmo.cyverse.org/application/images/1762


Or install the dependency yourself (Ubuntu 18.04 LTS):

https://jxuzy.blogspot.com/2019/11/install-cctools-ubuntu-1804lts.html

.. note:: 
   
   You will also need docker runtime to run this workflow.

* Those commands will compile and install cctools to your home directory, `makeflow` will be at `$HOME/cctools/bin/makeflow`, which is the path that `entrypoint.sh` uses.

.. code::

   git clone git://github.com/cooperative-computing-lab/cctools.git cctools-github-src
   cd cctools-github-src
   ./configure --prefix $HOME/cctools
   make
   make install

* Download test data (tarball), and decompressed it

.. code::
   
   iinit # Enter your iRODS credentials
   cd starTerra/stereoTop
   iget -K /iplant/home/shared/iplantcollaborative/example_data/starTerra/2018-05-15_5sets.tar
   tar -xvf 2018-5-15_5sets.tar


.. note:: 

   You can also get the data via other methods, as along as the data is in this directory (`starTerra/stereoTop`), and follows the same folder structure.

* To generate the list of input raw data files `raw_data_files.jx` from an iRODS path

.. code::
   
   python gen_files_list.py 2018-05-15 >  raw_data_files.jx


* Run the workflow, `-r 0` for 0 retry attempts if failed

.. code::

   chmod 755 entrypoint.sh
   ./entrypoint.sh -r 0


* Clean up output and logs

.. code::
   
   ./entrypoint.sh -c
   rm -f makeflow.jx.args.*


