Advanced User Manual
====================
.. Contents::

Who is this tutorial for?
-------------------------
This tutorial is meant for users who will be swapping between available extractors or integrating new ones. It will provide in-depth instructions into changing the workflow files and ....

You should be comfortable using:
  - Jx language
  - CCTools.
  - Docker containers.
  - iRods.
  - HPC.
  - More.

Why use PhytoOracle?
--------------------
PhytoOracle is a scalable, modular data pipeline that reduces processing times. If you are looking for a pipeline that provides the flexibility to add new extractors or develop some, this pipeline is for you.. 

Getting started
---------------
If you need help running the pipeline, please check out our `Plug-Play <https://github.com/emmanuelgonz/PhytoOracle/blob/master/docs/plug-play.rst>`_ tutorial.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Swapping extractors
-------------------
PhytoOracle makes it easy to swap between extractors. To being swapping, edit the Makeflow file as follows:

Editing our workflow 
~~~~~~~~~~~~~~~~~~~~
1. The makeflow jx needs rules, including the command and the inputs/outputs of your extractor under the "rules" section.

.. code-block:: RST
   
     {
      "rules": [
                  {
                      "command" : "/bin/echo hello world > output.txt",
                      "outputs" : [ "output.txt" ],
                      "inputs"  : [ ]
                  }
              ]
    }
|

The first part of phytoOracle uses a makeflow file to run a shell script to access singularity containers. 

Looking at the stereoTop makeflow jx file main_workflow_phase1.jx for example: 

.. code-block:: python
  "rules": [
    {
      # processing for a single set of data (from cleanmetadata to soilmask)
      "command": "./process_one_set.sh",
      "environment": {
        "RAW_DATA_PATH": DATA_SET["PATH"],
        "UUID": DATA_SET["UUID"]
      },
      
The command section contains an execution of the process_one_set.sh script.

Lets take a look at execution within the process_one_set.sh:

.. code-block:: BASH
    singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_BIN}
    
This is where the extractors are run as singularity containers. Makeflow will run the script as many times as you tell it to. If you run multiple containers per script, then they should all run every time the command within the "rules" is set to run. Either once per "command" or once for every variable in a certain range in a loop 

Remember to include any input files the extractor will need

.. code-bock:: python 

      "inputs": [
        "process_one_set.sh",
        DATA_SET["PATH"] + DATA_SET["UUID"] + "_metadata.json",
        "cached_betydb/bety_experiments.json",
        DATA_SET["PATH"] + DATA_SET["UUID"] + "_left.bin",
        DATA_SET["PATH"] + DATA_SET["UUID"] + "_right.bin"
      ],
      "outputs": [
        CLEANED_META_DIR + DATA_SET["UUID"] + "_metadata_cleaned.json",
        SOILMASK_DIR + DATA_SET["UUID"] + "_left_mask.tif",
        SOILMASK_DIR + DATA_SET["UUID"] + "_right_mask.tif"
      ]
    } for DATA_SET in DATA_FILE_LIST,  ### This will loop through each data file in the list
    
Makeflow also lets you create, move or delete files or directories as you run the extractors and generate data.

.. code-block:: BASH

      "rules": [
    {
      # Make directory to store FIELDMOSAIC files
      "command": "mkdir -p ${FIELDMOSAIC_DIR}",
      
Defining your values
~~~~~~~~~~~~~~~~~~~~
2. Define elements of your workflow:

.. code-block:: RST

   {
       "define": {
           # symbol definitions go here
       },
       "rules": [
           # Rules you created above go here
       ]
   }

Defining rules rules allows you to specify directory names or an RANGEs in loops

.. code-block:: python

  "define": {
    "CLEANED_META_DIR": "cleanmetadata_out/",
    "SOILMASK_DIR": "soil_mask_out/",
    "FIELDMOSAIC_DIR": "fieldmosaic_out/",
    "MOSAIC_BOUNDS": "-111.9750963 33.0764953 -111.9747967 33.074485715",
    "CANOPYCOVER_DIR": "canopy_cover_out/"
  },
  
This is usefull for specifying ranges such as the DATA_FILE_LIST used above.



Running your workflow 
~~~~~~~~~~~~~~~~~~~~~
3. Normally, to run a makeflow file locally you will need to include --jx when running makeflow to specify that the file is in JX format. 

.. code-block:: RST
    
    $ makeflow --jx hello-makeflow.jx
    
    parsing define-hello.jx...
    local resources: 4 cores, 7764 MB memory, 2097151 MB disk
    max running local jobs: 4
    checking define-hello.jx for consistency...
    define-hello.jx has 1 rules.
    starting workflow....
    submitting job: /bin/echo hello world! > output-from-define.txt
    submitted job 1376
    job 1376 completed
    

Lets see what happend when we ran the entrypoint.sh script: 

.. code-block:: BASH

    makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log $@

In this case we used --json to run the main_workflow_phase1.json, -T wq to broadcast the job to workqueue, -a to include it in the catalog server under the name phyto_oracle-atmo


The "command" section was received by any machine that was conneced using:

.. code-block:: BASH

    work_queue_factory -T local IP_ADDRESS 9123 -w 40 -W 44 --workers-per-cycle 10  -E "-b 20 --wall-time=3600" --cores=1 --memory=2000 --disk 10000 -dall -t 900

That command was the process_one_set.sh script which ran those singularity containers containing the extractors. 

Creating Multiple Jobs
~~~~~~~~~~~~~~~~~~~~~~
Workflows enable you to run analysis codes. Below is an example of how to string multiple jobs together:

1. Write your job and generate multiple instance of the job

.. code-block:: RST

    {
        "rules": [
                    {
                        "command" : "python ./example.py --parameter + N + " > output." + N + ".txt",
                        "inputs"  : [ "example.py" ],
                        "outputs" : [ "output." + N + ".txt" ]
                    } for N in [1, 2, 3]
                 ]
    }

2. Stitch Results

.. code-block:: RST

    {
        "command" : "/bin/cat + join(["output.1.txt","output.2.txt","output.3.txt"], " ") + " > output.all.txt",
        "inputs"  : [ "output." + N + ".txt" ] for N in [1,2,3] ],
        "outputs" : [ "output.all.txt" ]
    }
    
+ Or you could factor out the definition of the list and the range to the define section of the workflow as follows: 

.. code-block:: RST
    {
        "define" : {
            "RANGE"    : range(1,4),
            "FILELIST" : [ "output." + N + ".txt" for N in RANGE ],
        },

        "rules" : [
                    {
                        "command" : "python ./simulate.py --parameter " + N + " > output."+N+".txt",
                        "inputs"  : [ "simulate.py" ],
                        "outputs" : [ "output." + N + ".txt" ]
                    } for N in RANGE,
                    {
                        "command" : "/bin/cat " + join(FILELIST," ") + " > output.all.txt",
                        "inputs"  : FILELIST,
                        "outputs" : [ "output.all.txt" ]
                    }
                  ]
    }


Understand Jx language
Rule > command > inputs/outputs explicitly stated 
Swap with your Docker container
Variables in Makeflow file..
