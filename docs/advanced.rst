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
PhytoOracle makes it easy to swap between extractors. To begin swapping, edit the Makeflow file as follows:

Editing our workflow 
~~~~~~~~~~~~~~~~~~~~
1. Begin by stating the rules, including the command and the inputs/outputs of your extractor. 

.. code-block:: RST
   
     {
      "rules": [
                  {
                      "command" : # What will be used as input for the terminal,
                      "inputs"  : [# Files and/or directories that will carry the metadata ]
                      "outputs" : [# Files and/or directories which are specified in the "define" section ],
                      
                  }
              ]
    }
|

Defining your values
~~~~~~~~~~~~~~~~~~~~
2. Define elements of your workflow:

.. code-block:: RST

For this tutorial, we will be looking at parts of a .jx file that will be used to run the workflows:

   {
       "define": {
           # The dataset URL, the key for it, directories, and file variables will all be defined in this portion of the file.
       },
       "rules": [
           # The rules we have multiple sections, one or two for each extractor, as well as environments so the terminal and Docker can              communicate with one another.
       ]
   }
   
+ Continuing from the example in Step 1, this code is a brief look at what is to come when mastering the makeflow. For this example, we   will be looking into the LIDAR makeflow:

.. code-block:: RST 

    {
    "define": {
      "BETYDB_URL": "https://terraref.ncsa.illinois.edu/bety/",
      "BETYDB_KEY": "9999999999999999999999999999999999999999",

      # Pass down by the main_workflow
      "LEVEL_0_DATA_PATH": "small_test_set/PNG/2017-06-21__00-00-26-364/",
      "LEVEL_1_DATA_PATH": "small_test_set/PLY/2017-06-21__00-00-26-364/",
      "UUID": "b5246694-65d8-44b9-a99c-3d010c92ec64",

      "CLEANED_META_DIR": "cleanmetadata_out/",
      "LAS_DIR": "las_out/",
      "PLOTCLIP_DIR": "plotclip_out/",
      "CANOPY_HEIGHT_DIR": "canopy_height_out",

      "METADATA": LEVEL_0_DATA_PATH + UUID + "_metadata.json",
      "METADATA_CLEANED": CLEANED_META_DIR + UUID + "_metadata_cleaned.json",
      "EAST_PLY": LEVEL_1_DATA_PATH + UUID + "__Top-heading-east_0.ply",
      "WEST_PLY": LEVEL_1_DATA_PATH + UUID + "__Top-heading-west_0.ply",
      "EAST_LAS": LAS_DIR + UUID + "__Top-heading-east_0.las",
      "WEST_LAS": LAS_DIR + UUID + "__Top-heading-west_0.las",
      "WEST_MERGED_LAS": UUID + "__Top-heading-west_0_merged.las",
      "WEST_MERGED_CONTENT_TXT": UUID + "__Top-heading-west_0_merged_contents.txt",

      # per Plot variable
      "PLOT_DIR": PLOTCLIP_DIR + "MAC Field Scanner Season 4 Range 21 Column 1/",
      "PLOT_NAME": "MAC Field Scanner Season 4 Range 21 Column 1"
  },
  # Here is an example of what the 'rules' portion of the makeflow files will look like. Each rule section will have their own 
  # 'commands' as input for the terminal on the machine, the 'environment' should be seen as a link between communication for docker and
  # the terminal, as well as inputs and outputs which will consist of files and directories. Below is what the extractor cleanmetadata
  # would look like in a .jx file. 
  "rules": [
    {
      "command": "mkdir ${CLEANED_META_DIR}",
      "environment": {
        "CLEANED_META_DIR": CLEANED_META_DIR
      },
      "outputs": [ CLEANED_META_DIR ]
    },

    {
      # cleanmetadata
      "command":"BETYDB_URL=${BETYDB_URL} BETYDB_KEY=${BETYDB_KEY} singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}",
      "environment": {
        "BETYDB_URL": BETYDB_URL,
        "BETYDB_KEY": BETYDB_KEY,
        "METADATA": METADATA,
        "WORKING_SPACE": CLEANED_META_DIR,
        "SENSOR": "scanner3DTop",
        "USERID": ""
      },
      "inputs": [ CLEANED_META_DIR, METADATA ],
      "outputs": [ METADATA_CLEANED ]
    }
]

Running your workflow 
~~~~~~~~~~~~~~~~~~~~~
3. Now you can run it locally!

Here is a brief sub-tutorial on running a makeflow file:

.. code-block:: RST
    
    $ makeflow --jx define-hello.jx
    
    parsing define-hello.jx...
    local resources: 4 cores, 7764 MB memory, 2097151 MB disk
    max running local jobs: 4
    checking define-hello.jx for consistency...
    define-hello.jx has 1 rules.
    starting workflow....
    submitting job: /bin/echo hello world! > output-from-define.txt
    submitted job 1376
    job 1376 completed
    
+ Then run the following: 

.. code-block:: RST 
    
    $ cat output-from-define.txt 
    hello world!

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
