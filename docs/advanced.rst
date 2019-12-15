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
1. Begin by stating the rules, including the command and the inputs/outputs of your extractor. 

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
   
+ Continuing from the example in Step 1:

.. code-block:: RST 

    { 
    "define":{
                "message" : "hello world!"
             },
    "rules": [
                {
                    "command": "/bin/echo " +message+ " > output-from-define.txt",
                    "outputs": [ "output-from-define.txt" ],
                    "inputs":  [ ],
                }
             ]
    }

Running your workflow 
~~~~~~~~~~~~~~~~~~~~~
3. Now you can run it locally!

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
