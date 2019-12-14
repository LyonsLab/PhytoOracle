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

.. code-block::
    
    $ cat output-from-define.txt 
    hello world!
Creating Multiple Jobs
~~~~~~~~~~~~~~~~~~~~~~
Workflows enable you to run analysis codes. Below is an example of how to string multiple jobs together:
1. Write your job and generate multiple instance of the job: 
.. code-block::
    {
        "rules": [
                    {
                        "command" : "python ./example.py --parameter + N + " > output." + N + ".txt",
                        "inputs"  : [ "example.py" ],
                        "outputs" : [ "output." + N + ".txt" ]
                    } for N in [1, 2, 3]
                 ]
    }


Understand Jx language
Rule > command > inputs/outputs explicitly stated 
Swap with your Docker container
Variables in Makeflow file..
