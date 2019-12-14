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
If you need help running the pipeline, please check out our Plug-Play tutorial:

https://github.com/emmanuelgonz/PhytoOracle/blob/master/docs/plug-play.rst 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Swapping extractors
-------------------
PhytoOracle makes it easy to swap between extractors. To being swapping, edit the Makeflow file as follows:

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

3. Now you can run it locally!

.. code-block:: RST
    
    makeflow --jx define-hello.jx
    
    
Understand Jx language
Rule > command > inputs/outputs explicitly stated 
Swap with your Docker container
Variables in Makeflow file..
