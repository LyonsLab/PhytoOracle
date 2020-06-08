PhytoOracle
-----------

PhytoOracle is a scalable, distributed workflow manager for analyzing highthroughput phenotyping data.  
It is designed to process data from the `UA Gantry <https://uanews.arizona.edu/story/world-s-largest-robotic-field-scanner-now-place,>`_, but can be adapted to work on data coming from other platforms.  
PhytoOracle uses a manager-worker framework for distributed computing and can run jobs on nearly all unix-like environments. 

**Supported Sensors**

.. list-table::
   :header-rows: 1

   * - Sensor
     - Data Description
   * - StereoTopRGB
     - Data from RGB images is used for measuring the plant's canopy cover
   * - FlirIr
     - Infrared images are used to extract information about mean temperature between plants
   * - PSII
     - Measures the photosynthetic activity
   * - Stereotop3D
     - Used to measure the plant's physical structure
   * - Hyperspectral
     - Collects and processes information from across the electromagnetic spectrum
