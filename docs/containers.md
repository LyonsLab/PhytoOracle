stereoRGB
---------

| Extractor | Description | Link to Container |
| --------- | ----------- | ----------------- |
| Clean Metadata | Cleans, adjusts, and otherwise prepares metadata from the gantry for further downstream use by other transformers | https://hub.docker.com/r/agpipeline/cleanmetadata  |
| Bin to Tif | Converts an RGB binary file from the gantry, along with with cleaned metadata, and into a geo referenced TIFF image file | https://hub.docker.com/r/agpipeline/bin2tif  |
| Soil Mask | Masks out soil from RGB images and creates a new image with the soil removed (converted to black) | https://hub.docker.com/r/agpipeline/soilmask  |
| Field Mosaic | Combines multiple geo-referenced images into a single geo-referenced mosaic image. Produces images with varying resolution | https://hub.docker.com/r/agpipeline/fieldmosaic  |
| Canopy Cover | Calculates plot-level canopy cover from geo referenced images and writes the results into CSV files compatible with BETYdb and Geostreams | https://hub.docker.com/r/agpipeline/canopycover  |

Scanner3D
---------

| Extractor | Description | Link to Container |
| --------- | ----------- | ----------------- |
| Clean Metadata | Cleans, adjusts, and otherwise prepares metadata from the gantry for further downstream use by other transformers | https://hub.docker.com/r/agpipeline/cleanmetadata  |
| Ply to Las | Converts PLY files to LAS files | https://hub.docker.com/r/agpipeline/ply2las |
| Plot Clip | Retrieves plot boundaries and clips RGB, IR, and LAS/LAZ files to the plot boundaries | https://hub.docker.com/r/agpipeline/plotclip |
| Canopy Height | calculates canopy height from LAS files and writes CSV file ready for BETYdb and/or Geostreams database ingestion | https://hub.docker.com/r/agpipeline/canopy_height |

PSII Fluorescence
-----------------

| Extractor | Description | Link to Container |
| --------- | ----------- | ----------------- |
| Flip:1.0 | Converts binary images to PNG; Segments Images and Calculates Fluorescence aggregates | https://github.com/cyverse/docker-builds/tree/master/flip/1.0 |
