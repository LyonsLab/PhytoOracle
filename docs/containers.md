stereoRGB
---------

| Extractor | Description | Link to Container |
| --------- | ----------- | ----------------- |
| Clean Metadata | Cleans, adjusts, and otherwise prepares metadata from the gantry for further downstream use by other transformers | https://hub.docker.com/r/agpipeline/cleanmetadata  |
| Bin to Tif | Converts an RGB binary file from the gantry, along with with cleaned metadata, and into a geo referenced TIFF image file | https://hub.docker.com/r/agpipeline/bin2tif  |
| Soil Mask | Masks out soil from RGB images and creates a new image with the soil removed (converted to black) | https://hub.docker.com/r/agpipeline/soilmask  |
| Field Mosaic | Combines multiple geo-referenced images into a single geo-referenced mosaic image. Produces images with varying resolution | https://hub.docker.com/r/agpipeline/fieldmosaic  |
| Canopy Cover | Calculates plot-level canopy cover from geo referenced images and writes the results into CSV files compatible with BETYdb and Geostreams | https://hub.docker.com/r/agpipeline/canopycover  |
