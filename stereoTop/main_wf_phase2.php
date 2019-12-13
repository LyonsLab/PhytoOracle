<?php
  $filename = "raw_data_files.json";
  $handle = fopen($filename, "r");
  $contents = fread($handle, filesize($filename));

  fclose($handle);

  if(empty(json_decode($contents, true)))
  {
    print("fail to parse json");
    exit();
  }
  $json = json_decode($contents, true);
  $data_file_list = $json["DATA_FILE_LIST"];

  $CLEANED_META_DIR = "cleanmetadata_out/";
  $SOILMASK_DIR = "soil_mask_out/";
  $FIELDMOSAIC_DIR = "fieldmosaic_out/";
  $CANOPYCOVER_DIR = "canopy_cover_out/";

  $MOSAIC_LIST_FILE = $FIELDMOSAIC_DIR . "filelist.txt";
  $SENSOR = "stereoTop";
  $MOSAIC_BOUNDS = "-111.9750963 33.0764953 -111.9747967 33.074485715";

?>
{
  "define": {
  },
  "rules": [

    {
      # Make directory to store FIELDMOSAIC files
      "command": "mkdir -p ${FIELDMOSAIC_DIR}",
      "environment": {
        "FIELDMOSAIC_DIR": "<?=$FIELDMOSAIC_DIR?>"
      },
      "inputs": [],
      "outputs": [
        "<?=$FIELDMOSAIC_DIR?>"
      ]
    },
    {
      # Create MOSAIC_LIST_FILE file from the soilmask files
      "command": "ls ${SOILMASK_DIR}*.tif > ${MOSAIC_LIST_FILE}",
      "environment": {
        "SOILMASK_DIR": "<?=$SOILMASK_DIR?>",
        "MOSAIC_LIST_FILE": "<?=$MOSAIC_LIST_FILE?>"
      },
      "inputs": [
        "<?=$FIELDMOSAIC_DIR?>"
      ]
      + [<?php foreach ($data_file_list as &$data_set) :?> "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_mask.tif", <?php endforeach?>]
      + [<?php foreach ($data_file_list as &$data_set) :?> "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_right_mask.tif", <?php endforeach?>],
      "outputs": [
        "<?=$MOSAIC_LIST_FILE?>"
      ]
    },


    {
      # Generate field mosaic from soil mask TIFFs
      "command": "singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/fieldmosaic:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${MOSAIC_LIST_FILE} ${SENSOR} \"${MOSAIC_BOUNDS}\"",
      "environment": {
        "MOSAIC_LIST_FILE": "<?=$MOSAIC_LIST_FILE?>",
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_file_list[0]["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$FIELDMOSAIC_DIR?>",
        "SENSOR": "stereoTop",
        "MOSAIC_BOUNDS": "<?=$MOSAIC_BOUNDS?>"
      },
      "inputs": [
        "<?=$FIELDMOSAIC_DIR?>",
        "<?=$MOSAIC_LIST_FILE?>",
        "<?=$CLEANED_META_DIR?>" + "<?=$data_file_list[0]["UUID"]?>" + "_metadata_cleaned.json"
      ],
      "outputs": [
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic.png",
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic.png.aux.xml",
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic.tif",
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic.vrt",
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic_10pct.tif",
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic_thumb.tif"
      ]
    },
    {
      # Generate canopy cover from field mosaic
      "command": "mkdir -p ${WORKING_SPACE} && singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/canopycover:latest --debug --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${CANOPY_COVER_INPUT_FILE}",
      "environment": {
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_file_list[0]["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$CANOPYCOVER_DIR?>",
        "CANOPY_COVER_INPUT_FILE": "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic.tif"
      },
      "inputs": [
        "<?=$CLEANED_META_DIR?>" + "<?=$data_file_list[0]["UUID"]?>" + "_metadata_cleaned.json",
        "<?=$FIELDMOSAIC_DIR?>" + "fullfield_mosaic.tif"
      ],
      "outputs": [
        "<?=$CANOPYCOVER_DIR?>" + "canopycover.csv",
        "<?=$CANOPYCOVER_DIR?>" + "canopycover_geostreams.csv"
      ]
    }
  ]
}
