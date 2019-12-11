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
  $TIFS_DIR = "bin2tif_out/";
  $SOILMASK_DIR = "soil_mask_out/";
  $FIELDMOSAIC_DIR = "fieldmosaic_out/";

  $MOSAIC_LIST_FILE = $FIELDMOSAIC_DIR . "filelist.txt";

  /*
  "METADATA_CLEANED_LIST": [<?php foreach ($data_file_list as &$data_set) :?> "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json", <?php endforeach?>],
  "LEFT_SOILMASK_LIST": [<?php foreach ($data_file_list as &$data_set) :?> "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_mask.tif", <?php endforeach?>],
  "RIGHT_SOILMASK_LIST": [<?php foreach ($data_file_list as &$data_set) :?> "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_right_mask.tif", <?php endforeach?>],
  */

?>
{
  "define": {
  },
  "rules": [

    <?php foreach ($data_file_list as &$data_set) :?>
    {
      # Make a cleaned copy of the metadata
      "command": "mkdir -p ${WORKING_SPACE} && BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ BETYDB_URL=${BETYDB_URL} BETYDB_KEY=${BETYDB_KEY} singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:latest --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}",
      "environment": {
        "SENSOR": "stereoTop",
        "METADATA": "<?=$data_set["PATH"]?>" + "<?=$data_set["UUID"]?>" + "_metadata.json",
        "WORKING_SPACE": "<?=$CLEANED_META_DIR?>",
        "USERID": ""
      },
      "inputs": [ "<?=$data_set["PATH"]?>" + "<?=$data_set["UUID"]?>" + "_metadata.json", "cached_betydb/bety_experiments.json" ],
      "outputs": [
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json"
      ]
    },


    {
      # Convert LEFT bin/RGB image to TIFF format
      "command": "mkdir -p ${WORKING_SPACE} && singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_BIN}",
      "environment": {
        "LEFT_BIN": "<?=$data_set["PATH"]?>" + "<?=$data_set["UUID"]?>" + "_left.bin",
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$TIFS_DIR?>"
      },
      "inputs": [
        "<?=$data_set["PATH"]?>" + "<?=$data_set["UUID"]?>" + "_left.bin",
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json"
      ],
      "outputs": [
        "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_left.tif"
      ]
    },
    {
      # Convert RIGHT bin/RGB image to TIFF format
      "command": "mkdir -p ${WORKING_SPACE} && singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${RIGHT_BIN}",
      "environment": {
        "RIGHT_BIN": "<?=$data_set["PATH"]?>" + "<?=$data_set["UUID"]?>" + "_right.bin",
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$TIFS_DIR?>"
      },
      "inputs": [
        "<?=$data_set["PATH"]?>" + "<?=$data_set["UUID"]?>" + "_right.bin",
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json"
      ],
      "outputs": [
        "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_right.tif"
      ]
    },

    {
      # Generate soil mask from LEFT TIFF image
      "command": "mkdir -p ${WORKING_SPACE} && singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/soilmask:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_TIF}",
      "environment": {
        "LEFT_TIF": "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_left.tif",
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$SOILMASK_DIR?>"
      },
      "inputs": [
        "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_left.tif",
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json"
      ],
      "outputs": [
        "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_mask.tif"
      ]
    },
    {
      # Generate soil mask from RIGHT TIFF image
      "command": "mkdir -p ${WORKING_SPACE} && singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/soilmask:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${RIGHT_TIF}",
      "environment": {
        "RIGHT_TIF": "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_right.tif",
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$SOILMASK_DIR?>"
      },
      "inputs": [
        "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_right.tif",
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json"
      ],
      "outputs": [
        "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_right_mask.tif"
      ]
    },
    <?php endforeach; ?>




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
    }
  ]
}
