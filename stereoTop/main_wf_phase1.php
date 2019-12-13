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
  $CANOPYCOVER_DIR = "canopy_cover_out/";

  $MOSAIC_LIST_FILE = $FIELDMOSAIC_DIR . "filelist.txt";
  $SENSOR = "stereoTop";
  $MOSAIC_BOUNDS = "-111.9750963 33.0764953 -111.9747967 33.074485715";
  /*
  "METADATA_CLEANED_LIST": [<?php foreach ($data_file_list as &$data_set) :?> "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json", <?php endforeach?>],
  "LEFT_SOILMASK_LIST": [<?php foreach ($data_file_list as &$data_set) :?> "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_mask.tif", <?php endforeach?>],
  "RIGHT_SOILMASK_LIST": [<?php foreach ($data_file_list as &$data_set) :?> "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_right_mask.tif", <?php endforeach?>],
  */
  $DATA_BASE_URL = "http://vm142-80.cyverse.org/";

?>
{
  "define": {
  },
  "rules": [

    <?php foreach ($data_file_list as &$data_set) :?>
    {
      # processing for a single set of data (from cleanmetadata to soilmask)
      "command": "./process_one_set.sh",
      "environment": {
        "DATA_BASE_URL": "<?=$DATA_BASE_URL?>",
        "RAW_DATA_PATH": "<?=$data_set["PATH"]?>",
        "UUID": "<?=$data_set["UUID"]?>"
      },
      "inputs": [
        "process_one_set.sh",
        "cached_betydb/bety_experiments.json"
      ],
      "outputs": [
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json",
        "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_mask.tif",
        "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_right_mask.tif"
      ]
    },
    <?php endforeach; ?>
  ]
}
