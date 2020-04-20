<?php
  error_reporting(E_ALL);

  $filename = "bundle_list.json";
  $handle = fopen($filename, "r");
  $contents = fread($handle, filesize($filename));

  fclose($handle);

  if(empty(json_decode($contents, true)))
  {
    print("fail to parse json");
    exit();
  }
  $json = json_decode($contents, true);
  $bundle_list = $json["BUNDLE_LIST"];

  $CLEANED_META_DIR = "cleanmetadata_out/";
  $TIFS_DIR = "bin2tif_out/";
  $SOILMASK_DIR = "soil_mask_out/";
  $FIELDMOSAIC_DIR = "fieldmosaic_out/";
  $CANOPYCOVER_DIR = "canopy_cover_out/";

  $MOSAIC_LIST_FILE = $FIELDMOSAIC_DIR . "filelist.txt";
  $SENSOR = "stereoTop";
  $MOSAIC_BOUNDS = "-111.9750963 33.0764953 -111.9747967 33.074485715";

  $DATA_BASE_URL = "http://vm142-80.cyverse.org/";

?>
{
  "define": {
  },
  "rules": [

    <?php foreach ($bundle_list as &$bundle) :?>
    {
      # processing for one bundle of data sets (from cleanmetadata to soilmask)
      "command": "echo ${BUNDLE_JSON} && python3 process_bundle.py ${BUNDLE_JSON}",
      "environment": {
        "DATA_BASE_URL": "<?=$DATA_BASE_URL?>",
        "BUNDLE_JSON": "bundle/bundle_" + "<?=$bundle["ID"]?>" + ".json"
      },
      "inputs": [
        "process_bundle.py",
        "process_one_set.sh",
        "bundle/bundle_" + "<?=$bundle["ID"]?>" + ".json",
        "cached_betydb/bety_experiments.json"
      ],
      "outputs":
      [
      <?php foreach ($bundle["DATA_SETS"] as &$data_set): ?>
        "<?=$CLEANED_META_DIR?>" + "<?=$data_set["UUID"]?>" + "_metadata_cleaned.json",
        "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_mask.tif",
        "<?=$SOILMASK_DIR?>" + "<?=$data_set["UUID"]?>" + "_right_mask.tif",
      <?php endforeach; ?>
      ]
    },
    <?php endforeach; ?>
  ]
}
