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

  $BETYDB_URL = "http://128.196.65.186:8000/bety/";
  $BETYDB_KEY = "wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD";
  $CLEANED_META_DIR = "cleanmetadata_out/";
  $TIFS_DIR = "bin2tif_out/";
  $GPSCORRECT_DIR = "gpscorrect_out/";
  $SOILMASK_DIR = "soil_mask_out/";
  $FIELDMOSAIC_DIR = "fieldmosaic_out/";
  $CANOPYCOVER_DIR = "canopy_cover_out/";
  $PLOTCLIP_DIR = "plotclip_out/";

  $MOSAIC_LIST_FILE = $FIELDMOSAIC_DIR . "filelist.txt";
  $SENSOR = "stereoTop";
  $MOSAIC_BOUNDS = "-111.9747932 33.0764785 -111.9750545 33.0745238";

  $DATA_BASE_URL = "128.196.142.19/";

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
	"<?=$GPSCORRECT_DIR?>" + "<?=$data_set["UUID"]?>" + "_left_corrected.tif",
	"<?=$data_set["UUID"]?>" + "_plotclip.tar",
      <?php endforeach; ?>
      ]
    },
    <?php endforeach; ?>
  ]
}
