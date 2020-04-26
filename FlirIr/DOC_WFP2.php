<?php
  error_reporting(E_ALL);

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
  $TIFS_DIR = "flir2tif_out/";
  $FIELDMOSAIC_DIR = "fieldmosaic_out/";
  $MTEMP_DIR = "meantemp_out/"; 
  
  $MOSAIC_LIST_FILE = $FIELDMOSAIC_DIR . "filelist.txt";
  $SENSOR = "flirIrCamera";
  $MOSAIC_BOUNDS = "-111.9747932 33.0764785 -111.9750545 33.0745238";

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
      # Create MOSAIC_LIST_FILE file from the TIFS files
      "command": "ls ${TIFS_DIR}*.tif > ${MOSAIC_LIST_FILE}",
      "environment": {
        "TIFS_DIR": "<?=$TIFS_DIR?>",
        "MOSAIC_LIST_FILE": "<?=$MOSAIC_LIST_FILE?>"
      },
      "inputs": [
        "<?=$FIELDMOSAIC_DIR?>"
      ]
      + [<?php foreach ($data_file_list as &$data_set) :?> "<?=$TIFS_DIR?>" + "<?=$data_set["UUID"]?>" + "_ir.tif", <?php endforeach?>],
      "outputs": [
        "<?=$MOSAIC_LIST_FILE?>"
      ]
    },


    {
      # Generate field mosaic from ir TIFFs
      "command": "sudo docker run --rm --mount src=`pwd`,target=/mnt,type=bind agpipeline/fieldmosaic:2.0 -d --working_space /mnt${WORKING_SPACE} --metadata /mnt/${METADATA} /mnt/${MOSAIC_LIST_FILE} ${SENSOR} \"${MOSAIC_BOUNDS}\""

      "environment": {
        "MOSAIC_LIST_FILE": "<?=$MOSAIC_LIST_FILE?>",
        "METADATA": "<?=$CLEANED_META_DIR?>" + "<?=$data_file_list[0]["UUID"]?>" + "_metadata_cleaned.json",
        "WORKING_SPACE": "<?=$FIELDMOSAIC_DIR?>",
        "SENSOR": "flirIrCamera",
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
    }
  ]
}
