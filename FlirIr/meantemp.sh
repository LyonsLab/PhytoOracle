#!/bin/bash
  
# Define arguments
homedir=$(pwd)
TITLE='title'
AUTHOR='author'
YEAR='year'
METADATA=$(ls cleanmetadata_out/ | head -n 1)

# Crate working directory
mkdir -p meantemp_out
chmod 755 meantemp_out

# Create list of files that need to be processed
ls flir2tif_out > t0.txt
sed 's/^/flir2tif_out\//' t0.txt > meanlist.txt

# Run meantemp transformer
BETYDB_URL=http://128.196.65.186:8000/bety/ BETYDB_KEY=YUKPF38ZxMB0UOkP6etB9bNOjTjIeWFj0RbNGIg5 singularity run -B $homedir:/mnt --pwd /mnt docker://agpipeline/meantemp:3.0 --result print --working_space meantemp_out/ --metadata cleanmetadata_out/$METADATA --citation_author $AUTHOR --citation_title $TITLE --citation_year $YEAR  `echo $(<meanlist.txt)`

# Edit output
cd meantemp_out
rm meantemp.csv
cut -d ',' -f3,4,5,6,7 meantemp_geostreams.csv > t0.csv
awk -F, '{print $4,$5,$1,$2,$3}' OFS=, t0.csv > t1.csv
sed 's/^.\{9\}//' t1.csv > meantemp.csv
sed -i '1d' meantemp.csv
sed -i '1 i\Plot, Temperature, Latitute, Longitude, Time' meantemp.csv

# cut -d ',' -f3,4,5,6,7 meantemp_geostreams.csv > t0.csv 
# awk -F, '{print $4,$5,$1,$2,$3}' OFS=, t0.csv > t1.csv
# sed '1 a Plot, Temperature, Latitute, Longitude, Time' t2.csv > meantemp.csv 

# Cleanup
rm meantemp_geostreams.csv t*
rm ../t0.txt ../meanlist.txt
