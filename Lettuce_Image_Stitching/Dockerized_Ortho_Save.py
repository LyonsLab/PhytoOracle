import argparse
import numpy as np
import cv2
import random
import math
import multiprocessing
import datetime
import sys
import gc
import pickle
import os
import threading
import socket
import statistics
import datetime
from Customized_myltiprocessing import MyPool
from heapq import heappush, heappop, heapify
from collections import OrderedDict,Counter

from Block_wise_GPS_correction import *
import settings

def main(scan_date):

	sys.setrecursionlimit(10**6)
	
	settings.lettuce_coords = read_lettuce_heads_coordinates()

	field_new = Field(is_single_group=settings.is_single_group,use_corrected=True)
	field_new.draw_and_save_field(is_old=False)
	

def get_args():

	parser = argparse.ArgumentParser(description='Geo-correction on HPC.')
	parser.add_argument('-d','--destination', type=str, help='address of the destination folder on HPC (usually on xdisk where everything is stored).')
	parser.add_argument('-b','--bin_2tif', type=str, help='the address of the bin_2tif folder.')
	parser.add_argument('-g','--gps_coord', type=str, help='the address of the GPS coordinates csv file.')
	parser.add_argument('-s','--scan_date', type=str, help='the name of the specific scan to work on.')
	parser.add_argument('-c','--config_file', type=str, help='the name of the config file to use.')
	parser.add_argument('-l','--lid_address', type=str, help='the address of the lid file.')
	parser.add_argument('-u','--uav_lettuce_address', type=str, help='the address of the uav lettuce coordinates file.')

	args = parser.parse_args()

	return args

start_time = datetime.datetime.now()

args = get_args()

scan_date = args.scan_date
config_file = args.config_file
destination = args.destination
lid_file_address = args.lid_address
uav_lettuce_address = args.uav_lettuce_address
bin2tiff_address = args.bin_2tif
gps_coord_file = args.gps_coord


settings.initialize_settings_HPC(scan_date,config_file,destination,lid_file_address,uav_lettuce_address,bin2tiff_address,gps_coord_file)

main(scan_date)

end_time = datetime.datetime.now()

report_time(start_time,end_time)

sys.stdout = original