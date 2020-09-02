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

	field = Field(is_single_group=settings.is_single_group)
	
	field.save_plot()

	old_lid_base_error = field.calculate_lid_based_error()

	old_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	field.create_patches_SIFT_files()
	
	field.draw_and_save_field(is_old=True)

	field.correct_field()

	field.save_new_coordinate()

	new_lid_base_error = field.calculate_lid_based_error()
	new_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	print('------------------ ERROR MEASUREMENT ------------------ ')


	print('OLD Lid base Mean and Stdev: {0}'.format(old_lid_base_error))
	print('OLD SI: {0}'.format(np.mean(old_RMSE[:,3])))
	

	print('NEW Lid base Mean and Stdev: {0}'.format(new_lid_base_error))
	print('NEW SI: {0}'.format(np.mean(new_RMSE[:,3])))

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

os.mkdir('{0}/{1}'.format(destination,scan_date))
os.mkdir('{0}/{1}/SIFT'.format(destination,scan_date))
os.mkdir('{0}/{1}/logs'.format(destination,scan_date))

print('Geo-correction started. Log is being saved in {0}'.format(destination))

original = sys.stdout

sys.stdout = open('{0}/{1}/{2}.txt'.format(destination,scan_date,'geo_correction_output'), 'w')

settings.initialize_settings_HPC(scan_date,config_file,destination,lid_file_address,uav_lettuce_address,bin2tiff_address,gps_coord_file)

print_settings()
main(scan_date)

end_time = datetime.datetime.now()

report_time(start_time,end_time)

sys.stdout = original