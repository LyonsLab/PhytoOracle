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
	
	old_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	field.create_patches_SIFT_files()
	
	field.draw_and_save_field(is_old=True)

	field.correct_field()

	field.save_new_coordinate()

	field.draw_and_save_field(is_old=False)

	new_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	print('------------------ ERROR MEASUREMENT ------------------ ')

	print('OLD SI: {0}'.format(np.mean(old_RMSE[:,3])))
	
	print('NEW SI: {0}'.format(np.mean(new_RMSE[:,3])))



start_time = datetime.datetime.now()

scan_date = sys.argv[1]
config_file = sys.argv[2]
local_address = sys.argv[3]

print('Geo-correction started. Log is being saved in {0}'.format(local_address))

original = sys.stdout
sys.stdout = open('{0}/{1}_out/{2}.txt'.format(local_address,scan_date,'geo_correction_output'), 'w+')

settings.initialize_settings_FLIR(scan_date,config_file,local_address)

print_settings()
main(scan_date)

end_time = datetime.datetime.now()

report_time(start_time,end_time)

sys.stdout = original

