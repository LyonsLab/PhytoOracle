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
	
	# field.save_plot()

	# old_lid_base_error = field.calculate_lid_based_error()

	# old_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	field.create_patches_SIFT_files()
	
	field.draw_and_save_field(is_old=True)

	field.correct_field()

	field.save_new_coordinate()

	field.draw_and_save_field(is_old=False)


	# new_lid_base_error = field.calculate_lid_based_error()
	# new_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	# print('------------------ ERROR MEASUREMENT ------------------ ')


	# print('OLD Lid base Mean and Stdev: {0}'.format(old_lid_base_error))
	# print('OLD SI: {0}'.format(np.mean(old_RMSE[:,3])))
	

	# print('NEW Lid base Mean and Stdev: {0}'.format(new_lid_base_error))
	# print('NEW SI: {0}'.format(np.mean(new_RMSE[:,3])))



start_time = datetime.datetime.now()

scan_date = sys.argv[1]
config_file = sys.argv[2]
local_address = sys.argv[3]


settings.initialize_settings_test(scan_date,config_file,local_address,7,None)

print_settings()
main(scan_date)

end_time = datetime.datetime.now()

report_time(start_time,end_time)


