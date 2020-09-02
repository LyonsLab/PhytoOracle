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
from heapq import heappush, heappop, heapify

PATCH_SIZE = (3296, 2472)
PATCH_SIZE_GPS = (8.899999997424857e-06,1.0199999998405929e-05)
HEIGHT_RATIO_FOR_ROW_SEPARATION = 0.1
NUMBER_OF_ROWS_IN_GROUPS = 10
NUMBER_OF_GOOD_MATCHES_FOR_GROUP_WISE_CORRECTION = 3000
GPS_TO_IMAGE_RATIO = (PATCH_SIZE_GPS[0]/PATCH_SIZE[1],PATCH_SIZE_GPS[1]/PATCH_SIZE[0])

def convert_to_gray(img):
	
	coefficients = [-1,1,2] 
	m = np.array(coefficients).reshape((1,3))
	img_g = cv2.transform(img, m)

	return img_g

def adjust_gamma(image, gamma=1.0):

	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	
	return cv2.LUT(image, table)

def normalize_img(img):

	# img_r = img[:,:,2]
	# img_g = img[:,:,1]
	# img_b = img[:,:,0]

	# img_r = (255*np.array((img_r-np.min(img_r))/(np.max(img_r)-np.min(img_r)))).astype('uint8')
	# img_g = (255*np.array((img_g-np.min(img_g))/(np.max(img_g)-np.min(img_g)))).astype('uint8')
	# img_b = (255*np.array((img_b-np.min(img_b))/(np.max(img_b)-np.min(img_b)))).astype('uint8')

	# img[:,:,2] = img_r
	# img[:,:,1] = img_g
	# img[:,:,0] = img_b

	img = (255*np.array((img-np.min(img))/(np.max(img)-np.min(img)))).astype('uint8')

	return img 

def load_preprocess_image(address):
	img = cv2.imread(address)
	# img = normalize_img(img)
	
	# img = adjust_gamma(img,1.2)
	img = img.astype('uint8')
	img_g = convert_to_gray(img)

	return img, img_g

def choose_SIFT_key_points(patch,x1,y1,x2,y2,SIFT_address,show=False):
	kp = []
	desc = []

	# (kp_tmp,desc_tmp) = pickle.load(open('{0}/{1}_SIFT.data'.format(SIFT_address,patch.name.replace('.tif','')), "rb"))
	kp_tmp = patch.SIFT_kp_locations
	desc_tmp = patch.SIFT_kp_desc

	for i,k in enumerate(kp_tmp):
		if k[0]>= x1 and k[0]<=x2 and k[1]>=y1 and k[1]<=y2:
			kp.append(k)

			desc.append(list(np.array(desc_tmp[i,:])))

	desc = np.array(desc)

	if show:
		patch.load_img('/storage/ariyanzarei/2020-01-08-rgb/bin2tif_out')

		img_res = patch.img.copy()
		# img_res = cv2.drawKeypoints(img_res,kp,img_res)
		ratio = img_res.shape[0]/img_res.shape[1]
		cv2.rectangle(img_res,(x1,y1),(x2,y2),(0,0,255),20)
		img_res = cv2.resize(img_res, (500, int(500*ratio))) 
		cv2.imshow('fig {0}'.format(patch.name),img_res)
		cv2.waitKey(0)
		# cv2.imwrite('tmp.bmp',img_res)

		patch.del_img()
	
	return kp,desc

def detect_SIFT_key_points(img,x1,y1,x2,y2,n,show=False):
	sift = cv2.xfeatures2d.SIFT_create()
	main_img = img.copy()
	img = img[y1:y2,x1:x2]
	kp,desc = sift.detectAndCompute(img,None)

	kp_n = []
	for k in kp:
		kp_n.append(cv2.KeyPoint(k.pt[0]+x1,k.pt[1]+y1,k.size))

	kp = kp_n

	if show:
		img_res = main_img.copy()
		img_res = cv2.drawKeypoints(img_res,kp_n,img_res)
		ratio = img_res.shape[0]/img_res.shape[1]
		cv2.rectangle(img_res,(x1,y1),(x2,y2),(0,0,255),20)
		img_res = cv2.resize(img_res, (500, int(500*ratio))) 
		cv2.imshow('fig {0}'.format(n),img_res)
		cv2.waitKey(0)	

	return kp_n,desc

def matched_distance(p1,p2):

	return math.sqrt(np.sum((p1-p2)**2))

def get_good_matches(desc1,desc2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		if m[0].distance < 0.8*m[1].distance:
			good.append(m)
	matches = np.asarray(good)

	return matches

def find_homography_gps_only(ov_2_on_1,ov_1_on_2,p1,p2):	

	# dst = [[ov_2_on_1[0],ov_2_on_1[3]] , [ov_2_on_1[2],ov_2_on_1[1]] , [ov_2_on_1[0],ov_2_on_1[1]]]
	# src = [[ov_1_on_2[0],ov_1_on_2[3]] , [ov_1_on_2[2],ov_1_on_2[1]] , [ov_1_on_2[0],ov_1_on_2[1]]]
	
	# dst = np.float32(dst)
	# src = np.float32(src)
	
	# H = cv2.getAffineTransform(dst, src)
	
	# H = np.append(H,np.array([[0,0,1]]),axis=0)
	# H[0:2,0:2] = np.array([[1,0],[0,1]])
	# return H

	ratio = ((p1.GPS_coords.UR_coord[0]-p1.GPS_coords.UL_coord[0])/p1.size[1],-(p1.GPS_coords.UL_coord[1]-p1.GPS_coords.LL_coord[1])/p1.size[0])

	diff_GPS = ((p1.GPS_coords.UL_coord[0]-p2.GPS_coords.UL_coord[0])/ratio[0],(p1.GPS_coords.UL_coord[1]-p2.GPS_coords.UL_coord[1])/ratio[1])

	H = np.eye(3)
	H[0,2] = diff_GPS[0]
	H[1,2] = diff_GPS[1]
	# print(H)
	return H

def find_homography(matches,kp1,kp2,ov_2_on_1,ov_1_on_2,add_gps):	
	if add_gps:
		dst = np.array([[ov_2_on_1[0],ov_2_on_1[3]]]).reshape(-1,1,2)
		dst = np.append(dst,np.array([[ov_2_on_1[2],ov_2_on_1[1]]]).reshape(-1,1,2),axis=0)
		src = np.array([[ov_1_on_2[0],ov_1_on_2[3]]]).reshape(-1,1,2)
		src = np.append(src,np.array([[ov_1_on_2[2],ov_1_on_2[1]]]).reshape(-1,1,2),axis=0)
		dst = np.append(dst,np.array([[ov_2_on_1[0],ov_2_on_1[1]]]).reshape(-1,1,2),axis=0)
		dst = np.append(dst,np.array([[ov_2_on_1[2],ov_2_on_1[3]]]).reshape(-1,1,2),axis=0)
		src = np.append(src,np.array([[ov_1_on_2[0],ov_1_on_2[1]]]).reshape(-1,1,2),axis=0)
		src = np.append(src,np.array([[ov_1_on_2[2],ov_1_on_2[3]]]).reshape(-1,1,2),axis=0)
		
	if len(matches)>1:
		# src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		# dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		src = np.float32([ kp1[m.queryIdx] for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx] for m in matches[:,0] ]).reshape(-1,1,2)
	else:
		return None,0

	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 9000, confidence = 0.999, refineIters = 15)
	
	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	return H,np.sum(masked)/len(masked)
	
def is_point_inside(p1_1,p1_2,p1_3,p1_4,p2):
	if p2[0]>=p1_1[0] and p2[0]<=p1_2[0] and p2[1]>=p1_1[1] and p2[1]<=p1_3[1]:
		return True
	else:
		return False

def get_overlapped_region(dst,rgb_img1,rgb_img2,H):
	c1 = [0,0,1]
	c2 = [rgb_img1.shape[1],0,1]
	c3 = [0,rgb_img1.shape[0],1]
	c4 = [rgb_img1.shape[1],rgb_img1.shape[0],1]

	c1 = H.dot(c1).astype(int)
	c2 = H.dot(c2).astype(int)
	c3 = H.dot(c3).astype(int)
	c4 = H.dot(c4).astype(int)

	c2_1 = [rgb_img1.shape[1],rgb_img1.shape[0],1]
	c2_2 = [rgb_img1.shape[1]+rgb_img2.shape[1],rgb_img1.shape[0],1]
	c2_3 = [rgb_img1.shape[1],rgb_img1.shape[0]+rgb_img2.shape[0],1]
	c2_4 = [rgb_img1.shape[1]+rgb_img2.shape[1],rgb_img1.shape[0]+rgb_img2.shape[0],1]
	
	p1_x = c2_1[0]
	p1_y = c2_1[1]
	p2_x = c2_4[0]
	p2_y = c2_4[1]

	# print(c2_1,c2_2,c2_3,c2_4)
	# print(c1,c2,c3,c4)

	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c1):
		p1_x = c1[0]
		p1_y = c1[1]
		# print('UL of img1')

	if is_point_inside(c1,c2,c3,c4,c2_4):
		p1_x = c1[0]
		p1_y = c1[1]
		# print('LR of img2')
	
	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c4):
		p2_y = c4[1]
		p2_x = c4[0]
		# print('LR of img1')

	if is_point_inside(c1,c2,c3,c4,c2_1):
		p2_y = c4[1]
		p2_x = c4[0]
		# print('UL of img2')

	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c2):
		p2_x = c2[0]
		p1_y = c2[1]
		# print('UR of img1')

	if is_point_inside(c1,c2,c3,c4,c2_3):
		p2_x = c2[0]
		p1_y = c2[1]
		# print('LL of img2')

	if is_point_inside(c2_1,c2_2,c2_3,c2_4,c3):
		p1_x = c3[0]
		p2_y = c3[1]
		# print('LL of img1')

	if is_point_inside(c1,c2,c3,c4,c2_2):
		p1_x = c3[0]
		p2_y = c3[1]
		# print('UR of img2')

	# cv2.rectangle(dst,(p1_x,p1_y),(p2_x,p2_y),(0,0,255),20)

	return np.copy(dst[p1_y:p2_y,p1_x:p2_x]),(p1_x,p1_y,p2_x,p2_y)

def revise_homography(H,rgb_img1,rgb_img2,img1,img2,move_steps,mse,length_rev):
	main_H = np.copy(H)
	min_H = np.copy(H)
	min_MSE = mse

	for r in range(0,length_rev):
		i = random.randint(-move_steps,move_steps+1)
		j = random.randint(-move_steps,move_steps+1)

		H = np.copy(main_H)
		H[0,2]+=i
		H[1,2]+=j

		dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
		overlap1,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

		dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
		overlap2,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)
		
		MSE = np.mean((overlap1-overlap2)**2)

		if MSE<min_MSE:
			min_MSE=MSE
			min_H = H 	


	return min_H

def stitch(rgb_img1,rgb_img2,img1,img2,H,overlap,show=False,write_out=False,apply_average=False,revise_h=False,revise_move_steps=20,length_rev=100):
	T = np.array([[1,0,rgb_img1.shape[1]],[0,1,rgb_img1.shape[0]],[0,0,1]])

	H = T.dot(H)
	# H = np.eye(3)

	# rgb_img2[overlap[1]:overlap[3],overlap[0]:overlap[2],:] = 0

	dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
	overlap1,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

	dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
	overlap2,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

	mse_overlap1 = overlap1.copy()
	mse_overlap1[overlap1==0] = overlap2[overlap1==0]
	mse_overlap2 = overlap2.copy()
	mse_overlap2[overlap2==0] = overlap1[overlap2==0]

	MSE = np.mean((mse_overlap1-mse_overlap2)**2)

	del mse_overlap1
	del mse_overlap2

	if revise_h:
		H = revise_homography(H,rgb_img1,rgb_img2,img1,img2,revise_move_steps,MSE,length_rev)

		dst = cv2.warpPerspective(rgb_img1, H,(rgb_img2.shape[1] + 2*rgb_img1.shape[1], rgb_img2.shape[0]+2*rgb_img1.shape[0]))
		overlap1,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)

		dst[rgb_img1.shape[0]:rgb_img1.shape[0]+rgb_img2.shape[0], rgb_img1.shape[1]:rgb_img1.shape[1]+rgb_img2.shape[1]] = rgb_img2
		overlap2,pnts = get_overlapped_region(dst,rgb_img1,rgb_img2,H)
		
		mse_overlap1 = overlap1.copy()
		mse_overlap1[overlap1==0] = overlap2[overlap1==0]
		mse_overlap2 = overlap2.copy()
		mse_overlap2[overlap2==0] = overlap1[overlap2==0]
		
		MSE = np.mean((mse_overlap1-mse_overlap2)**2)
		
		del mse_overlap1
		del mse_overlap2

	overlap1[overlap1==0]=overlap2[overlap1==0]
	overlap2[overlap2==0]=overlap1[overlap2==0]
	
	dst[pnts[1]:pnts[3],pnts[0]:pnts[2]] = overlap2

	if apply_average:

		final_average = ((overlap1+overlap2)/2).astype('uint8')
		dst[pnts[1]:pnts[3],pnts[0]:pnts[2]] = final_average

	ratio = dst.shape[0]/dst.shape[1]
	dst2 = cv2.resize(dst, (1300, int(1300*ratio))) 
	
	print('\tMSE: {0}'.format(MSE))

	gray = convert_to_gray(dst2)
	coords = cv2.findNonZero(gray) 
	x, y, w, h = cv2.boundingRect(coords) 
	dst2 = dst2[y:y+h, x:x+w,:]

	gray = convert_to_gray(dst)
	coords = cv2.findNonZero(gray) 
	x, y, w, h = cv2.boundingRect(coords) 
	dst = dst[y:y+h, x:x+w,:]
	
	if show:
		cv2.imshow('fig 3',dst2)
		cv2.waitKey(0)

	if write_out:
		cv2.imwrite('output.jpg',dst)

	return dst,MSE

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

class Patch_GPS_coordinate:
	def __init__(self,UL_coord,UR_coord,LL_coord,LR_coord,Center):
		self.UL_coord = UL_coord
		self.UR_coord = UR_coord
		self.LL_coord = LL_coord
		self.LR_coord = LR_coord
		self.Center = Center

	def is_coord_inside(self, coord):
		if coord[0]>=self.UL_coord[0] and coord[0]<=self.UR_coord[0] and coord[1]<=self.UL_coord[1] and coord[1]>=self.LL_coord[1]:
			return True
		else:
			return False

	def __str__(self):
		return '---------------------------\nUL:{0}\nUR:{1}\nLL:{2}\nLR:{3}\n---------------------------\n'.format(self.UL_coord,self.UR_coord,self.LL_coord,self.LR_coord)
	
class Patch:

	def __init__(self,name,rgb_img,img,coords,size=None):
		self.name = name
		self.rgb_img = rgb_img
		self.img = img
		self.GPS_coords = coords
		if size == None:
			self.size = np.shape(img)
		else:
			self.size = size
		self.GPS_Corrected = False
		self.area_score = 0
		self.overlaps = None
		self.SIFT_kp_locations = None
		self.SIFT_kp_desc = None


	def __le__(self,other):
		return (self.area_score<=other.area_score)

	def __lt__(self,other):
		return (self.area_score<other.area_score)

	def __eq__(self,other):
		return (self.name == other.name)

	def load_img(self,patch_folder):
		img,img_g = load_preprocess_image('{0}/{1}'.format(patch_folder,self.name))
		self.rgb_img = img
		self.img = img_g
		self.size = np.shape(img)


	def del_img(self):
		self.rgb_img = None
		self.img = None
		gc.collect()

	def load_SIFT(self,SIFT_address):
		# print('load sift for {0}'.format(self.name))
		(kp_tmp,desc_tmp) = pickle.load(open('{0}/{1}_SIFT.data'.format(SIFT_address,self.name.replace('.tif','')), "rb"))
		self.SIFT_kp_locations = kp_tmp
		self.SIFT_kp_desc = desc_tmp


	def del_SIFT(self):
		# print('delete sift for {0}'.format(self.name))
		self.SIFT_kp_locations = None
		self.SIFT_kp_desc = None
		gc.collect()

	def claculate_area_score(self):
		score = 0
		for n in self.overlaps:
			overlap_rect = self.get_overlap_rectangle(n,False)
			score+=abs(overlap_rect[2]-overlap_rect[0])*abs(overlap_rect[3]-overlap_rect[1])

		self.area_score = -1*score

	def correct_GPS_based_on_point(self,point_in_img,point_in_GPS):
		ratio_x = point_in_img[0]/self.size[1]
		ratio_y = point_in_img[1]/self.size[0]

		diff_x_GPS = (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0])*ratio_x
		diff_y_GPS = (self.GPS_coords.UL_coord[1]-self.GPS_coords.LL_coord[1])*ratio_y

		old_GPS_point = (self.GPS_coords.UL_coord[0]+diff_x_GPS,self.GPS_coords.UL_coord[1]-diff_y_GPS)

		diff_GPS_after_correction = (old_GPS_point[0]-point_in_GPS[0],old_GPS_point[1]-point_in_GPS[1])


		new_UR = (round(self.GPS_coords.UR_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.UR_coord[1]-diff_GPS_after_correction[1],7))
		new_UL = (round(self.GPS_coords.UL_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.UL_coord[1]-diff_GPS_after_correction[1],7))
		new_LL = (round(self.GPS_coords.LL_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.LL_coord[1]-diff_GPS_after_correction[1],7))
		new_LR = (round(self.GPS_coords.LR_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.LR_coord[1]-diff_GPS_after_correction[1],7))
		new_center = (round(self.GPS_coords.Center[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.Center[1]-diff_GPS_after_correction[1],7))

		new_coords = Patch_GPS_coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

		self.GPS_coords = new_coords

	def visualize_with_single_GPS_point(self,point,point_img,r):
		if self.rgb_img is None:
			return

		output = self.rgb_img.copy()
		cv2.circle(output,point_img,20,(0,255,0),thickness=-1)
		cv2.circle(output,point_img,r,(255,0,0),thickness=15)

		ratio = self.rgb_img.shape[0]/self.rgb_img.shape[1]
		output = cv2.resize(output, (500, int(500*ratio))) 

		ratio_x = (point[0] - self.GPS_coords.UL_coord[0])/(self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0])
		ratio_y = (self.GPS_coords.UL_coord[1] - point[1])/(self.GPS_coords.UL_coord[1]-self.GPS_coords.LL_coord[1])

		shp = np.shape(output)
		cv2.circle(output,(int(ratio_x*shp[1]),int(ratio_y*shp[0])),20,(0,0,255),thickness=-1)

		cv2.imshow('GPS',output)
		cv2.waitKey(0)



	def has_overlap(self,p):
		if self.GPS_coords.is_coord_inside(p.GPS_coords.UL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.UR_coord) or\
			self.GPS_coords.is_coord_inside(p.GPS_coords.LL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.LR_coord):
			return True
		else:
			return False

	def get_overlap_rectangle(self,patch,increase_size=True):
		p1_x = 0
		p1_y = 0
		p2_x = self.size[1]
		p2_y = self.size[0]

		detect_overlap = False

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.UL_coord):
		# 	detect_overlap = True
		# 	p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.LR_coord):
		# 	detect_overlap = True
		# 	p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.UR_coord):
		# 	detect_overlap = True
		# 	p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p1_y = int(((patch.GPS_coords.UR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.LL_coord):
		# 	detect_overlap = True
		# 	p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p2_y = int(((patch.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.UL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			# print(patch.name+' upper border is inside')
			# p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		if patch.GPS_coords.LL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.LL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			# print(patch.name+' lower border is inside')
			# p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UR_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UR_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			# print(patch.name+' right border is inside')
			p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			# p1_y = int(((patch.GPS_coords.UR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UL_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UL_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			# print(patch.name+' left border is inside')
			p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			# p2_y = int(((patch.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.is_coord_inside(self.GPS_coords.UL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.UR_coord) and \
		patch.GPS_coords.is_coord_inside(self.GPS_coords.LL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.LR_coord):
			p1_x = 0
			p1_y = 0
			p2_x = self.size[1]
			p2_y = self.size[0]
			detect_overlap = True

		if increase_size:
			if p1_x>0+self.size[1]/10:
				p1_x-=self.size[1]/10

			if p2_x<9*self.size[1]/10:
				p2_x+=self.size[1]/10

			if p1_y>0+self.size[0]/10:
				p1_y-=self.size[0]/10

			if p2_y<9*self.size[0]/10:
				p2_y+=self.size[0]/10

		if detect_overlap == False:
			return 0,0,0,0

		return int(p1_x),int(p1_y),int(p2_x),int(p2_y)

class Patch_2:

	def __init__(self,name,rgb_img,img,coords,size=None):
		self.name = name
		self.rgb_img = rgb_img
		self.img = img
		self.GPS_coords = coords
		if size == None:
			self.size = np.shape(img)
		else:
			self.size = size
		self.GPS_Corrected = False
		self.neighbors = []
		self.average_score = 0
	
	def __eq__(self,other):
		return (self.name == other.name)

	def __le__(self,other):
		return (self.average_score<=other.average_score)

	def __lt__(self,other):
		return (self.average_score<other.average_score)

	def calculate_average_score(self):
		score = 0

		for n in self.neighbors:
			if n[0].GPS_Corrected:
				score += (n[5]*n[4])/((n[1][2]-n[1][0])*(n[1][3]-n[1][1]))

		self.average_score = -1*score

	def load_img(self,patch_folder):
		img,img_g = load_preprocess_image('{0}/{1}'.format(patch_folder,self.name))
		self.rgb_img = img
		self.img = img_g
		self.size = np.shape(img)


	def del_img(self):
		self.rgb_img = None
		self.img = None
		gc.collect()


	def correct_GPS_based_on_point(self,point_in_img,point_in_GPS):
		ratio_x = point_in_img[0]/self.size[1]
		ratio_y = point_in_img[1]/self.size[0]

		diff_x_GPS = (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0])*ratio_x
		diff_y_GPS = (self.GPS_coords.UL_coord[1]-self.GPS_coords.LL_coord[1])*ratio_y

		old_GPS_point = (self.GPS_coords.UL_coord[0]+diff_x_GPS,self.GPS_coords.UL_coord[1]-diff_y_GPS)

		diff_GPS_after_correction = (old_GPS_point[0]-point_in_GPS[0],old_GPS_point[1]-point_in_GPS[1])


		new_UR = (round(self.GPS_coords.UR_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.UR_coord[1]-diff_GPS_after_correction[1],7))
		new_UL = (round(self.GPS_coords.UL_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.UL_coord[1]-diff_GPS_after_correction[1],7))
		new_LL = (round(self.GPS_coords.LL_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.LL_coord[1]-diff_GPS_after_correction[1],7))
		new_LR = (round(self.GPS_coords.LR_coord[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.LR_coord[1]-diff_GPS_after_correction[1],7))
		new_center = (round(self.GPS_coords.Center[0]-diff_GPS_after_correction[0],7),round(self.GPS_coords.Center[1]-diff_GPS_after_correction[1],7))

		new_coords = Patch_GPS_coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

		self.GPS_coords = new_coords

	def visualize_with_single_GPS_point(self,point,point_img,r):
		if self.rgb_img is None:
			return

		output = self.rgb_img.copy()
		cv2.circle(output,point_img,20,(0,255,0),thickness=-1)
		cv2.circle(output,point_img,r,(255,0,0),thickness=15)

		ratio = self.rgb_img.shape[0]/self.rgb_img.shape[1]
		output = cv2.resize(output, (500, int(500*ratio))) 

		ratio_x = (point[0] - self.GPS_coords.UL_coord[0])/(self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0])
		ratio_y = (self.GPS_coords.UL_coord[1] - point[1])/(self.GPS_coords.UL_coord[1]-self.GPS_coords.LL_coord[1])

		shp = np.shape(output)
		cv2.circle(output,(int(ratio_x*shp[1]),int(ratio_y*shp[0])),20,(0,0,255),thickness=-1)

		cv2.imshow('GPS',output)
		cv2.waitKey(0)

	def has_overlap(self,p):
		if self.GPS_coords.is_coord_inside(p.GPS_coords.UL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.UR_coord) or\
			self.GPS_coords.is_coord_inside(p.GPS_coords.LL_coord) or self.GPS_coords.is_coord_inside(p.GPS_coords.LR_coord):
			return True
		else:
			return False

	def get_overlap_rectangle(self,patch,increase_size=True):
		p1_x = 0
		p1_y = 0
		p2_x = self.size[1]
		p2_y = self.size[0]

		detect_overlap = False

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.UL_coord):
		# 	detect_overlap = True
		# 	p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.LR_coord):
		# 	detect_overlap = True
		# 	p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.UR_coord):
		# 	detect_overlap = True
		# 	p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p1_y = int(((patch.GPS_coords.UR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		# if self.GPS_coords.is_coord_inside(patch.GPS_coords.LL_coord):
		# 	detect_overlap = True
		# 	p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
		# 	p2_y = int(((patch.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.UL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			# print(patch.name+' upper border is inside')
			# p1_x = int(((patch.GPS_coords.UL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		if patch.GPS_coords.LL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.LL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			# print(patch.name+' lower border is inside')
			# p2_x = int(((patch.GPS_coords.LR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UR_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UR_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			# print(patch.name+' right border is inside')
			p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			# p1_y = int(((patch.GPS_coords.UR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UL_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UL_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			# print(patch.name+' left border is inside')
			p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			# p2_y = int(((patch.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.is_coord_inside(self.GPS_coords.UL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.UR_coord) and \
		patch.GPS_coords.is_coord_inside(self.GPS_coords.LL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.LR_coord):
			p1_x = 0
			p1_y = 0
			p2_x = self.size[1]
			p2_y = self.size[0]
			detect_overlap = True

		if increase_size:
			if p1_x>0+self.size[1]/10:
				p1_x-=self.size[1]/10

			if p2_x<9*self.size[1]/10:
				p2_x+=self.size[1]/10

			if p1_y>0+self.size[0]/10:
				p1_y-=self.size[0]/10

			if p2_y<9*self.size[0]/10:
				p2_y+=self.size[0]/10

		if detect_overlap == False:
			return 0,0,0,0

		return int(p1_x),int(p1_y),int(p2_x),int(p2_y)

def read_all_data():

	patches = []

	# with open('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/files.txt') as f:
	# 	lines = f.read()
	# 	for l in lines.split('\n'):
	# 		if l == '':
	# 			break

	# 		rgb,img = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/{0}.tif'.format(l))
	# 		patches.append(Patch(l,rgb,img,None))
	
	with open('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt') as f:
		lines = f.read()
		lines = lines.replace('"','')

		for l in lines.split('\n'):
			if l == '':
				break
			if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
				continue

			features = l.split(',')

			rgb,img = load_preprocess_image('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures/{0}'.format(features[0]))

			upper_left = (float(features[1]),float(features[2]))
			lower_left = (float(features[3]),float(features[4]))
			upper_right = (float(features[5]),float(features[6]))
			lower_right = (float(features[7]),float(features[8]))
			center = (float(features[9]),float(features[10]))

			coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)

			# rgb = cv2.putText(rgb, features[0].split('-')[0], (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 10, cv2.LINE_AA) 

			patch = Patch(l,rgb,img,coord)
			patches.append(patch)

	return patches

def parallel_patch_creator(address,filename,coord,SIFT_address,calc_SIFT):
	
	if calc_SIFT:
		rgb,img = load_preprocess_image('{0}/{1}'.format(address,filename))
		kp,desc = detect_SIFT_key_points(img,0,0,img.shape[1],img.shape[0],filename,False)
		# *** kp_tmp = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
		kp_tmp = [(p.pt[0], p.pt[1]) for p in kp]
		pickle.dump((kp_tmp,desc), open('{0}/{1}_SIFT.data'.format(SIFT_address,filename.replace('.tif','')), "wb"))
		del kp,kp_tmp,desc
		del rgb,img
		
	# print('Patch created and SIFT generated for {0}'.format(filename))
	sys.stdout.flush()
	
	size = (3296, 2472)

	p = Patch_2(filename,None,None,coord,size)
	
	return p

def parallel_patch_creator_helper(args):

	return parallel_patch_creator(*args)

def read_all_data_on_server(patches_address,metadatafile_address,SIFT_address,calc_SIFT):
	global no_of_cores_to_use

	# patches = []

	# with open(metadatafile_address) as f:
	# 	lines = f.read()
	# 	lines = lines.replace('"','')

	# 	for l in lines.split('\n'):
	# 		if l == '':
	# 			break
	# 		if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
	# 			continue

	# 		features = l.split(',')

	# 		filename = features[0]
	# 		upper_left = (float(features[1]),float(features[2]))
	# 		lower_left = (float(features[3]),float(features[4]))
	# 		upper_right = (float(features[5]),float(features[6]))
	# 		lower_right = (float(features[7]),float(features[8]))
	# 		center = (float(features[9]),float(features[10]))

	# 		print('{0}/{1}'.format(patches_address,filename))
	# 		sys.stdout.flush()
	# 		rgb,img = load_preprocess_image('{0}/{1}'.format(patches_address,filename))
	# 		kp,desc = detect_SIFT_key_points(img,0,0,img.shape[1],img.shape[0],filename,False)
			

	# 		coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)
	# 		size = np.shape(img)

	# 		patch = Patch(filename,None,None,coord,size)
			
	# 		patches.append(patch)

	# return patches

	# ----------------- parallelism SIFT detecting --------------------------

	
	args_list = []

	with open(metadatafile_address) as f:
		lines = f.read()
		lines = lines.replace('"','')

		for l in lines.split('\n'):
			if l == '':
				break
			if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
				continue

			features = l.split(',')

			filename = features[0]
			upper_left = (float(features[1]),float(features[2]))
			lower_left = (float(features[3]),float(features[4]))
			upper_right = (float(features[5]),float(features[6]))
			lower_right = (float(features[7]),float(features[8]))
			center = (float(features[9]),float(features[10]))

			coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)
			
			args_list.append((patches_address,filename,coord,SIFT_address,calc_SIFT))

		
		processes = multiprocessing.Pool(no_of_cores_to_use)
		
		# iterable = processes.imap(parallel_patch_creator_helper,args_list)
		# results = []
		# for arg in args_list:
		# 	r = next(iterable)
		# 	# tmp_kp = [cv2.KeyPoint(x=p[0][0],y=p[0][1],_size=p[1], _angle=p[2],_response=p[3], _octave=p[4], _class_id=p[5]) for p in r.Keypoints_location] 
		# 	# r.Keypoints_location = tmp_kp
		# 	results.append(r)
		# 	gc.collect()
			


		results = processes.map(parallel_patch_creator_helper,args_list)
		processes.close()

		# for r in results:
		# 	tmp_kp = [cv2.KeyPoint(x=p[0][0],y=p[0][1],_size=p[1], _angle=p[2],_response=p[3], _octave=p[4], _class_id=p[5]) for p in r.Keypoints_location] 
		# 	r.Keypoints_location = tmp_kp

	return results
	
def draw_GPS_coords_on_patch(patch,coord1,coord2,coord3,coord4):
	img = patch.rgb_img.copy()

	if patch.GPS_coords.is_coord_inside(coord1):
		
		x = int(((coord1[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord1[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(0,0,255),thickness=-1)
	if patch.GPS_coords.is_coord_inside(coord2):
		
		x = int(((coord2[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord2[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(0,255,0),thickness=-1)
	if patch.GPS_coords.is_coord_inside(coord3):
		
		x = int(((coord3[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord3[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(255,0,0),thickness=-1)
	if patch.GPS_coords.is_coord_inside(coord4):
		
		x = int(((coord4[0]-patch.GPS_coords.UL_coord[0]) / (patch.GPS_coords.UR_coord[0]-patch.GPS_coords.UL_coord[0]))*patch.size[1])
		y = int(((coord4[1]-patch.GPS_coords.UR_coord[1]) / (patch.GPS_coords.LR_coord[1]-patch.GPS_coords.UR_coord[1]))*patch.size[0])
		cv2.circle(img,(x,y),100,(255,0,255),thickness=-1)

	return img

def get_orientation(p1,p2,o1,o2):

	if o1[0]==0 and o1[1]==0 and o1[3]==p1.size[0]:
		# return '2 on left of 1'
		return 0
	if o2[0]==0 and o2[1]==0 and o2[3]==p2.size[0]:
		# return '1 on left of 2'
		return 1

	if o1[0]==0 and o1[1]==0 and o1[2]<p1.size[1]:
		# return '2 over 1 to the left'
		return 2
	if o1[0]>0 and o1[1]==0 and o1[2]==p1.size[1]:
		# return '2 over 1 to the right'
		return 3
	if o1[0]==0 and o1[1]==0 and o1[2]==p1.size[1]:
		# return '2 over 1'
		return 4
	
	if o2[0]==0 and o2[1]==0 and o2[2]<p2.size[1]:
		# return '1 over 2 to the left'
		return 5
	if o2[0]>0 and o2[1]==0 and o2[2]==p2.size[1]:
		# return '1 over 2 to the right'
		return 6
	if o2[0]==0 and o2[1]==0 and o2[2]==p2.size[1]:
		# return '1 over 2'
		return 7


	return 'else'

def find_stitched_coords(coord1,coord2):
	min_dim_0 = min(coord1.UL_coord[0],coord1.UR_coord[0],coord2.UL_coord[0],coord2.UR_coord[0])
	max_dim_0 = max(coord1.UL_coord[0],coord1.UR_coord[0],coord2.UL_coord[0],coord2.UR_coord[0])
	min_dim_1 = min(coord1.UL_coord[1],coord1.LL_coord[1],coord2.UL_coord[1],coord2.LL_coord[1])
	max_dim_1 = max(coord1.UL_coord[1],coord1.LL_coord[1],coord2.UL_coord[1],coord2.LL_coord[1])

	coord = Patch_GPS_coordinate((min_dim_0,max_dim_1),(max_dim_0,max_dim_1),(min_dim_0,min_dim_1),(max_dim_0,min_dim_1),((min_dim_0+max_dim_0)/2,(min_dim_1+max_dim_1)/2))
	
	return coord

def choose_patch_with_largest_overlap(patches):
	max_f = 0
	max_p = None

	for p in patches:
		new_patches=[ptmp for ptmp in patches if ptmp!=p]
		overlaps = [p_n for p_n in new_patches if (p.has_overlap(p_n) or p_n.has_overlap(p))]
		p_grbg ,f = get_best_overlap(p,overlaps)

		if f>max_f:
			maxf = f
			max_p = p

	return max_p

def get_best_overlap_based_perc_inliers(patch):

	corr_neighbors = [n[0] for n in patch.neighbors if n[0].GPS_Corrected]

	best_neighbor = None
	best_neighbor_param = [None,None,None,None,None,None,None]
	best_neighbor_score = -1

	for p in corr_neighbors:
		patch_from_other_way = [n for n in p.neighbors if n[0]==patch]
		if len(patch_from_other_way) == 0:
			continue

		patch_from_other_way = patch_from_other_way[0]
		score = (patch_from_other_way[5]*patch_from_other_way[4])/((patch_from_other_way[1][2]-patch_from_other_way[1][0])*(patch_from_other_way[1][3]-patch_from_other_way[1][1]))

		if score>best_neighbor_score:
			best_neighbor_score = score
			best_neighbor = p
			best_neighbor_param = patch_from_other_way

	return best_neighbor,best_neighbor_param[2],best_neighbor_param[1],best_neighbor_param[3],best_neighbor_param[4],best_neighbor_param[5],best_neighbor_param[6]

def get_best_overlap(p,overlaps,increase_size=True):
	max_f = -1
	max_p = None

	for p_ov in overlaps:
		rect_on_1 = p.get_overlap_rectangle(p_ov,increase_size)
		rect_on_2 = p_ov.get_overlap_rectangle(p,increase_size)
		f = (rect_on_1[2]-rect_on_1[0])*(rect_on_1[3]-rect_on_1[1])
		# f += abs(np.mean(p.img[rect_on_1[1]:rect_on_1[3],rect_on_1[0]:rect_on_1[2]])-np.mean(p_ov.img[rect_on_2[1]:rect_on_2[3],rect_on_2[0]:rect_on_2[2]]))*-1
		if f>max_f:
			max_f = f
			max_p = p_ov

	return max_p,max_f

def Get_GPS_Error(H,ov_1_on_2,ov_2_on_1):
	p2 = np.array([[ov_1_on_2[0],ov_1_on_2[1]]])
	p1 = np.array([[ov_2_on_1[0],ov_2_on_1[1]]])
	
	p1_translated = H.dot([p1[0,0],p1[0,1],1])

	# return int(matched_distance(p2,np.array([p1_translated[0],p1_translated[1]])))
	return int(abs(p2[0,0]-p1_translated[0])),int(abs(p2[0,1]-p1_translated[1]))

def get_new_GPS_Coords(p1,p2,H):

	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)
	
	# print(c1)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (p2.GPS_coords.UR_coord[0] - p2.GPS_coords.UL_coord[0])/(p2.size[1])
	gps_scale_y = (p2.GPS_coords.LL_coord[1] - p2.GPS_coords.UL_coord[1])/(p2.size[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	# print(diff_x,diff_y)
	# print(p1.GPS_coords.UL_coord)
	new_UL = (round(p2.GPS_coords.UL_coord[0]-diff_x,7),round(p2.GPS_coords.UL_coord[1]-diff_y,7))
	# print(new_UL)
	diff_UL = (p1.GPS_coords.UL_coord[0]-new_UL[0],p1.GPS_coords.UL_coord[1]-new_UL[1])

	new_UR = (p1.GPS_coords.UR_coord[0]-diff_UL[0],p1.GPS_coords.UR_coord[1]-diff_UL[1])
	new_LL = (p1.GPS_coords.LL_coord[0]-diff_UL[0],p1.GPS_coords.LL_coord[1]-diff_UL[1])
	new_LR = (p1.GPS_coords.LR_coord[0]-diff_UL[0],p1.GPS_coords.LR_coord[1]-diff_UL[1])
	new_center = (p1.GPS_coords.Center[0]-diff_UL[0],p1.GPS_coords.Center[1]-diff_UL[1])

	new_coords = Patch_GPS_coordinate(new_UL,new_UR,new_LL,new_LR,new_center)

	return new_coords

def stitch_complete(patches,show,show2):
	patches_tmp = patches.copy()

	i = 0

	len_no_change = 0

	while len(patches_tmp)>1 and len_no_change<len(patches_tmp):
		p = patches_tmp.pop()
		len_no_change+=1

		# p = choose_patch_with_largest_overlap(patches_tmp)
		# patches_tmp.remove(p)

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		# p2 = overlaps[0]
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			# print(ov_2_on_1)
			# print(ov_1_on_2)
			# print(p.GPS_coords)
			# print(p2.GPS_coords)
			# cv2.namedWindow('fig1',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('fig2',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('fig1', 600,600)
			# cv2.resizeWindow('fig2', 600,600)
			# cv2.imshow('fig1',p.rgb_img)
			# cv2.imshow('fig2',p2.rgb_img)
			# cv2.waitKey(0)
			continue

		avg_overlap_1 = np.sum(p.img[ov_2_on_1[1]:ov_2_on_1[3],ov_2_on_1[0]:ov_2_on_1[2]])
		avg_overlap_2 = np.sum(p.img[ov_1_on_2[1]:ov_1_on_2[3],ov_1_on_2[0]:ov_1_on_2[2]])

		# if avg_overlap_1 == 0 or avg_overlap_2 == 0:
		# 	print('Type3 >>> Blank overlap for {0}. push back...'.format(p.name))
		# 	patches_tmp.insert(0,p)
		# 	continue

		patches_tmp.remove(p2)
		
		kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show)
		kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

		print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))

		if percentage_inliers<=0.1 or len(matches) < 40:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('\t*** Not enough inliers ...')
			continue

		gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and \
		gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('*** High GPS error ...')
			continue


		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H,ov_1_on_2,show)

		if MSE > 90:
			patches_tmp.insert(0,p)
			patches_tmp.insert(0,p2)
			print('\t*** High MSE ...')
			continue

		len_no_change = 0
		
		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		del p
		del p2

		i+=1

	print('############ Stitching based on GPS only ############')

	len_no_change = 0 

	while len(patches_tmp)>1 and len_no_change<len(patches_tmp):
		p = patches_tmp.pop()
		len_no_change+=1

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps,increase_size=False)
		# p2 = overlaps[0]
		
		ov_2_on_1 = p.get_overlap_rectangle(p2,increase_size=False)
		ov_1_on_2 = p2.get_overlap_rectangle(p,increase_size=False)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			# print(ov_2_on_1)
			# print(ov_1_on_2)
			# print(p.GPS_coords)
			# print(p2.GPS_coords)
			# cv2.namedWindow('fig1',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('fig2',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('fig1', 600,600)
			# cv2.resizeWindow('fig2', 600,600)
			# img_res1 = p.rgb_img
			# img_res2 = p2.rgb_img
			# cv2.rectangle(img_res1,(ov_2_on_1[0],ov_2_on_1[1]),(ov_2_on_1[2],ov_2_on_1[3]),(0,0,255),20)
			# cv2.rectangle(img_res2,(ov_1_on_2[0],ov_1_on_2[1]),(ov_1_on_2[2],ov_1_on_2[3]),(0,0,255),20)
			# cv2.imshow('fig1',img_res1)
			# cv2.imshow('fig2',img_res2)
			# cv2.waitKey(0)
			continue

		avg_overlap_1 = np.sum(p.img[ov_2_on_1[1]:ov_2_on_1[3],ov_2_on_1[0]:ov_2_on_1[2]])
		avg_overlap_2 = np.sum(p.img[ov_1_on_2[1]:ov_1_on_2[3],ov_1_on_2[0]:ov_1_on_2[2]])

		patches_tmp.remove(p2)

		kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show2)
		kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show2)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,True)
		
		print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))

		if percentage_inliers==0 or H is None:
			H = find_homography_gps_only(ov_2_on_1,ov_1_on_2)
		else:
			gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and \
		gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			H = find_homography_gps_only(ov_2_on_1,ov_1_on_2)

		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H,ov_1_on_2,show2)

		len_no_change = 0

		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		del p
		del p2

		i+=1

	return patches_tmp

def correct_GPS_coords(patches,show,show2,SIFT_address):

	patches_tmp = patches.copy()
	not_corrected_patches = patches.copy()

	i = 0

	best_f = 0
	best_p = patches_tmp[0]

	for p in patches_tmp:
		overlaps = [p_n for p_n in patches_tmp if p_n!=p and (p.has_overlap(p_n) or p_n.has_overlap(p))]
		p.overlaps = overlaps

		f = len(overlaps)

		if f>best_f:
			best_f = f
			best_p = p

	best_p.GPS_Corrected = True
	not_corrected_patches.remove(best_p)

	while True:

		sys.stdout.flush()

		# not_corrected_patches = [p for p in patches_tmp if p.GPS_Corrected == False]
		if len(not_corrected_patches) == 0:
			break

		p = not_corrected_patches.pop()

		overlaps = [p_n for p_n in p.overlaps if ((p_n.GPS_Corrected))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			not_corrected_patches.insert(0,p)

			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			not_corrected_patches.insert(0,p)
			
			continue

		# not_corrected_patches.insert(0,p)

		# kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show)
		# kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show)

		kp1,desc1 = choose_SIFT_key_points(p,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],SIFT_address)
		kp2,desc2 = choose_SIFT_key_points(p2,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],SIFT_address)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

		print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))

		if percentage_inliers<=0.1 or len(matches) < 40:
			not_corrected_patches.insert(0,p)
			# patches_tmp.insert(0,p2)
			print('\t*** Not enough inliers ...')
			continue

		gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and \
		gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			not_corrected_patches.insert(0,p)
			# patches_tmp.insert(0,p2)
			print('*** High GPS error ...')
			continue

		if show2:
			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

		p.GPS_coords = get_new_GPS_Coords(p,p2,H)
		p.GPS_Corrected = True
		# not_corrected_patches.remove(p)

		print('GPC corrected for {0}.'.format(p.name))

		if show2:
			ov_2_on_1 = p.get_overlap_rectangle(p2)
			ov_1_on_2 = p2.get_overlap_rectangle(p)

			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

	return patches_tmp

def visualize_single_run(H,p,p2,x1,y1,x2,y2,x11,y11,x22,y22,SIFT_address):
	print(p.name)
	print(p2.name)

	img1,black1 = load_preprocess_image('{0}/{1}'.format(SIFT_address.replace('SIFT','bin2tif_out'),p.name))
	img2,black2 = load_preprocess_image('{0}/{1}'.format(SIFT_address.replace('SIFT','bin2tif_out'),p2.name))

	ratio = img1.shape[0]/img1.shape[1]
	cv2.rectangle(img1,(x1,y1),(x2,y2),(0,0,255),20)
	img1 = cv2.resize(img1, (500, int(500*ratio))) 

	ratio = img2.shape[0]/img2.shape[1]
	cv2.rectangle(img2,(x11,y11),(x22,y22),(0,0,255),20)
	img2 = cv2.resize(img2, (500, int(500*ratio))) 

	cv2.imshow('fig {0}'.format(1),img1)
	cv2.imshow('fig {0}'.format(2),img2)
	cv2.waitKey(0)	

	stitch(img1,img2,black1,black2,H,(x11,y11,x22,y22),True)

def evaluate_beneficiary_overlap(p1,p2,H,patch_folder,ov1,ov2):

	p1_ul = [0,0,1]
	p1_ur = [p1.size[1],0,1]
	p1_ll = [0,p1.size[0],1]
	p1_lr = [p1.size[1],p1.size[0],1]

	p1_ul_new = H.dot(p1_ul).astype(int)
	p1_ur_new = H.dot(p1_ur).astype(int)
	p1_ll_new = H.dot(p1_ll).astype(int)
	p1_lr_new = H.dot(p1_lr).astype(int)
	
	p1_x1 = 0
	p1_y1 = 0
	p1_x2 = p1.size[1]
	p1_y2 = p1.size[0]

	p2_x1 = 0
	p2_y1 = 0
	p2_x2 = p2.size[1]
	p2_y2 = p2.size[0]

	flag = False

	if p1_ul_new[0]>=0 and p1_ul_new[0]<p2.size[1] and p1_ul_new[1]>=0 and p1_ul_new[1]<p2.size[0]:
		p2_x1 = p1_ul_new[0]
		p2_y1 = p1_ul_new[1]

		p1_x2 = p2.size[1] - p1_ul_new[0]
		p1_y2 = p2.size[0] - p1_ul_new[1]

		flag = True

	if p1_ur_new[0]>=0 and p1_ur_new[0]<p2.size[1] and p1_ur_new[1]>=0 and p1_ur_new[1]<p2.size[0]:
		p2_x2 = p1_ur_new[0]
		p2_y1 = p1_ur_new[1]

		p1_x1 = p2.size[1] - p1_ur_new[0]
		p1_y2 = p2.size[0] - p1_ur_new[1]

		flag = True

	if p1_ll_new[0]>=0 and p1_ll_new[0]<p2.size[1] and p1_ll_new[1]>=0 and p1_ll_new[1]<p2.size[0]:
		p2_x1 = p1_ll_new[0]
		p2_y2 = p1_ll_new[1]

		p1_x2 = p2.size[1] - p1_ll_new[0]
		p1_y1 = p2.size[0] - p1_ll_new[1]

		flag = True

	if p1_lr_new[0]>=0 and p1_lr_new[0]<p2.size[1] and p1_lr_new[1]>=0 and p1_lr_new[1]<p2.size[0]:
		p2_x2 = p1_lr_new[0]
		p2_y2 = p1_lr_new[1]

		p1_x1 = p2.size[1] - p1_lr_new[0]
		p1_y1 = p2.size[0] - p1_lr_new[1]

		flag = True

	if not flag:
		p1_x1 = 0
		p1_y1 = 0
		p1_x2 = 0
		p1_y2 = 0

		p2_x1 = 0
		p2_y1 = 0
		p2_x2 = 0
		p2_y2 = 0

		print('out of bound overlap {0} to {1}.'.format(p1.name,p2.name))
		return -1

	p1.load_img(patch_folder)
	p2.load_img(patch_folder)
	overlap_1_img = p1.rgb_img[p1_y1:p1_y2,p1_x1:p1_x2,:]
	overlap_2_img = p2.rgb_img[p2_y1:p2_y2,p2_x1:p2_x2,:]

	shape_1 = np.shape(overlap_1_img)
	shape_2 = np.shape(overlap_2_img)

	if shape_1[0] == 0 or shape_1[1] == 0 or shape_2[0] == 0 or shape_2[1] == 0:
		print('very small overlap after transformation {0} to {1}.'.format(p1.name,p2.name))
		return -1

	overlap_1_img = cv2.cvtColor(overlap_1_img, cv2.COLOR_BGR2GRAY)
	overlap_2_img = cv2.cvtColor(overlap_2_img, cv2.COLOR_BGR2GRAY)

	overlap_1_img = cv2.blur(overlap_1_img,(5,5))
	overlap_2_img = cv2.blur(overlap_2_img,(5,5))

	ret1,overlap_1_img = cv2.threshold(overlap_1_img,0,255,cv2.THRESH_OTSU)
	ret1,overlap_2_img = cv2.threshold(overlap_2_img,0,255,cv2.THRESH_OTSU)

	tmp_size = np.shape(overlap_1_img)
	
	overlap_1_img[overlap_1_img==255] = 1
	overlap_2_img[overlap_2_img==255] = 1

	xnor_images = np.logical_xor(overlap_1_img,overlap_2_img)

	# dissimilarity = round(np.sum(xnor_images)/(tmp_size[0]*tmp_size[1]),2)
	dissimilarity = np.sum(xnor_images)

	
	
	# print(dissimilarity)

	overlap_1_img[overlap_1_img==1] = 255
	overlap_2_img[overlap_2_img==1] = 255

	cv2.namedWindow('1',cv2.WINDOW_NORMAL)
	cv2.namedWindow('2',cv2.WINDOW_NORMAL)
	cv2.namedWindow('p1',cv2.WINDOW_NORMAL)
	cv2.namedWindow('p2',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('1', 500,500)
	cv2.resizeWindow('2', 500,500)
	cv2.resizeWindow('p1', 500,500)
	cv2.resizeWindow('p2', 500,500)
	img1 = p1.rgb_img
	img2 = p2.rgb_img

	cv2.rectangle(img1,(p1_x1,p1_y1),(p1_x2,p1_y2),(0,0,255),20)
	cv2.rectangle(img2,(p2_x1,p2_y1),(p2_x2,p2_y2),(0,0,255),20)

	cv2.rectangle(img1,(ov1[0],ov1[1]),(ov1[2],ov1[3]),(0,255,0),20)
	cv2.rectangle(img2,(ov2[0],ov2[1]),(ov2[2],ov2[3]),(0,255,0),20)

	cv2.imshow('p1',img1)
	cv2.imshow('p2',img2)
	cv2.imshow('1',overlap_1_img)
	cv2.imshow('2',overlap_2_img)
	cv2.waitKey(0)

	p1.del_img()
	p2.del_img()
	
	return dissimilarity

def get_is_above(p1,p2):

	if p1.GPS_coords.Center[1] > p2.GPS_coords.Center[1]:
		return True
	else:
		return False

def get_pairwise_transformation_info(p1,p2,SIFT_address,patch_folder):

	overlap1 = p1.get_overlap_rectangle(p2)
	overlap2 = p2.get_overlap_rectangle(p1)
	
	if overlap1[2]-overlap1[0]<p1.size[1]/5 and overlap1[3]-overlap1[1]<p1.size[0]/5:
		# very small overlap
		# print('very small overlap')
		return None,None,None,None,None,None,None

	kp1,desc1 = choose_SIFT_key_points(p1,overlap1[0],overlap1[1],overlap1[2],overlap1[3],SIFT_address)
	kp2,desc2 = choose_SIFT_key_points(p2,overlap2[0],overlap2[1],overlap2[2],overlap2[3],SIFT_address)

	matches = get_good_matches(desc2,desc1)

	num_matches = len(matches)

	H,percentage_inliers = find_homography(matches,kp2,kp1,overlap1,overlap2,False)

	if H is None:
		# low number of matches, bad transformation
		# print('low number matches')
		return None,None,None,None,None,None,None

	percentage_inliers = round(percentage_inliers*100,2)

	gps_err = Get_GPS_Error(H,overlap2,overlap1)

	overlap_status = evaluate_beneficiary_overlap(p1,p2,H,patch_folder,overlap1,overlap2)

	# is_above = get_is_above(p1,p2) or get_is_above(p2,p1)

	# if is_above and 1-overlap_status < 0.60:
	# 	# bad vertical overlap
	# 	# print('bad vertical overlap.')
	# 	return None,None,None,None,None,None,None

	if overlap_status == -1:
		# outside of the boundry transformation
		return None,None,None,None,None,None,None

	return overlap1,overlap2,H,num_matches,percentage_inliers,gps_err,overlap_status

def precalculate_pairwise_transformation_info_and_add_neighbors(patches,SIFT_address,group_id,patch_folder):
	remove_neighbors = []
	main_patch = None

	for p in patches:

		if p.GPS_Corrected:
			main_patch = p

		for n in patches:

			if n != p and (p.has_overlap(n) or n.has_overlap(p)):

				overlap_on_n,overlap_on_p,H,num_matches,percentage_inliers,gps_err,dissimilarity = get_pairwise_transformation_info(n,p,SIFT_address,patch_folder)
				
				# do not add if not a good overlap
				if overlap_on_n == None:
					remove_neighbors.append((n,p))
					continue
				

				p.neighbors.append((n,overlap_on_n,overlap_on_p,H,num_matches,percentage_inliers,gps_err,dissimilarity))

		print('GROPU ID: {0} - Calculated Transformation and error values for {1} neighbors of {2}'.format(group_id,len(p.neighbors),p.name))
		sys.stdout.flush()

	for a,b in remove_neighbors:
		new_neighbors = []

		for n in a.neighbors:
			if b != n[0]:
				new_neighbors.append(n)
			else:
				print('{0} removed from {1} neighbors.'.format(b.name,a.name))
		a.neighbors = new_neighbors


	if main_patch is None:
		main_patch = patches[0]
		main_patch.GPS_Corrected = True

	return main_patch			

def get_list_of_corrected_neighbors_queue_1(patch,group_id):
	
	corrected_neighbors = []

	for n,overlap_on_n,overlap_on_p,H,num_matches,percentage_inliers,gps_err,dissimilarity in patch.neighbors:
		
		if (not n.GPS_Corrected):

			if (percentage_inliers>10 and num_matches >= 40):
				if (gps_err[0] <= (overlap_on_p[2]-overlap_on_p[0])/2 and gps_err[1] <= (overlap_on_p[3]-overlap_on_p[1])/2):

					result_string = 'GROPU ID: {0} - '.format(group_id)
					result_string +='Patch {0}'.format(n.name)
					
					n.GPS_coords = get_new_GPS_Coords(n,patch,H)
					
					n.GPS_Corrected = True

					result_string+=' <Q1: Corrected> --> ({0},{1}%,<{2},{3}>,{4})'.format(num_matches,percentage_inliers,gps_err[0],gps_err[1],\
					calculate_overlap_area_number(overlap_on_p))
					print(result_string)

					corrected_neighbors.append(n)
			# 	else:
			# 		print('--- {0} NOT CORRECTED <GPS Error>'.format(n.name))
			# else:
			# 	print('--- {0} NOT CORRECTED <LOW INLIER>'.format(n.name))
	
	sys.stdout.flush()

	return corrected_neighbors

def get_list_of_corrected_neighbors_queue_2(patch,group_id):
	
	corrected_neighbors = []

	for n,overlap_on_n,overlap_on_p,H,num_matches,percentage_inliers,gps_err,dissimilarity in patch.neighbors:
		if not n.GPS_Corrected:
			result_string = 'GROPU ID: {0} - '.format(group_id)
			result_string +='Patch {0}'.format(n.name)

			if (gps_err[0] <= (overlap_on_p[2]-overlap_on_p[0])/2 and gps_err[1] <= (overlap_on_p[3]-overlap_on_p[1])/2):
				
				n.GPS_coords = get_new_GPS_Coords(n,patch,H)

				n.GPS_Corrected = True
				
				result_string+=' <Q2: Low Inliers Relaxed> --> ({0},{1}%,<{2},{3}>,{4})'.format(num_matches,percentage_inliers,gps_err[0],gps_err[1],\
					calculate_overlap_area_number(overlap_on_p))

				corrected_neighbors.append(n)
			else:
				n.GPS_Corrected = True

				result_string+=' <Q2: GPS Relaxed> --> ({0},{1}%,<{2},{3}>,{4},{5})'.format(num_matches,percentage_inliers,gps_err[0],gps_err[1],\
					calculate_overlap_area_number(overlap_on_p))

				corrected_neighbors.append(n)

			print(result_string)
	
	sys.stdout.flush()

	return corrected_neighbors

def get_corrected_string(patches):

	final_results = ''

	for p in patches:
		p.GPS_coords.UL_coord = (round(p.GPS_coords.UL_coord[0],7),round(p.GPS_coords.UL_coord[1],7))
		p.GPS_coords.LL_coord = (round(p.GPS_coords.LL_coord[0],7),round(p.GPS_coords.LL_coord[1],7))
		p.GPS_coords.UR_coord = (round(p.GPS_coords.UR_coord[0],7),round(p.GPS_coords.UR_coord[1],7))
		p.GPS_coords.LR_coord = (round(p.GPS_coords.LR_coord[0],7),round(p.GPS_coords.LR_coord[1],7))
		p.GPS_coords.Center = (round(p.GPS_coords.Center[0],7),round(p.GPS_coords.Center[1],7))

		final_results += '{:s},"{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}"\n'\
		.format(p.name,p.GPS_coords.UL_coord[0],p.GPS_coords.UL_coord[1],p.GPS_coords.LL_coord[0],p.GPS_coords.LL_coord[1],p.GPS_coords.UR_coord[0],p.GPS_coords.UR_coord[1]\
			,p.GPS_coords.LR_coord[0],p.GPS_coords.LR_coord[1],p.GPS_coords.Center[0],p.GPS_coords.Center[1])

	final_results = final_results.replace('(','"').replace(')','"')

	return final_results

def pop_best_from_queue(queue):
	patch = queue[0]

	queue.remove(patch)
	return patch

def calculate_overlap_area_number(overlap):

	return abs(overlap[0]-overlap[2])*abs(overlap[1]-overlap[3])


class Graph():

	def __init__(self,no_vertex,vertex_names):
		self.vertecis_number = no_vertex
		self.vertex_index_to_name_dict = {}
		self.vertex_name_to_index_dict = {}
		for i,v in enumerate(vertex_names):
			self.vertex_index_to_name_dict[i] = v
			self.vertex_name_to_index_dict[v] = i

		self.edges = [[-1 for column in range(no_vertex)] for row in range(no_vertex)]

	def initialize_edge_weights(self,patches):
		
		for p in patches:
			for n in p.neighbors:
				# print(p in [ne[0] for ne in n[0].neighbors])

				if self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]] == -1:
					self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]] = round(n[7],2)
					self.edges[self.vertex_name_to_index_dict[n[0].name]][self.vertex_name_to_index_dict[p.name]] =  round(n[7],2)
				else:
					if n[7] > self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]]:
						self.edges[self.vertex_name_to_index_dict[p.name]][self.vertex_name_to_index_dict[n[0].name]] = round(n[7],2)
						self.edges[self.vertex_name_to_index_dict[n[0].name]][self.vertex_name_to_index_dict[p.name]] = round(n[7],2)

	def print_graph(self):
		print(self.vertex_name_to_index_dict)
		print(self.edges)


	def find_min_key(self,keys,mstSet):
		min_value = sys.maxsize

		for v in range(self.vertecis_number): 
			if keys[v] < min_value and mstSet[v] == False: 
				min_value = keys[v] 
				min_index = v 

		return min_index 


	def generate_MST_prim(self,starting_vertex):
		keys = [sys.maxsize]*self.vertecis_number
		parents = [None]*self.vertecis_number
		mstSet = [False]*self.vertecis_number

		keys[self.vertex_name_to_index_dict[starting_vertex]] = 0
		parents[self.vertex_name_to_index_dict[starting_vertex]] = -1

		for count in range(self.vertecis_number):
			u = self.find_min_key(keys,mstSet)
			mstSet[u] = True

			for v in range(self.vertecis_number):
				if self.edges[u][v] != -1 and mstSet[v] == False and keys[v] > self.edges[u][v]:
					keys[v] = self.edges[u][v]
					parents[v] = u

		
		return parents

	def get_patches_dict(self,patches):
		dict_patches={}

		for p in patches:
			dict_patches[p.name] = p

		return dict_patches

	def revise_GPS_from_generated_MST(self,patches,parents):
		dict_patches = self.get_patches_dict(patches)

		queue_traverse = []
		
		for v,p in enumerate(parents):
			if p == -1:
				queue_traverse = [v]
				break

		while len(queue_traverse) > 0:
			u = queue_traverse.pop()

			for v,p in enumerate(parents):
				if p == u:
					queue_traverse = [v] + queue_traverse

					patch = dict_patches[self.vertex_index_to_name_dict[v]]
					parent_patch = dict_patches[self.vertex_index_to_name_dict[p]]
					H = [n[3] for n in parent_patch.neighbors if n[0] == patch]
					H = H[0]

					patch.GPS_coords = get_new_GPS_Coords(patch,parent_patch,H)
					patch.GPS_Corrected = True

		string_corrected = get_corrected_string(patches)
		return string_corrected


def correct_GPS_MST(patches,SIFT_address,patch_folder,group_id='None'):
	patches_tmp = patches.copy()
	starting_patch = precalculate_pairwise_transformation_info_and_add_neighbors(patches_tmp,SIFT_address,group_id,patch_folder)

	G = Graph(len(patches_tmp),[p.name for p in patches_tmp])
	G.initialize_edge_weights(patches_tmp)

	parents = G.generate_MST_prim(starting_patch.name)
	return G.revise_GPS_from_generated_MST(patches_tmp,parents)

def correct_GPS_MST_helper(args):

	return correct_GPS_MST(*args)


def correct_GPS_two_queues(patches,SIFT_address,patch_folder,group_id='None'):

	starting_patch = precalculate_pairwise_transformation_info_and_add_neighbors(patches,SIFT_address,group_id,patch_folder)

	queue_1 = [starting_patch]
	queue_2 = []
	current_queue = 1

	while True:

		# print('{0}{1}'.format('*' if current_queue == 1 else '',[p.name for p in queue_1]))
		# print('{0}{1}'.format('*' if current_queue == 2 else '',[p.name for p in queue_2]))
		# print('----------------------------------------------------')

		
		if current_queue == 1:
			patch = queue_1.pop()
			# patch = pop_best_from_queue(queue_1)
		else:
			# patch = pop_best_from_queue(queue_2)
			patch = queue_2.pop()

		if current_queue == 1:
			corrected_list = get_list_of_corrected_neighbors_queue_1(patch,group_id)
		else:
			corrected_list = get_list_of_corrected_neighbors_queue_2(patch,group_id)

		queue_1 = corrected_list + queue_1

		if current_queue == 1:

			if len([n for n in patch.neighbors if not n[0].GPS_Corrected])>0:
				queue_2.insert(0,patch)
		else:
			if len(corrected_list)>0:
				current_queue = 1	

		if current_queue == 1 and len(queue_1) == 0:
			current_queue = 2

		if len(queue_2) == 0 and len(queue_1) == 0:
			break


	string_corrected = get_corrected_string(patches)
	return string_corrected

def correct_GPS_two_queues_helper(args):

	return correct_GPS_two_queues(*args)


def correct_GPS_coords_new_code_no_heap_precalculate(patches,SIFT_address,patch_folder,group_id='None'):


	patches_tmp = patches.copy()
	starting_patch = precalculate_pairwise_transformation_info_and_add_neighbors(patches_tmp,SIFT_address,group_id,patch_folder)

	# for n in patches_tmp[0].neighbors:
	# 	print(n[7])
	# 	me_from_other = [p for p in n[0].neighbors if p[0] == patches_tmp[0]]
	# 	me_from_other = me_from_other[0]
	# 	print(me_from_other[7])

	not_corrected_patches = [n[0] for n in starting_patch.neighbors]
	for p in not_corrected_patches:
		p.calculate_average_score()

	heapify(not_corrected_patches)


	while True:

		result_string = 'GROPU ID: {0} - '.format(group_id)

		sys.stdout.flush()

		if len(not_corrected_patches) == 0:
			break

		p = heappop(not_corrected_patches)

		result_string+='Patch {0}'.format(p.name)

		p2,overlap_on_p2,overlap_on_p,H,num_matches,percentage_inliers,gps_err = get_best_overlap_based_perc_inliers(p)		

		if p2 is None:
			result_string+=' <ERR: NO Connected Neighbor>'
			print(result_string)

			continue

		result_string+=' <OK: Perfect mode> --> ({0},{1}%,<{2},{3}>,{4},{5})'.format(num_matches,percentage_inliers,gps_err[0],gps_err[1],\
					calculate_overlap_area_number(overlap_on_p),p.average_score)

		
		p.GPS_coords = get_new_GPS_Coords(p,p2,H)
		p.GPS_Corrected = True

		need_heapify = False

		for p_n in p.neighbors:

			if (not p_n[0].GPS_Corrected):
				p_n[0].calculate_average_score()

				if (p_n[0] not in not_corrected_patches):
					heappush(not_corrected_patches,p_n[0])
				else:
					need_heapify = True

		heapify(not_corrected_patches)

		print(result_string)

	string_corrected = get_corrected_string(patches_tmp)
	return string_corrected

def correct_GPS_coords_new_code_no_heap_precalculate_helper(args):

	return correct_GPS_coords_new_code_no_heap_precalculate(*args)

def correct_GPS_coords_new_code(patches,show,show2,SIFT_address,group_id='None'):

	OK_GPS_ERR_AVG = (0,0)
	OK_GPS_N = 0

	patches_tmp = patches.copy()
	not_corrected_patches = []

	for p in patches_tmp:
		overlaps = [p_n for p_n in patches_tmp if p_n.name != p.name and (p.has_overlap(p_n) or p_n.has_overlap(p))]
		p.overlaps = overlaps
		p.claculate_area_score()

		if p.GPS_Corrected:
			p.load_SIFT(SIFT_address)
			not_corrected_patches = not_corrected_patches+overlaps
			for p_tmp in overlaps:
				p_tmp.load_SIFT(SIFT_address)

	heapify(not_corrected_patches)

	number_of_iterations_without_change = 0

	while True:

		result_string = 'GROPU ID: {0} - '.format(group_id)
		# random.shuffle(not_corrected_patches)

		sys.stdout.flush()

		# not_corrected_patches = [p for p in patches_tmp if p.GPS_Corrected == False]
		if len(not_corrected_patches) == 0:
			break

		p = heappop(not_corrected_patches)

		if number_of_iterations_without_change > 3*len(not_corrected_patches):
			break

		result_string+='Patch {0}'.format(p.name)

		overlaps = [p_n for p_n in p.overlaps if p_n.GPS_Corrected]

		if len(overlaps) == 0:
			result_string+=' <ERR: No overlaps>'
			print(result_string)
			p.area_score=0
			heappush(not_corrected_patches,p)
			number_of_iterations_without_change+=1

			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			result_string+=' <ERR: Empty overlap>'
			print(result_string)
			p.area_score=0
			heappush(not_corrected_patches,p)
			number_of_iterations_without_change+=1
			continue

		# kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show)
		# kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show)

		kp1,desc1 = choose_SIFT_key_points(p,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],SIFT_address)
		kp2,desc2 = choose_SIFT_key_points(p2,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],SIFT_address)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

		percentage_inliers = round(percentage_inliers*100,2)

		if number_of_iterations_without_change <= len(not_corrected_patches)+1 and (percentage_inliers<=10 or len(matches) < 40):
			result_string+=' <ERR: Low inliers> --> ({0},{1}%)'.format(len(matches),percentage_inliers)
			print(result_string)
			number_of_iterations_without_change+=1
			p.area_score=0
			heappush(not_corrected_patches,p)
			# visualize_single_run(H,p,p2,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],SIFT_address)
			continue

		gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			result_string+=' <ERR: High GPS error> --> ({0},{1}%,<{2},{3}>)'.format(len(matches),percentage_inliers,gps_err[0],gps_err[1])
			print(result_string)
			number_of_iterations_without_change+=1
			p.area_score=0
			heappush(not_corrected_patches,p)
			# visualize_single_run(H,p,p2,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],SIFT_address)
			continue
			
		if show2:
			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

		if number_of_iterations_without_change > len(not_corrected_patches)+1:
			result_string+=' <OK: Relaxed mode> --> ({0},{1}%)'.format(len(matches),percentage_inliers)
		else:
			result_string+=' <OK: Perfect mode> --> ({0},{1}%)'.format(len(matches),percentage_inliers)
			OK_GPS_N+=1
			OK_GPS_ERR_AVG=(OK_GPS_ERR_AVG[0]+gps_err[0],OK_GPS_ERR_AVG[1]+gps_err[1])

		p.GPS_coords = get_new_GPS_Coords(p,p2,H)
		p.GPS_Corrected = True
		
		# not_corrected_patches = list(set(not_corrected_patches + [p_n for p_n in p.overlaps if not p_n.GPS_Corrected]))

		heapify_flag = False

		for p_n in p.overlaps:

			if (not p_n.GPS_Corrected):
				if (p_n not in not_corrected_patches):
					p_n.load_SIFT(SIFT_address)
					p_n.claculate_area_score()
					heappush(not_corrected_patches,p_n)
				else:
					p_n.claculate_area_score()
					heapify_flag = True

			if p_n.GPS_Corrected and len([p_c for p_c in p_n.overlaps if not p_c.GPS_Corrected]) == 0:
				p_n.del_SIFT()

		if heapify_flag:
			heapify(not_corrected_patches)

		print(result_string)

		number_of_iterations_without_change = 0

		if show2:
			ov_2_on_1 = p.get_overlap_rectangle(p2)
			ov_1_on_2 = p2.get_overlap_rectangle(p)

			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)


	print('******* GROUP ID: {0} - Entering FULL Relax Mode ********'.format(group_id))

	while True:

		result_string = 'GROPU ID: {0} - '.format(group_id)
		# random.shuffle(not_corrected_patches)

		sys.stdout.flush()

		# not_corrected_patches = [p for p in patches_tmp if p.GPS_Corrected == False]
		if len(not_corrected_patches) == 0 or number_of_iterations_without_change > len(not_corrected_patches)+1:
			break

		p = heappop(not_corrected_patches)

		result_string+='Patch {0}'.format(p.name)

		overlaps = [p_n for p_n in p.overlaps if p_n.GPS_Corrected]

		if len(overlaps) == 0:
			result_string+=' <ERR: No overlaps>'
			print(result_string)
			p.area_score=0
			heappush(not_corrected_patches,p)
			number_of_iterations_without_change+=1
			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			result_string+=' <ERR: Empty overlap>'
			print(result_string)
			p.area_score=0
			heappush(not_corrected_patches,p)
			number_of_iterations_without_change+=1
			continue

		# kp1,desc1 = detect_SIFT_key_points(p.img,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],1,show)
		# kp2,desc2 = detect_SIFT_key_points(p2.img,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],2,show)

		kp1,desc1 = choose_SIFT_key_points(p,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],SIFT_address)
		kp2,desc2 = choose_SIFT_key_points(p2,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],SIFT_address)

		matches = get_good_matches(desc2,desc1)

		H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

		percentage_inliers = round(percentage_inliers*100,2)

		gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
		
		if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
			H = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result_string+=' <OK: GPS ONLY FULL Relax mode> --> ({0},{1}%,<{2},{3}>)'.format(len(matches),percentage_inliers,gps_err[0],gps_err[1])
		else:
			result_string+=' <OK: FULL Relax mode> --> ({0},{1}%)'.format(len(matches),percentage_inliers,gps_err[0],gps_err[1])

		if show2:
			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

		p.GPS_coords = get_new_GPS_Coords(p,p2,H)
		p.GPS_Corrected = True
		
		# not_corrected_patches = list(set(not_corrected_patches + [p_n for p_n in p.overlaps if not p_n.GPS_Corrected]))

		heapify_flag = False

		for p_n in p.overlaps:

			if (not p_n.GPS_Corrected):
				if (p_n not in not_corrected_patches):
					p_n.load_SIFT(SIFT_address)
					p_n.claculate_area_score()
					heappush(not_corrected_patches,p_n)
				else:
					p_n.claculate_area_score()
					heapify_flag = True

			if p_n.GPS_Corrected and len([p_c for p_c in p_n.overlaps if not p_c.GPS_Corrected]) == 0:
				p_n.del_SIFT()

		if heapify_flag:
			heapify(not_corrected_patches)

		print(result_string)

		number_of_iterations_without_change = 0

		if show2:
			ov_2_on_1 = p.get_overlap_rectangle(p2)
			ov_1_on_2 = p2.get_overlap_rectangle(p)

			G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
			result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

	print('GPS ERROR: ({0},{1})'.format(OK_GPS_ERR_AVG[0]/OK_GPS_N,OK_GPS_ERR_AVG[1]/OK_GPS_N))

	return patches_tmp

def correct_GPS_coords_new_code_helper(args):

	return correct_GPS_coords_new_code(*args)

def correct_GPS_parallel_inner_function(p,SIFT_address,show2,relaxed_mode):

	overlaps = [p_n for p_n in p.overlaps if p_n.GPS_Corrected]

	if len(overlaps) == 0:
		print('Type1 >>> No corrected overlaps for {0}. push back...'.format(p.name))
		sys.stdout.flush()
		return [p]

	p2,f_grbg = get_best_overlap(p,overlaps)
	
	ov_2_on_1 = p.get_overlap_rectangle(p2)
	ov_1_on_2 = p2.get_overlap_rectangle(p)
	
	if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
	or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
		print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
		sys.stdout.flush()
		return [p]

	kp1,desc1 = choose_SIFT_key_points(p,ov_2_on_1[0],ov_2_on_1[1],ov_2_on_1[2],ov_2_on_1[3],SIFT_address)
	kp2,desc2 = choose_SIFT_key_points(p2,ov_1_on_2[0],ov_1_on_2[1],ov_1_on_2[2],ov_1_on_2[3],SIFT_address)

	matches = get_good_matches(desc2,desc1)

	H,percentage_inliers = find_homography(matches,kp2,kp1,ov_2_on_1,ov_1_on_2,False)

	print('----Number of matches: {0}\n\tPercentage Iliers: {1}'.format(len(matches),percentage_inliers))
	sys.stdout.flush()

	if not relaxed_mode and (percentage_inliers<=0.1 or len(matches) < 40):
		print('\t*** Not enough inliers ...')
		sys.stdout.flush()
		return [p]

	gps_err = Get_GPS_Error(H,ov_1_on_2,ov_2_on_1)
	
	if gps_err[0] > (ov_1_on_2[2]-ov_1_on_2[0])/2 and gps_err[1] > (ov_1_on_2[3]-ov_1_on_2[1])/2:
		if relaxed_mode:
			H = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
		else:
			print('*** High GPS error ...')
			sys.stdout.flush()
			return [p]
		

	if show2:
		G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)

	p.GPS_coords = get_new_GPS_Coords(p,p2,H)
	p.GPS_Corrected = True

	print('GPC corrected for {0}.'.format(p.name))
	sys.stdout.flush()

	if show2:
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)

		G = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)
		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,G,ov_1_on_2,show2)
	
	return [p_n for p_n in p.overlaps if not p_n.GPS_Corrected]

def correct_GPS_parallel_inner_function_helper(args):
	return correct_GPS_parallel_inner_function(*args)

def correct_GPS_coords_new_code_parallel(patches,show,show2,SIFT_address):

	patches_tmp = patches.copy()

	not_corrected_patches =[]
	
	best_f = 0
	best_p = patches_tmp[0]

	for p in patches_tmp:
		overlaps = [p_n for p_n in patches_tmp if p_n!=p and (p.has_overlap(p_n) or p_n.has_overlap(p))]
		p.overlaps = overlaps

		f = len(overlaps)

		if f>best_f:
			best_f = f
			best_p = p

	best_p.GPS_Corrected = True
	not_corrected_patches = not_corrected_patches+best_p.overlaps

	number_of_iterations_without_change = 0
	max_nodes = 4
	relaxed_mode = False

	while True:
		print(len([p.name for p in not_corrected_patches]))
		# random.shuffle(not_corrected_patches)

		# not_corrected_patches = [p for p in patches_tmp if p.GPS_Corrected == False]
		if len(not_corrected_patches) == 0 :
			break

		pool_size = min(max_nodes,len(not_corrected_patches))

		if number_of_iterations_without_change>len(not_corrected_patches)/pool_size:
			relaxed_mode = True

		processes = multiprocessing.Pool(pool_size)

		args = []
		
		for i in range(0,pool_size):
			args.append((not_corrected_patches.pop(),SIFT_address,show2,relaxed_mode))
		
		results = processes.map(correct_GPS_parallel_inner_function_helper,args)

		number_of_iterations_without_change+=1

		for r in results:
			if len(r)>1:
				number_of_iterations_without_change = 0

			for it in r:
				if it.name not in [p.name for p in not_corrected_patches]:
					not_corrected_patches.insert(0,it)

		# not_corrected_patches = list(set(not_corrected_patches+results))

		processes.close()


		

	return patches_tmp

def stitch_based_on_corrected_GPS(patches,show):
	patches_tmp = patches.copy()

	i = 0

	len_no_change = 0

	while len(patches_tmp)>1 and len_no_change<len(patches_tmp):
		p = patches_tmp.pop()
		len_no_change+=1

		# p = choose_patch_with_largest_overlap(patches_tmp)
		# patches_tmp.remove(p)

		overlaps = [p_n for p_n in patches_tmp if (p.has_overlap(p_n) or p_n.has_overlap(p))]

		if len(overlaps) == 0:
			print('Type1 >>> No overlaps for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			continue

		p2,f_grbg = get_best_overlap(p,overlaps)
		
		ov_2_on_1 = p.get_overlap_rectangle(p2)
		ov_1_on_2 = p2.get_overlap_rectangle(p)
		
		if (ov_2_on_1[0] == 0 and ov_2_on_1[1] == 0 and ov_2_on_1[2] == 0 and ov_2_on_1[3] == 0)\
		or (ov_1_on_2[0] == 0 and ov_1_on_2[1] == 0 and ov_1_on_2[2] == 0 and ov_1_on_2[3] == 0):
			print('Type2 >>> Empty overlap for {0}. push back...'.format(p.name))
			patches_tmp.insert(0,p)
			
			continue

		patches_tmp.remove(p2)
		
		H = find_homography_gps_only(ov_2_on_1,ov_1_on_2,p,p2)

		result, MSE = stitch(p.rgb_img,p2.rgb_img,p.img,p2.img,H,ov_1_on_2,show)

		len_no_change = 0
		
		stitched_coords = find_stitched_coords(p.GPS_coords,p2.GPS_coords)

		new_patch = Patch('New_{0}'.format(i),result,convert_to_gray(result),stitched_coords)

		patches_tmp.append(new_patch)

		del p
		del p2

		i+=1

	return patches_tmp

def show_and_save_final_patches(patches):
	for p in patches:
		# cv2.imwrite('{0}.jpg'.format(p.name),p.rgb_img)
		ratio = p.rgb_img.shape[0]/p.rgb_img.shape[1]
		img_res = cv2.resize(p.rgb_img, (700, int(700*ratio))) 
		cv2.imshow('fig',img_res)
		cv2.waitKey(0)	

def save_coordinates(final_patches,filename):
	
	final_results = 'Filename,Upper left,Lower left,Upper right,Lower right,Center\n'

	for p in final_patches:
		p.GPS_coords.UL_coord = (round(p.GPS_coords.UL_coord[0],7),round(p.GPS_coords.UL_coord[1],7))
		p.GPS_coords.LL_coord = (round(p.GPS_coords.LL_coord[0],7),round(p.GPS_coords.LL_coord[1],7))
		p.GPS_coords.UR_coord = (round(p.GPS_coords.UR_coord[0],7),round(p.GPS_coords.UR_coord[1],7))
		p.GPS_coords.LR_coord = (round(p.GPS_coords.LR_coord[0],7),round(p.GPS_coords.LR_coord[1],7))
		p.GPS_coords.Center = (round(p.GPS_coords.Center[0],7),round(p.GPS_coords.Center[1],7))

		final_results += '{:s},"{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}","{:.7f},{:.7f}"\n'\
		.format(p.name,p.GPS_coords.UL_coord[0],p.GPS_coords.UL_coord[1],p.GPS_coords.LL_coord[0],p.GPS_coords.LL_coord[1],p.GPS_coords.UR_coord[0],p.GPS_coords.UR_coord[1]\
			,p.GPS_coords.LR_coord[0],p.GPS_coords.LR_coord[1],p.GPS_coords.Center[0],p.GPS_coords.Center[1])

	final_results = final_results.replace('(','"').replace(')','"')

	with open(filename,'w') as f:
		f.write(final_results)

def save_coordinates_from_string(results,filename):
	
	final_results = 'Filename,Upper left,Lower left,Upper right,Lower right,Center\n'

	for r in results:
		final_results+=r

	with open(filename,'w') as f:
		f.write(final_results)

def correct_GPS_two_queues_groups(groups,SIFT_address,patch_folder):
	global no_of_cores_to_use

	list_args = []

	for g in groups:
		list_args.append((groups[g],SIFT_address,patch_folder,g))

	processes = multiprocessing.Pool(min(len(groups),no_of_cores_to_use))

	results = processes.map(correct_GPS_two_queues_helper,list_args)
	processes.close()

	return results

def correct_GPS_new_code_groups(groups,show,show2,SIFT_address):
	global no_of_cores_to_use

	list_args = []

	for g in groups:
		list_args.append((groups[g],show,show2,SIFT_address,g))

	processes = multiprocessing.Pool(min(len(groups),no_of_cores_to_use))

	list_final_patches = processes.map(correct_GPS_coords_new_code_helper,list_args)
	processes.close()

	patches = []

	for p_list in list_final_patches:
		patches += p_list

	return patches

def correct_GPS_new_code_no_heap_precalculate_groups(groups,SIFT_address,patch_folder):
	global no_of_cores_to_use

	list_args = []

	for g in groups:
		list_args.append((groups[g],SIFT_address,patch_folder,g))

	processes = multiprocessing.Pool(min(len(groups),no_of_cores_to_use))

	results = processes.map(correct_GPS_coords_new_code_no_heap_precalculate_helper,list_args)
	processes.close()

	return results

def correct_GPS_MST_groups(groups,SIFT_address,patch_folder):
	global no_of_cores_to_use

	list_args = []

	for g in groups:
		list_args.append((groups[g],SIFT_address,patch_folder,g))

	processes = multiprocessing.Pool(min(len(groups),no_of_cores_to_use))

	results = processes.map(correct_GPS_MST_helper,list_args)
	processes.close()

	return results


def fit_circle(xs,ys):

	us = xs - np.mean(xs)
	vs = ys - np.mean(ys)

	A1 = np.sum(us**2)
	B1 = np.sum(us*vs)
	C1 = 0.5*(np.sum(us**3)+np.sum(us*(vs**2)))
	A2 = B1
	B2 = np.sum(vs**2)
	C2 = 0.5*(np.sum(vs**3)+np.sum(vs*(us**2)))

	v = (A1*C2 - A2*C1)/(A1*B2 - A2*B1)
	u = (C1-B1*v)/A1

	r = int(math.sqrt(u**2+v**2+(A1+B2)/np.shape(xs)[0]))

	x = int(u+np.mean(xs))
	y = int(v+np.mean(ys))

	return x,y,r

def circle_error(x,y,r,xs,ys):
	err = 0

	for i in range(0,np.shape(xs)[0]):
		d = math.sqrt((x-xs[i])**2+(y-ys[i])**2)
		if d>2*r:
			err+=0
		elif d<r/2:
			err+=0
		else:
			err += abs(d - r)

	return err

def ransac(xs,ys,iterations,number_of_points):
	best_x = -1
	best_y = -1
	best_r = -1
	min_error = None

	for i in range(0,iterations):
		
		indexes = random.sample(range(0,np.shape(xs)[0]),number_of_points)
	
		xs_r = xs[indexes]
		ys_r = ys[indexes]

		x,y,r = fit_circle(xs_r,ys_r)
		err = circle_error(x,y,r,xs,ys)

		if min_error == None or min_error>err:
			min_error = err
			best_x = x
			best_y = y
			best_r = r

	return best_x,best_y,best_r

def get_unique_lists(xs,ys):
	tmp, ind1 = np.unique(xs,return_index=True)
	tmp, ind2 = np.unique(ys,return_index=True)

	ind = np.intersect1d(ind1,ind2)

	return xs[ind],ys[ind]

def get_lid_in_patch(address,img_name,ransac_iter=100,ransac_min_num_fit=10):
	
	img = cv2.imread('{0}/{1}'.format(address,img_name))
	
	img[:,:,1:3] = 0

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	img = adjust_gamma(img,2.5)
	
	max_intensity = np.amax(img)
	
	t = max_intensity-2
	
	(thresh, img) = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
	kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)	

	kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	


	shp = np.shape(img)

	img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	new_contours = []

	for c in contours:
		for p in c:
			new_contours.append([p[0][0],p[0][1]])
	
	new_contours = np.array(new_contours)

	if np.shape(new_contours)[0]<ransac_min_num_fit:
		return -1,-1,-1

	xs = np.array(new_contours[:,0])
	ys = np.array(new_contours[:,1])

	xs,ys = get_unique_lists(xs,ys)

	if np.shape(xs)[0]<ransac_min_num_fit:
		return -1,-1,-1

	x,y,r = ransac(xs,ys,100,10)
	
	if x >= 0 and x < shp[1] and y >= 0 and y < shp[0] and r >= 400 and r <= 500:
		return x,y,r
	else:
		return -1,-1,-1



def get_lids(address):
	lids = {}

	with open(address) as f:
		lines = f.read()

		for l in lines.split('\n'):
			if l == '':
				break

			features = l.split(',')

			marker = features[0]
			lat = features[1]
			lon = features[2]

			lids[marker] = (float(lon),float(lat))

	return lids

def get_name_of_patches_with_lids(address,lids):
	
	patches_names_with_lid = []

	with open(address) as f:
		lines = f.read()
		lines = lines.replace('"','')

		for l in lines.split('\n'):
			if l == '':
				break
			if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
				continue

			features = l.split(',')

			filename = features[0]
			upper_left = (float(features[1]),float(features[2]))
			lower_left = (float(features[3]),float(features[4]))
			upper_right = (float(features[5]),float(features[6]))
			lower_right = (float(features[7]),float(features[8]))
			center = (float(features[9]),float(features[10]))

			coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)
			
			for l in lids:
				if coord.is_coord_inside(lids[l]):
					patches_names_with_lid.append((l,filename,coord))

	return patches_names_with_lid

def create_lid_patch(patches_folder,p_name,coord,lids,l_marker):
	x,y,r = get_lid_in_patch(patches_folder,p_name)

	if x==-1 and y==-1 and r==-1:
		return None,None

	p = Patch_2(p_name,None,None,coord,(-1,-1))
	p.load_img(patches_folder)

	# p.visualize_with_single_GPS_point(lids[l_marker],(x,y),r)
	p.correct_GPS_based_on_point((x,y),lids[l_marker])
	# p.visualize_with_single_GPS_point(lids[l_marker],(x,y),r)

	p.GPS_Corrected = True

	return p,l_marker

def create_lid_patch_helper(args):

	return create_lid_patch(*args)

def get_groups_and_patches_with_lids(patches_folder,coordinate_address,SIFT_address,lids):
	global no_of_cores_to_use

	possible_patches_with_lids = get_name_of_patches_with_lids(coordinate_address,lids)
	list_all_groups = {}
	assigned_patches_names = []


	args_list = []

	for l_marker,p_name,coord in possible_patches_with_lids:
		args_list.append((patches_folder,p_name,coord,lids,l_marker))

	processes = multiprocessing.Pool(no_of_cores_to_use)

	lid_patches_with_l = processes.map(create_lid_patch_helper,args_list)
	processes.close()

	for p,l in lid_patches_with_l:
		if l == None:
			continue

		if (l in list_all_groups.keys()) and len(list_all_groups[l]) > 0:
			assigned_patches_names.remove(list_all_groups[l][0].name)
			
		assigned_patches_names.append(p.name)
		list_all_groups[l] = [p]

	new_lids = {}

	for l in list_all_groups:
		new_lids[l] = lids[l]

	
	patches = read_all_data_on_server(patches_folder,coordinate_address,SIFT_address,False)

	for p in patches:
		if p.name not in assigned_patches_names:
			list_all_groups[get_nearest_lid_patch(new_lids,p)].append(p)

	print('Detected {0} groups as follow:'.format(len(list_all_groups.values())))

	for g in list_all_groups:
		print('Group {0} with {1} images.'.format(g,len(list_all_groups[g])))

	# save_group_data(list_all_groups,new_lids,len(patches),'/data/plant/full_scans/2020-01-08-rgb/plt.npy')

	return list_all_groups



def GPS_based_distance(coord1,coord2):

	return math.sqrt((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2)

def get_nearest_lid_coord(lids,coord):
	nearest = -1
	min_distance = sys.maxsize

	for lid in lids:
		d = GPS_based_distance(lids[lid],coord)
		if d<min_distance:
			nearest = lid
			min_distance = d

	return nearest

def get_nearest_lid_patch(lids,patch):
	
	nearestes = []

	nearestes.append(get_nearest_lid_coord(lids,patch.GPS_coords.UR_coord))
	nearestes.append(get_nearest_lid_coord(lids,patch.GPS_coords.UL_coord))
	nearestes.append(get_nearest_lid_coord(lids,patch.GPS_coords.LR_coord))
	nearestes.append(get_nearest_lid_coord(lids,patch.GPS_coords.LL_coord))
	nearestes.append(get_nearest_lid_coord(lids,patch.GPS_coords.Center))

	res = [l for l in lids if l == max(set(nearestes), key = nearestes.count)]
	return res[0]

def group_images_by_nearest_lid(lids,patches):
	
	list_all_groups = {}

	for l in lids:
		list_all_groups[l] = []

	for p in patches:
		list_all_groups[get_nearest_lid_patch(lids,p)].append(p)

	return list_all_groups

def plot_groups(address):
	import matplotlib.pyplot as plt

	data = np.load(address)
	plt.axis('equal')

	lids_data = []

	for d in data:
		if d[5]==1:
			lids_data.append(d)
		else:
			plt.scatter(d[0],d[1],color=(d[2],d[3],d[4]))
			
	for d in lids_data:
		plt.scatter(d[0],d[1],color=(0,0,0),marker='x',s=30)
		
	plt.show()

def save_group_data(groups,lids,n,address):
	
	data = np.zeros((n,6))
	i = 0

	not_used_colors = [(26,188,156),(46,204,113),(52,152,219),(155,89,182),(52,73,94),(22,160,133),(39,174,96),(41,128,185),(142,68,173),(44,62,80)\
	,(241,196,15),(230,126,34),(231,76,60),(236,240,241),(149,165,166),(243,156,18),(211,84,0),(192,57,43),(189,195,199),(127,140,141)]

	colors = {}
	for g in groups:
		c = not_used_colors.pop()
		colors[g] = (float(c[0]/255.0),float(c[1]/255.0),float(c[2]/255.0))

	for g in groups:
		for p in groups[g]:
			data[i,0] = p.GPS_coords.Center[0]*1000 
			data[i,1] = p.GPS_coords.Center[1]*1000
			data[i,2] = colors[g][0]
			data[i,3] = colors[g][1]
			data[i,4] = colors[g][2]

			if p.GPS_coords.is_coord_inside(lids[g]):
				data[i,5] = 1
				# print(p.name)

			else:
				data[i,5] = 0
			i+=1
	
	np.save(address,data)



def stitch_based_on_corrected_GPS_helper(args):
	for p in args[0]:
		p.load_img(args[3])

	stitched = stitch_based_on_corrected_GPS(args[0],args[1])

	if len(stitched)==1:
		cv2.imwrite('{0}/row_{1}.jpg'.format(args[2],args[4]),stitched[0].rgb_img)
		print('Saved for row {0}.'.format(args[4]))
	else:
		print('Error for row {0}. Number of stitched images: {1}.'.format(args[4],len(stitched)))

def recalculate_keypoints_locations(p,SIFT_folder,x_difference,y_difference):
	upper_kp = []
	upper_desc = []
	lower_kp = []
	lower_desc = []

	(kp_tmp,desc_tmp) = pickle.load(open('{0}/{1}_SIFT.data'.format(SIFT_folder,p.name.replace('.tif','')), "rb"))

	for i,k in enumerate(kp_tmp):

		if k[1]<p.size[0]/5:
			# calculate new locations
			upper_kp.append((k[0]+x_difference,k[1]+y_difference))
			upper_desc.append(list(np.array(desc_tmp[i,:])))
		elif k[1]>=p.size[0]*4/5:
			# calculate new locations
			lower_kp.append((k[0]+x_difference,k[1]+y_difference))
			lower_desc.append(list(np.array(desc_tmp[i,:])))

	upper_desc = np.array(upper_desc)
	lower_desc = np.array(lower_desc)

	# print('patch {0} done...'.format(p.name))
	return upper_kp,upper_desc,lower_kp,lower_desc



def get_good_matches_for_horizontal(desc1,desc2,kp1,kp2,patch_size):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		point_1 = kp1[m[0].queryIdx]
		point_2 = kp2[m[0].trainIdx]

		if 0 <= abs(point_1[1]-point_2[1]) <= patch_size[0]/20 and\
		0 <= abs(point_1[0]-point_2[0]) <= patch_size[1] and\
		m[0].distance < 0.8*m[1].distance:
			good.append(m)

	matches = np.asarray(good)

	return matches

def draw_matches(p1,p2,kp1,kp2,matches,j):
	
	result = np.zeros((p1.size[0],p1.size[1]*2,3))
	result[:,0:p1.size[1],:] = p1.rgb_img
	result[:,p1.size[1]:p1.size[1]*2,:] = p2.rgb_img

	i=0
	for m in matches[:,0]:
		if i>100:
			break
		point_1 = kp1[m.queryIdx]
		point_2 = kp2[m.trainIdx]
		point_1 = (int(point_1[0]),int(point_1[1]))
		point_2 = (int(point_2[0])+p1.size[1],int(point_2[1]))
		cv2.line(result,point_1,point_2,(0,0,255),2)
		cv2.circle(result,point_1,10,(0,255,0),thickness=-1)
		cv2.circle(result,point_2,10,(255,0,0),thickness=-1)
		i+=1

	result = cv2.resize(result,(int(result.shape[1]/10),int(result.shape[0]/10)))
	cv2.imwrite('matches_{0}.bmp'.format(j),result)

def correct_horizontal_neighbors(p1,p2,SIFT_address,patch_folder,i):
	overlap1 = p1.get_overlap_rectangle(p2)
	overlap2 = p2.get_overlap_rectangle(p1)
	
	kp1,desc1 = choose_SIFT_key_points(p1,overlap1[0],overlap1[1],overlap1[2],overlap1[3],SIFT_address)
	kp2,desc2 = choose_SIFT_key_points(p2,overlap2[0],overlap2[1],overlap2[2],overlap2[3],SIFT_address)

	matches = get_good_matches_for_horizontal(desc2,desc1,kp2,kp1,p1.size)
	
	# p1.load_img(patch_folder)
	# p2.load_img(patch_folder)
	# draw_matches(p2,p1,kp2,kp1,matches,i)

	if len(matches)<3:
		return None

	H,percentage_inliers = find_homography(matches,kp2,kp1,overlap1,overlap2,False)
	# print(round(percentage_inliers,2))

	# result,mse = stitch(p1.rgb_img,p2.rgb_img,p1.img,p2.img,H,overlap1)
	# result = cv2.resize(result,(int(result.shape[1]/5),int(result.shape[0]/5)))
	# cv2.imwrite('matches_{0}.bmp'.format(i),result)

	coord = get_new_GPS_Coords(p1,p2,H)
	if p1.GPS_coords.Center[1]-coord.Center[1]>abs(p1.GPS_coords.UL_coord[1]-p1.GPS_coords.LL_coord[1])/20:
		return None

	return coord

def get_top_n_good_matches(desc1,desc2,kp1,kp2,n,size_patch):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)

	good = []
	for m in matches:
		point_1 = kp1[m[0].queryIdx]
		point_2 = kp2[m[0].trainIdx]

		if size_patch[0] >= abs(point_1[1]-point_2[1]) >= (9/10)*size_patch[0] and \
		(1/5)*size_patch[1] >= abs(point_1[0]-point_2[0]) >= 0 and \
		m[0].distance < 0.8*m[1].distance:
			good.append(m)

	sorted_matches = sorted(good, key=lambda x: x[0].distance)

	good = []

	if len(sorted_matches)>n:
		good += sorted_matches[0:n]
	else:
		good += sorted_matches

	matches = np.asarray(good)

	return matches

def calculate_homography_for_super_patches(kp,desc,prev_kp,prev_desc,matches):

	src_list = []
	dst_list = []

	for i,mtch in enumerate(matches):
		src_list += [kp[i][m.queryIdx] for m in mtch[:,0]]
		dst_list += [prev_kp[i][m.trainIdx] for m in mtch[:,0]]

	src = np.float32(src_list).reshape(-1,1,2)
	dst = np.float32(dst_list).reshape(-1,1,2)
	
	H, masked = cv2.estimateAffinePartial2D(dst, src, maxIters = 9000, confidence = 0.999, refineIters = 15)
	
	H = np.append(H,np.array([[0,0,1]]),axis=0)
	H[0:2,0:2] = np.array([[1,0],[0,1]])
	print(H)
	return H

class SuperPatch():

	def __init__(self,row_n,list_patches,gps_coords,SIFT_folder):
		self.row_number = row_n
		self.patches = list_patches
		for p in self.patches:
			(kp_tmp,desc_tmp) = pickle.load(open('{0}/{1}_SIFT.data'.format(SIFT_folder,p.name.replace('.tif','')), "rb"))
			p.SIFT_kp_locations = kp_tmp
			p.SIFT_kp_desc = desc_tmp

		self.GPS_coords = gps_coords
		self.x_ratio_GPS_over_pixel = (list_patches[0].GPS_coords.UR_coord[0] - list_patches[0].GPS_coords.UL_coord[0])/list_patches[0].size[1]
		self.y_ratio_GPS_over_pixel = (list_patches[0].GPS_coords.UL_coord[1] - list_patches[0].GPS_coords.LL_coord[1])/list_patches[0].size[0]

		self.size = (int((self.GPS_coords.UL_coord[1]-self.GPS_coords.LL_coord[1])/self.y_ratio_GPS_over_pixel),\
			int((self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0])/self.x_ratio_GPS_over_pixel))

		# self.upper_kp, self.upper_desc, self.lower_kp, self.lower_desc = self.calculate_super_sift_points(SIFT_folder)
		# self.remove_randomly()

	def draw_super_patch(self,patch_folder,name_of):
		
		result = np.zeros((self.size[0]+40,self.size[1]+40,3), np.uint8)

		for p in self.patches:
			p.load_img(patch_folder)
			x_diff = p.GPS_coords.UL_coord[0] - self.GPS_coords.UL_coord[0]
			y_diff = self.GPS_coords.UL_coord[1] - p.GPS_coords.UL_coord[1]
			st_x = int(x_diff/self.x_ratio_GPS_over_pixel)
			st_y = int(y_diff/self.y_ratio_GPS_over_pixel)
			print(st_x,st_y)
			# print('.')
			try:
				result[st_y:st_y+p.size[0],st_x:st_x+p.size[1],:] = p.rgb_img
			except Exception as e:
				print(e)
			# cv2.rectangle(result,(st_x+1,st_y+1),(st_x+p.size[1]-1,st_y+p.size[0]-1),(0,0,255),20)
			p.del_img()

		result = cv2.resize(result,(int(result.shape[1]/10),int(result.shape[0]/10)))
		cv2.imwrite('rows_{0}.bmp'.format(name_of),result)

	def draw_super_patch_and_lines(self,matches,kp1,kp2,patch_folder,name_of):

		result = np.zeros((self.size[0]+40,self.size[1]+40,3), np.uint8)

		for p in self.patches:
			p.load_img(patch_folder)
			x_diff = p.GPS_coords.UL_coord[0] - self.GPS_coords.UL_coord[0]
			y_diff = self.GPS_coords.UL_coord[1] - p.GPS_coords.UL_coord[1]
			st_x = int(x_diff/self.x_ratio_GPS_over_pixel)
			st_y = int(y_diff/self.y_ratio_GPS_over_pixel)
			result[st_y:st_y+p.size[0],st_x:st_x+p.size[1],:] = p.rgb_img

			for m in matches[:,0]:
			
				point_1 = kp1[m.queryIdx]
				point_2 = kp2[m.trainIdx]
				point_1 = (int(point_1[0]),int(point_1[1]))
				point_2 = (int(point_2[0]),int(point_2[1]))
				cv2.line(result,point_1,point_2,(0,0,255),2)

			p.del_img()

		result = cv2.resize(result,(int(result.shape[1]/10),int(result.shape[0]/10)))
		cv2.imwrite('rows_{0}.bmp'.format(name_of),result)

	def recalculate_size_and_coords(self):
		up = self.patches[0].GPS_coords.UL_coord[1]
		down = self.patches[0].GPS_coords.LL_coord[1]
		left = self.patches[0].GPS_coords.UL_coord[0]
		right = self.patches[0].GPS_coords.UR_coord[0]

		for p in self.patches:
			if p.GPS_coords.UL_coord[1]>=up:
				up=p.GPS_coords.UL_coord[1]

			if p.GPS_coords.LL_coord[1]<=down:
				down=p.GPS_coords.LL_coord[1]

			if p.GPS_coords.UL_coord[0]<=left:
				left=p.GPS_coords.UL_coord[0]

			if p.GPS_coords.UR_coord[0]>=right:
				right=p.GPS_coords.UR_coord[0]

		UL_coord = (left,up)
		UR_coord = (right,up)
		LL_coord = (left,down)
		LR_coord = (right,down)
		Center = ((left+right)/2,(down+up)/2)

		coord = Patch_GPS_coordinate(UL_coord,UR_coord,LL_coord,LR_coord,Center)
		self.GPS_coords = coord
		self.size = (int((self.GPS_coords.UL_coord[1]-self.GPS_coords.LL_coord[1])/self.y_ratio_GPS_over_pixel),\
			int((self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0])/self.x_ratio_GPS_over_pixel))


	def correct_supper_patch_internally(self,SIFT_address,patch_folder):
		prev_patch = None

		for i,p in enumerate(self.patches):
			
			if prev_patch is None:
				prev_patch = p
				continue

			coord = correct_horizontal_neighbors(p,prev_patch,SIFT_address,patch_folder,i)

			if coord is not None:
				p.GPS_coords = coord

			prev_patch = p

		self.recalculate_size_and_coords()
	
	def correct_all_patches_and_self_by_H(self,H,prev_super_patch):

		new_coords = get_new_GPS_Coords(self,prev_super_patch,H)
		diff_x = (self.GPS_coords.UL_coord[0]-new_coords.UL_coord[0])
		diff_y = (self.GPS_coords.UL_coord[1]-new_coords.UL_coord[1])
		
		for p in self.patches:
			new_UL = (p.GPS_coords.UL_coord[0]-diff_x,p.GPS_coords.UL_coord[1]-diff_y)
			new_UR = (p.GPS_coords.UR_coord[0]-diff_x,p.GPS_coords.UR_coord[1]-diff_y)
			new_LL = (p.GPS_coords.LL_coord[0]-diff_x,p.GPS_coords.LL_coord[1]-diff_y)
			new_LR = (p.GPS_coords.LR_coord[0]-diff_x,p.GPS_coords.LR_coord[1]-diff_y)
			new_center = (p.GPS_coords.Center[0]-diff_x,p.GPS_coords.Center[1]-diff_y)

			new_p_coord = Patch_GPS_coordinate(new_UL,new_UR,new_LL,new_LR,new_center)
			p.GPS_coords = new_p_coord

		self.recalculate_size_and_coords()

		

	def correct_whole_based_on_super_patch(self,prev_super_patch,SIFT_folder,patch_folder):

		# matches = []
		# kp = []
		# desc = []
		# prev_kp = []
		# prev_desc = []

		# for inner_p in self.patches:
			
		# 	for prev_inner_p in prev_super_patch.patches:
		# 		if inner_p.has_overlap(prev_inner_p) or prev_inner_p.has_overlap(inner_p):
		# 			overlap1 = inner_p.get_overlap_rectangle(prev_inner_p)
		# 			overlap2 = prev_inner_p.get_overlap_rectangle(inner_p)
					
		# 			# if overlap1[2]-overlap1[0]<inner_p.size[1]/2:
		# 			# 	continue

		# 			kp1,desc1 = choose_SIFT_key_points(inner_p,overlap1[0],overlap1[1],overlap1[2],overlap1[3],SIFT_folder)
		# 			kp2,desc2 = choose_SIFT_key_points(prev_inner_p,overlap2[0],overlap2[1],overlap2[2],overlap2[3],SIFT_folder)

		# 			kp.append(kp1)
		# 			desc.append(desc1)
		# 			prev_kp.append(kp2)
		# 			prev_desc.append(desc2)
					
		# 			matches.append(get_top_n_good_matches(desc2,desc1,kp2,kp1,3000,inner_p.size))

		# H = calculate_homography_for_super_patches(prev_kp,prev_desc,kp,desc,matches)
		
		# self.correct_all_patches_and_self_by_H(H,prev_super_patch)

		overlap1 = self.get_overlap_rectangle(prev_super_patch)
		overlap2 = prev_super_patch.get_overlap_rectangle(self)
		
		kp1 = self.lower_kp
		desc1 = self.lower_desc
		kp2 = prev_super_patch.upper_kp
		desc2 = prev_super_patch.upper_desc

		matches = get_top_n_good_matches(desc2,desc1,kp2,kp1,150000,self.patches[0].size)

		# self.draw_super_patch_and_lines(matches,kp1,kp2,patch_folder,'lines')

		H,percentage_inliers = find_homography(matches,kp2,kp1,overlap1,overlap2,False)

		self.correct_all_patches_and_self_by_H(H,prev_super_patch)

	def remove_randomly(self):
		upper_indexes = range(0,np.shape(self.upper_desc)[0])
		lower_indexes = range(0,np.shape(self.lower_desc)[0])

		if len(self.upper_kp)>262143:
			upper_sample_indexes = random.sample(upper_indexes,262143)
			self.upper_kp = [self.upper_kp[i] for i in upper_sample_indexes]
			self.upper_desc = self.upper_desc[upper_sample_indexes,:]

		if len(self.lower_kp)>262143:
			lower_sample_indexes = random.sample(lower_indexes,262143)
			self.lower_kp = [self.lower_kp[i] for i in lower_sample_indexes]
			self.lower_desc = self.lower_desc[lower_sample_indexes,:]


	def calculate_difference_from_UL(self,p):

		x_difference_from_UL = p.GPS_coords.UL_coord[0] - self.GPS_coords.UL_coord[0]
		y_difference_from_UL = self.GPS_coords.UL_coord[1] - p.GPS_coords.UL_coord[1]

		return x_difference_from_UL/self.x_ratio_GPS_over_pixel,y_difference_from_UL/self.y_ratio_GPS_over_pixel

	def calculate_super_sift_points(self,SIFT_folder):
		global no_of_cores_to_use

		upper_kp = []
		upper_desc = None
		lower_kp = []
		lower_desc = None

		args = []

		for p in self.patches:
			x_difference,y_difference = self.calculate_difference_from_UL(p)
			ukp,uds,lkp,lds = recalculate_keypoints_locations(p,SIFT_folder,x_difference,y_difference)

			upper_kp+=ukp
			if upper_desc is None:
				upper_desc = uds.copy()
			else:
				upper_desc = np.append(upper_desc,uds,axis=0)
			
			lower_kp+=lkp
			if lower_desc is None:
				lower_desc = lds.copy()
			else:
				lower_desc = np.append(lower_desc,lds,axis=0)

			

		return upper_kp,upper_desc,lower_kp,lower_desc

	def draw_kp(self):
		import matplotlib.pyplot as plt
		plt.axis('equal')

		upper_kp1 = [k[0] for k in self.upper_kp]
		upper_kp2 = [k[1] for k in self.upper_kp]
		plt.scatter(upper_kp1,upper_kp2,color='green',marker='.')

		lower_kp1 = [k[0] for k in self.lower_kp]
		lower_kp2 = [k[1] for k in self.lower_kp]

		plt.scatter(lower_kp1,lower_kp2,color='red',marker='.')

		# for k in self.upper_kp:
		# 	plt.scatter(k[0],k[1],color='green')

		# for k in self.lower_kp:
		# 	plt.scatter(k[0],k[1],color='red')

		plt.savefig('rows.png')

	def get_overlap_rectangle(self,patch,increase_size=True):
		p1_x = 0
		p1_y = 0
		p2_x = self.size[1]
		p2_y = self.size[0]

		detect_overlap = False

		if patch.GPS_coords.UL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.UL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			p1_y = int(((patch.GPS_coords.UL_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])
		
		if patch.GPS_coords.LL_coord[1]>=self.GPS_coords.LL_coord[1] and patch.GPS_coords.LL_coord[1]<=self.GPS_coords.UL_coord[1]:
			detect_overlap = True
			p2_y = int(((patch.GPS_coords.LR_coord[1]-self.GPS_coords.UL_coord[1]) / (self.GPS_coords.LL_coord[1]-self.GPS_coords.UL_coord[1]))*self.size[0])

		if patch.GPS_coords.UR_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UR_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			p2_x = int(((patch.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			
		if patch.GPS_coords.UL_coord[0]<=self.GPS_coords.UR_coord[0] and patch.GPS_coords.UL_coord[0]>=self.GPS_coords.UL_coord[0]:
			detect_overlap = True
			p1_x = int(((patch.GPS_coords.LL_coord[0]-self.GPS_coords.UL_coord[0]) / (self.GPS_coords.UR_coord[0]-self.GPS_coords.UL_coord[0]))*self.size[1])
			
		if patch.GPS_coords.is_coord_inside(self.GPS_coords.UL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.UR_coord) and \
		patch.GPS_coords.is_coord_inside(self.GPS_coords.LL_coord) and patch.GPS_coords.is_coord_inside(self.GPS_coords.LR_coord):
			p1_x = 0
			p1_y = 0
			p2_x = self.size[1]
			p2_y = self.size[0]
			detect_overlap = True

		if increase_size:
			if p1_x>0+self.size[1]/10:
				p1_x-=self.size[1]/10

			if p2_x<9*self.size[1]/10:
				p2_x+=self.size[1]/10

			if p1_y>0+self.size[0]/10:
				p1_y-=self.size[0]/10

			if p2_y<9*self.size[0]/10:
				p2_y+=self.size[0]/10

		if detect_overlap == False:
			return 0,0,0,0

		return int(p1_x),int(p1_y),int(p2_x),int(p2_y)

def detect_rows(address):
	center_second_dim_rows = []
	height_in_GPS = None
	patches = []
	size = (3296, 2472)

	with open(address) as f:
		lines = f.read()
		lines = lines.replace('"','')

		for l in lines.split('\n'):
			if l == '':
				break
			if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
				continue

			features = l.split(',')

			filename = features[0]
			upper_left = (float(features[1]),float(features[2]))
			lower_left = (float(features[3]),float(features[4]))
			upper_right = (float(features[5]),float(features[6]))
			lower_right = (float(features[7]),float(features[8]))
			center = (float(features[9]),float(features[10]))

			coord = Patch_GPS_coordinate(upper_left,upper_right,lower_left,lower_right,center)
			patches.append(Patch_2(filename,None,None,coord,size))


			if height_in_GPS is None:
				height_in_GPS = abs(upper_left[1]-lower_left[1])
				
			is_new = True

			for c in center_second_dim_rows:
				if abs(center[1]-c[1]) < height_in_GPS/10:
					is_new = False

			if is_new:
				center_second_dim_rows.append(center)

	patches_groups_by_rows = {}
	iterator = 0

	center_second_dim_rows = sorted(center_second_dim_rows, key=lambda x: x[1])

	for c in center_second_dim_rows:
		patches_groups_by_rows[(round(c[0],7),round(c[1],7))] = []

	for p in patches:
		min_distance = height_in_GPS*2
		min_row = None

		for c in center_second_dim_rows:
			distance = abs(p.GPS_coords.Center[1]-c[1])
			if distance<min_distance:
				min_distance = distance
				min_row = c

		patches_groups_by_rows[(round(min_row[0],7),round(min_row[1],7))].append(p)


	patches_groups_by_rows_new = []
	
	for g in patches_groups_by_rows:
		newlist = sorted(patches_groups_by_rows[g], key=lambda x: x.GPS_coords.Center[0], reverse=False)
		
		patches_groups_by_rows_new.append(newlist)

	# print(len(patches_groups_by_rows))
	# print(len(patches_groups_by_rows[g]))

	# import matplotlib.pyplot as plt
	
	# plt.axis('equal')

	# color = 'red'
	# total=0

	# for g in patches_groups_by_rows:
	# 	# print('{0}: {1}'.format(g,len(patches_groups_by_rows[g])))
	# 	total+=len(patches_groups_by_rows[g])

	# 	if color == 'red':
	# 		color = 'green'
	# 	else:
	# 		color = 'red'

	# 	for p in patches_groups_by_rows[g]:
	# 		plt.scatter(p.GPS_coords.Center[0],p.GPS_coords.Center[1],color=color,marker='.')
		
		
			
	# plt.savefig('rows.png')
	# print(total)

	# for g in patches_groups_by_rows:
	# 	newlist = sorted(patches_groups_by_rows[g], key=lambda x: x.GPS_coords.Center[0], reverse=False)
	# 	for p in newlist:
	# 		print(p.GPS_coords.UL_coord)
	# 		plt.scatter(p.GPS_coords.UL_coord[0],p.GPS_coords.UL_coord[1],color='red')
	# 		plt.scatter(p.GPS_coords.Center[0],p.GPS_coords.Center[1],color='green')
	# 		print(p.GPS_coords.Center)
	# 		plt.scatter(p.GPS_coords.LL_coord[0],p.GPS_coords.LL_coord[1],color='blue')
	# 	break

	# plt.savefig('rows.png')
	return patches_groups_by_rows_new

def save_rows(groups,path_to_save):
	result = []
	color = 0

	for g in groups:
		
		if color == 0:
			color = 1
		else:
			color = 0

		for p in groups[g]:
			result.append([p.GPS_coords.Center[0],p.GPS_coords.Center[1],color])
		
	np.save(path_to_save,np.array(result))

def draw_rows(path):
	import matplotlib.pyplot as plt

	plt.axis('equal')

	data = np.load(path)

	c = []
	for d in data:
		c.append('red' if d[2] == 1 else 'green')

	plt.scatter(data[:,0],data[:,1],color=c)

	plt.show()

def correct_supperpatches_iteratively(super_patches,SIFT_folder,patch_folder):

	spr = create_supper_patch_parallel(super_patches[0].patches+super_patches[1].patches,-1,SIFT_folder,patch_folder,True)
	spr.draw_super_patch(patch_folder,'combine')

	prev_super_patch = None

	for sp in super_patches:

		if prev_super_patch is None:
			prev_super_patch = sp
			continue

		sp.correct_whole_based_on_super_patch(prev_super_patch,SIFT_folder,patch_folder)


	spr = create_supper_patch_parallel(super_patches[0].patches+super_patches[1].patches,-1,SIFT_folder,patch_folder,True)
	spr.draw_super_patch(patch_folder,'combine_new')

def generate_superpatches(groups_by_rows,SIFT_folder,patch_folder):
	super_patches = []
	args = []

	for g,grp in enumerate(groups_by_rows):
		args.append((grp,g,SIFT_folder,patch_folder))

	processes = multiprocessing.Pool(no_of_cores_to_use)
	results = processes.map(create_supper_patch_parallel_helper,args)
	processes.close()

	super_patches = results

	# super_patches[8].draw_super_patch(patch_folder,'old')
	# super_patches[3].correct_supper_patch_internally(SIFT_folder,patch_folder)
	# super_patches[4].correct_supper_patch_internally(SIFT_folder,patch_folder)
	
	# super_patches[3].upper_kp, super_patches[3].upper_desc, super_patches[3].lower_kp, super_patches[3].lower_desc = super_patches[3].calculate_super_sift_points(SIFT_folder)
	# super_patches[3].remove_randomly()
	# super_patches[4].upper_kp, super_patches[4].upper_desc, super_patches[4].lower_kp, super_patches[4].lower_desc = super_patches[4].calculate_super_sift_points(SIFT_folder)
	# super_patches[4].remove_randomly()

	# sp = create_supper_patch_parallel(super_patches[3].patches+super_patches[4].patches,1,SIFT_folder,patch_folder)
	# sp.draw_super_patch(patch_folder,'combine')

	# super_patches[4].correct_whole_based_on_super_patch(super_patches[3],SIFT_folder,patch_folder)
	# # super_patches[8].draw_super_patch(patch_folder,'new')

	# sp = create_supper_patch_parallel(super_patches[3].patches+super_patches[4].patches,1,SIFT_folder,patch_folder)
	# sp.draw_super_patch(patch_folder,'combine_n')

	return super_patches

def create_supper_patch_parallel(patches,g,SIFT_folder,patch_folder,not_revise_internally=False):

	up = patches[0].GPS_coords.UL_coord[1]
	down = patches[0].GPS_coords.LL_coord[1]
	left = patches[0].GPS_coords.UL_coord[0]
	right = patches[0].GPS_coords.UR_coord[0]

	for p in patches:
		if p.GPS_coords.UL_coord[1]>=up:
			up=p.GPS_coords.UL_coord[1]

		if p.GPS_coords.LL_coord[1]<=down:
			down=p.GPS_coords.LL_coord[1]

		if p.GPS_coords.UL_coord[0]<=left:
			left=p.GPS_coords.UL_coord[0]

		if p.GPS_coords.UR_coord[0]>=right:
			right=p.GPS_coords.UR_coord[0]

		# plt.scatter(p.GPS_coords.Center[0],p.GPS_coords.Center[1],color='green')

	UL_coord = (left,up)
	UR_coord = (right,up)
	LL_coord = (left,down)
	LR_coord = (right,down)
	Center = ((left+right)/2,(down+up)/2)

	coord = Patch_GPS_coordinate(UL_coord,UR_coord,LL_coord,LR_coord,Center)

	sp = SuperPatch(g,patches,coord,SIFT_folder)
	
	if not not_revise_internally:
		# sp.draw_super_patch(patch_folder,'before_{0}'.format(g))
		sp.correct_supper_patch_internally(SIFT_folder,patch_folder)
		# sp.draw_super_patch(patch_folder,'after_{0}'.format(g))

		sp.upper_kp, sp.upper_desc, sp.lower_kp, sp.lower_desc = sp.calculate_super_sift_points(SIFT_folder)
		sp.remove_randomly()

	print('Super patch for row {0} has been successfully created and revised internally. '.format(g))
	sys.stdout.flush()

	return sp

def create_supper_patch_parallel_helper(args):
	
	return create_supper_patch_parallel(*args)



def stitch_rows(rows,path_to_save,image_path):
	iterator = 0
	args_list = []

	for r in rows:
		iterator +=1

		patches = rows[r]
		
		args_list.append((patches[0:30],False,path_to_save,image_path,iterator))
		break


	processes = multiprocessing.Pool(no_of_cores_to_use)
	processes.map(stitch_based_on_corrected_GPS_helper,args_list)

def correct_all_sub_patches(H,super_patch,previous_super_patch):
	c1 = [0,0,1]
	
	c1 = H.dot(c1).astype(int)
	
	# print(c1)

	diff_x = -c1[0]
	diff_y = -c1[1]

	gps_scale_x = (previous_super_patch.GPS_coords.UR_coord[0] - previous_super_patch.GPS_coords.UL_coord[0])/(previous_super_patch.size[1])
	gps_scale_y = (previous_super_patch.GPS_coords.LL_coord[1] - previous_super_patch.GPS_coords.UL_coord[1])/(previous_super_patch.size[0])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	new_UL = (round(previous_super_patch.GPS_coords.UL_coord[0]-diff_x,7),round(previous_super_patch.GPS_coords.UL_coord[1]-diff_y,7))
	# print(new_UL)
	diff_UL = (super_patch.GPS_coords.UL_coord[0]-new_UL[0],super_patch.GPS_coords.UL_coord[1]-new_UL[1])

	for p in super_patch.patches:
		new_UL = (p.GPS_coords.UL_coord[0]-diff_UL[0],p.GPS_coords.UL_coord[1]-diff_UL[1])
		new_UR = (p.GPS_coords.UR_coord[0]-diff_UL[0],p.GPS_coords.UR_coord[1]-diff_UL[1])
		new_LL = (p.GPS_coords.LL_coord[0]-diff_UL[0],p.GPS_coords.LL_coord[1]-diff_UL[1])
		new_LR = (p.GPS_coords.LR_coord[0]-diff_UL[0],p.GPS_coords.LR_coord[1]-diff_UL[1])
		new_center = (p.GPS_coords.Center[0]-diff_UL[0],p.GPS_coords.Center[1]-diff_UL[1])

		new_coords = Patch_GPS_coordinate(new_UL,new_UR,new_LL,new_LR,new_center)
		p.GPS_coords = new_coords

	super_new_UL = (super_patch.GPS_coords.UL_coord[0]-diff_UL[0],super_patch.GPS_coords.UL_coord[1]-diff_UL[1])
	super_new_UR = (super_patch.GPS_coords.UR_coord[0]-diff_UL[0],super_patch.GPS_coords.UR_coord[1]-diff_UL[1])
	super_new_LL = (super_patch.GPS_coords.LL_coord[0]-diff_UL[0],super_patch.GPS_coords.LL_coord[1]-diff_UL[1])
	super_new_LR = (super_patch.GPS_coords.LR_coord[0]-diff_UL[0],super_patch.GPS_coords.LR_coord[1]-diff_UL[1])
	super_new_center = (super_patch.GPS_coords.Center[0]-diff_UL[0],super_patch.GPS_coords.Center[1]-diff_UL[1])

	super_new_coords = Patch_GPS_coordinate(super_new_UL,super_new_UR,super_new_LL,super_new_LR,super_new_center)
	super_patch.GPS_coords = super_new_coords

def generate_superpatches_and_correct_GPS(groups_by_rows,SIFT_folder):

	results_final = 'Filename,Upper left,Lower left,Upper right,Lower right,Center\n'

	patches_count_done = 0
	previous_super_patch = None

	# import matplotlib.pyplot as plt
	# plt.axis('equal')


	for g in groups_by_rows:

		patches = groups_by_rows[g]

		up = patches[0].GPS_coords.UL_coord[1]
		down = patches[0].GPS_coords.LL_coord[1]
		left = patches[0].GPS_coords.UL_coord[0]
		right = patches[0].GPS_coords.UR_coord[0]

		for p in patches:
			if p.GPS_coords.UL_coord[1]>up:
				up=p.GPS_coords.UL_coord[1]

			if p.GPS_coords.LL_coord[1]<down:
				down=p.GPS_coords.LL_coord[1]

			if p.GPS_coords.UL_coord[0]<left:
				left=p.GPS_coords.UL_coord[0]

			if p.GPS_coords.UR_coord[0]>right:
				right=p.GPS_coords.UR_coord[0]

			# plt.scatter(p.GPS_coords.Center[0],p.GPS_coords.Center[1],color='green')

		UL_coord = (left,up)
		UR_coord = (right,up)
		LL_coord = (left,down)
		LR_coord = (right,down)
		Center = ((left+right)/2,(down+up)/2)

		coord = Patch_GPS_coordinate(UL_coord,UR_coord,LL_coord,LR_coord,Center)

		sp = SuperPatch(g,patches,coord,SIFT_folder)
		sp.draw_super_patch('/storage/ariyanzarei/2020-01-08-rgb/bin2tif_out')
		

		if previous_super_patch is not None:
			overlap1 = sp.get_overlap_rectangle(previous_super_patch)
			overlap2 = previous_super_patch.get_overlap_rectangle(sp)
			
			kp1 = sp.lower_kp
			desc1 = sp.lower_desc
			kp2 = previous_super_patch.upper_kp
			desc2 = previous_super_patch.upper_desc

			matches = get_good_matches(desc2,desc1)

			H,percentage_inliers = find_homography(matches,kp2,kp1,overlap1,overlap2,False)

			# plt.scatter(sp.GPS_coords.UL_coord[0],sp.GPS_coords.UL_coord[1],color='blue',marker='x')
			# plt.scatter(sp.GPS_coords.UR_coord[0],sp.GPS_coords.UR_coord[1],color='blue',marker='x')
			# plt.scatter(sp.GPS_coords.LL_coord[0],sp.GPS_coords.LL_coord[1],color='blue',marker='x')
			# plt.scatter(sp.GPS_coords.LR_coord[0],sp.GPS_coords.LR_coord[1],color='blue',marker='x')
			# plt.scatter(sp.GPS_coords.Center[0],sp.GPS_coords.Center[1],color='blue',marker='x')

			correct_all_sub_patches(H,sp,previous_super_patch)

			print('GPS for Super Patch # {0} corrected.'.format(patches_count_done))
			sys.stdout.flush()

			# plt.scatter(sp.GPS_coords.UL_coord[0],sp.GPS_coords.UL_coord[1],color='green',marker='x')
			# plt.scatter(sp.GPS_coords.UR_coord[0],sp.GPS_coords.UR_coord[1],color='green',marker='x')
			# plt.scatter(sp.GPS_coords.LL_coord[0],sp.GPS_coords.LL_coord[1],color='green',marker='x')
			# plt.scatter(sp.GPS_coords.LR_coord[0],sp.GPS_coords.LR_coord[1],color='green',marker='x')
			# plt.scatter(sp.GPS_coords.Center[0],sp.GPS_coords.Center[1],color='green',marker='x')

			# plt.savefig('rows.png')


		patches_count_done+=1
		results_final+=get_corrected_string(sp.patches)

		previous_super_patch = sp
		
	return results_final
		
def save_corrected_from_super_patches_string(res_string,corrected_filename):
	with open(corrected_filename,'w') as f:
		f.write(res_string)


def test_jittering(patch1,patch2):

	corrected_neighbors = [patch1,patch2]

	up = corrected_neighbors[0].GPS_coords.UL_coord[1]
	down = corrected_neighbors[0].GPS_coords.LL_coord[1]
	left = corrected_neighbors[0].GPS_coords.UL_coord[0]
	right = corrected_neighbors[0].GPS_coords.UR_coord[0]

	for p in corrected_neighbors:
		if p.GPS_coords.UL_coord[1]>=up:
			up=p.GPS_coords.UL_coord[1]

		if p.GPS_coords.LL_coord[1]<=down:
			down=p.GPS_coords.LL_coord[1]

		if p.GPS_coords.UL_coord[0]<=left:
			left=p.GPS_coords.UL_coord[0]

		if p.GPS_coords.UR_coord[0]>=right:
			right=p.GPS_coords.UR_coord[0]


	super_patch_size = (int(math.ceil((up-down)/GPS_TO_IMAGE_RATIO[1]))+100,int(math.ceil((right-left)/GPS_TO_IMAGE_RATIO[0]))+100,3)
	UL = (left,up)

	result = np.zeros(super_patch_size)

	for p in corrected_neighbors:
		p.load_img('/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures')	
		x_diff = p.GPS_coords.UL_coord[0] - UL[0]
		y_diff = UL[1] - p.GPS_coords.UL_coord[1]
		
		st_x = int(x_diff/GPS_TO_IMAGE_RATIO[0])
		st_y = int(y_diff/GPS_TO_IMAGE_RATIO[1])
		
		result[st_y:st_y+PATCH_SIZE[0],st_x:st_x+PATCH_SIZE[1],:] = p.rgb_img

	result = np.array(result).astype('uint8')
	result = cv2.resize(result,(int(result.shape[1]/5),int(result.shape[0]/5)))

	p1_x1,p1_y1,p1_x2,p1_y2 = patch1.get_overlap_rectangle(patch2)
	p2_x1,p2_y1,p2_x2,p2_y2 = patch2.get_overlap_rectangle(patch1)

	scr = calculate_dissimilarity(patch1,patch2,p1_x1,p1_y1,p1_x2,p1_y2,p2_x1,p2_y1,p2_x2,p2_y2) 
	print(scr)
	cv2.imshow('fig',result)
	cv2.waitKey(0)

def calculate_dissimilarity(p1,p2,p1_x1,p1_y1,p1_x2,p1_y2,p2_x1,p2_y1,p2_x2,p2_y2):
	overlap_1_img = p1.rgb_img[p1_y1:p1_y2,p1_x1:p1_x2,:]
	overlap_2_img = p2.rgb_img[p2_y1:p2_y2,p2_x1:p2_x2,:]

	shape_1 = np.shape(overlap_1_img)
	shape_2 = np.shape(overlap_2_img)

	if shape_1 != shape_2:
		if shape_1[0]<shape_2[0]:
			overlap_2_img = overlap_2_img[:shape_1[0],:,:]
			shape_2 = shape_1
		if shape_1[1]<shape_2[1]:
			overlap_2_img = overlap_2_img[:,:shape_1[1],:]
			shape_2 = shape_1
		
		if shape_2[0]<shape_1[0]:
			overlap_1_img = overlap_1_img[:shape_2[0],:,:]
			shape_1 = shape_2
		if shape_2[1]<shape_1[1]:
			overlap_1_img = overlap_1_img[:,:shape_2[1],:]
			shape_1 = shape_2

	if shape_1[0] == 0 or shape_1[1] == 0 or shape_2[0] == 0 or shape_2[1] == 0:
		return sys.maxsize

	overlap_1_img = cv2.cvtColor(overlap_1_img, cv2.COLOR_BGR2GRAY)
	overlap_2_img = cv2.cvtColor(overlap_2_img, cv2.COLOR_BGR2GRAY)

	overlap_1_img = cv2.blur(overlap_1_img,(5,5))
	overlap_2_img = cv2.blur(overlap_2_img,(5,5))

	ret1,overlap_1_img = cv2.threshold(overlap_1_img,0,255,cv2.THRESH_OTSU)
	ret1,overlap_2_img = cv2.threshold(overlap_2_img,0,255,cv2.THRESH_OTSU)

	tmp_size = np.shape(overlap_1_img)
	
	overlap_1_img[overlap_1_img==255] = 1
	overlap_2_img[overlap_2_img==255] = 1

	xnor_images = np.logical_xor(overlap_1_img,overlap_2_img)

	dissimilarity = round(np.sum(xnor_images)/(tmp_size[0]*tmp_size[1]),2)
	# dissimilarity =  np.sum((overlap_1_img.astype("float") - overlap_2_img.astype("float")) ** 2)
	# dissimilarity /= float(overlap_1_img.shape[0] * overlap_1_img.shape[1])
	

	return dissimilarity

def main():
	global server

	if server == 'coge':
		patch_folder = '/storage/ariyanzarei/2020-01-08-rgb/bin2tif_out'
		SIFT_folder = '/storage/ariyanzarei/2020-01-08-rgb/SIFT'
		lid_file = '/storage/ariyanzarei/2020-01-08-rgb/lids.txt'
		coordinates_file = '/storage/ariyanzarei/2020-01-08-rgb/2020-01-08_coordinates.csv'
		CORRECTED_coordinates_file = '/storage/ariyanzarei/2020-01-08-rgb/2020-01-08_coordinates_CORRECTED.csv'
		plot_npy_file = '/storage/ariyanzarei/2020-01-08-rgb/plt.npy'
		row_save_path = '/storage/ariyanzarei/2020-01-08-rgb/rows'

	elif server == 'laplace.cs.arizona.edu':
		patch_folder = '/data/plant/full_scans/2020-01-08-rgb/bin2tif_out'
		SIFT_folder = '/data/plant/full_scans/2020-01-08-rgb/SIFT'
		lid_file = '/data/plant/full_scans/2020-01-08-rgb/lids.txt'
		coordinates_file = '/data/plant/full_scans/metadata/2020-01-08_coordinates.csv'
		CORRECTED_coordinates_file = '/data/plant/full_scans/metadata/2020-01-08_coordinates_CORRECTED.csv'
		plot_npy_file = '/data/plant/full_scans/2020-01-08-rgb/plt.npy'

	elif server == 'ariyan':
		patch_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures'
		SIFT_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/SIFT'
		lid_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/lids.txt'
		coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords.txt'
		CORRECTED_coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt'
		plot_npy_file = '/home/ariyan/Desktop/plt.npy'


	if server == 'coge':
		print('RUNNING ON -- {0} --'.format(server))
		# lids = get_lids(lid_file)
		# groups = get_groups_and_patches_with_lids(patch_folder,coordinates_file,SIFT_folder,lids)
		# results = correct_GPS_MST_groups(groups,SIFT_folder,patch_folder)
		# save_coordinates_from_string(results,CORRECTED_coordinates_file)
		
		row_groups = detect_rows(coordinates_file)
		super_patches = generate_superpatches(row_groups[3:5],SIFT_folder,patch_folder)
		correct_supperpatches_iteratively(super_patches,SIFT_folder,patch_folder)
		# results = generate_superpatches_and_correct_GPS(row_groups,SIFT_folder)
		# save_rows(row_groups,plot_npy_file)
		# draw_rows(plot_npy_file)
		# save_corrected_from_super_patches_string(results,CORRECTED_coordinates_file)
		# stitch_rows(row_groups,row_save_path,patch_folder)


	elif server == 'laplace.cs.arizona.edu':
		print('RUNNING ON -- {0} --'.format(server))
		os.system("taskset -p -c 1,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,44,45,46 %d" % os.getpid())
		# lids = get_lids(lid_file)
		# groups = get_groups_and_patches_with_lids(patch_folder,coordinates_file,SIFT_folder,lids)
		# results = correct_GPS_MST_groups(groups,SIFT_folder,patch_folder)
		# save_coordinates_from_string(results,CORRECTED_coordinates_file)
		row_groups = detect_rows(coordinates_file)
		super_patches = generate_superpatches(row_groups[3:5],SIFT_folder,patch_folder)
		correct_supperpatches_iteratively(super_patches,SIFT_folder,patch_folder)


	elif server == 'ariyan':
		print('RUNNING ON -- {0} --'.format(server))
		patches = read_all_data_on_server(patch_folder,coordinates_file,SIFT_folder,False)
		test_jittering(patches[0],patches[3])
		patches[3].GPS_coords.UL_coord = (patches[3].GPS_coords.UL_coord[0]+0.00000003,patches[3].GPS_coords.UL_coord[1])
		patches[3].GPS_coords.UR_coord = (patches[3].GPS_coords.UR_coord[0]+0.00000003,patches[3].GPS_coords.UR_coord[1])
		patches[3].GPS_coords.LL_coord = (patches[3].GPS_coords.LL_coord[0]+0.00000003,patches[3].GPS_coords.LL_coord[1])
		patches[3].GPS_coords.LR_coord = (patches[3].GPS_coords.LR_coord[0]+0.00000003,patches[3].GPS_coords.LR_coord[1])
		patches[3].GPS_coords.Center = (patches[3].GPS_coords.Center[0]+0.00000003,patches[3].GPS_coords.Center[1])
		test_jittering(patches[0],patches[3])
		# patches[0].GPS_Corrected = True
		
		# results = correct_GPS_MST_groups({'131':patches},SIFT_folder,patch_folder)
		# # results = correct_GPS_new_code_no_heap_precalculate_groups({'131':patches},SIFT_folder,patch_folder)
		# save_coordinates_from_string(results,CORRECTED_coordinates_file)

		# patches = read_all_data()
		# final_patches = stitch_based_on_corrected_GPS(patches,True)
		# show_and_save_final_patches(final_patches)
		# print(patches[0].GPS_coords)
		# print(patches[1].GPS_coords.UL_coord[1]-patches[1].GPS_coords.LL_coord[1])
		# print(patches[1].GPS_coords.UR_coord[0]-patches[1].GPS_coords.UL_coord[0])
		# draw_rows(plot_npy_file)
		

	# patches = read_all_data_on_server(patch_folder,coordinates_file,SIFT_folder,False)
	# patches[0].GPS_Corrected = True
	# lids = get_lids(lid_file)
	# groups = get_groups_and_patches_with_lids(patch_folder,coordinates_file,SIFT_folder,lids)
	# results = correct_GPS_two_queues_groups({'131':patches},SIFT_folder,patch_folder)
	# results = correct_GPS_new_code_no_heap_precalculate_groups({'131':patches},SIFT_folder,patch_folder)
	# save_coordinates_from_string(results,CORRECTED_coordinates_file)

	# patches[0].GPS_Corrected = True
	# final_patches = correct_GPS_coords_new_code(patches,False,False,SIFT_folder)
	# final_patches = correct_GPS_two_queues(patches,SIFT_folder)
	# save_coordinates(corrected_patches,CORRECTED_coordinates_file)

	# patches = read_all_data_on_server(patch_folder,coordinates_file,SIFT_folder,False)
	# lids = get_lids(lid_file)
	# save_group_data(group_images_by_nearest_lid(lids,patches),lids,len(patches),plot_npy_file)
	# plot_groups(plot_npy_file,lids)

	# patches = read_all_data()
	# final_patches = stitch_based_on_corrected_GPS(patches,True)
	# show_and_save_final_patches(final_patches)

	# final_patches = stitch_complete(patches,True,True)
	# final_patches = correct_GPS_coords(patches,False,False,SIFT_folder)
	# final_patches = correct_GPS_coords_new_code_parallel(patches,False,False,SIFT_folder)	

	# patches = read_all_data_on_server(patch_folder,coordinates_file,SIFT_folder,False)
	# final_patches = correct_GPS_coords_new_code(patches,False,False,SIFT_folder)
	# save_coordinates(final_patches,CORRECTED_coordinates_file)

	# patches = read_all_data_on_server(patch_folder,coordinates_file,SIFT_folder,False)
	# lids = get_lids(lid_file)
	# groups = get_groups_and_patches_with_lids(patch_folder,coordinates_file,SIFT_folder,lids)
	# results = correct_GPS_new_code_no_heap_precalculate_groups(groups,SIFT_folder,patch_folder)
	# results = correct_GPS_two_queues_groups(groups,SIFT_folder)
	# corrected_patches = correct_GPS_new_code_groups(groups,False,False,SIFT_folder)
	# save_coordinates_from_string(results,CORRECTED_coordinates_file)
	
	# save_group_data(group_images_by_nearest_lid(lids,patches),lids,len(patches),plot_npy_file)
	# get_name_of_patches_with_lids(coordinates_file,lids)
	
	# plot_groups('/home/ariyan/Desktop/plt.npy')
 

def report_time(start,end):
	print('-----------------------------------------------------------')
	print('Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.'.format(start,end,end-start))

server_core = {'coge':64,'laplace.cs.arizona.edu':20,'ariyan':4}
# server = 'coge' #'laplace' 'local'
server = socket.gethostname()
no_of_cores_to_use = server_core[server]

start_time = datetime.datetime.now()
main()
end_time = datetime.datetime.now()
report_time(start_time,end_time)