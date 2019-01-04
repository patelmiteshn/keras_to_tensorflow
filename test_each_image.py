import posenet
import numpy as np
import sys
import random
import cv2
import h5py
import os.path
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import math
from random import uniform
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import re
import json


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))

random.seed(datetime.now())


def readDataFromFile(input_h5py_file):
	with h5py.File(input_h5py_file, "r") as f:
		data_dict = {}

		for group in f.keys():
			for key in f[group].keys():
				if key == 'image_mean':
					image_mean = f['posenet'][key]
					image_mean = np.reshape(image_mean, (224, 224, 3))
					data_dict['image_mean'] = image_mean
				else:
					data_dict[key] = f[group][key][0]

		return data_dict


def subtractMean(image, image_mean):
	image = np.reshape(image, (224, 224, 3))
	image = np.transpose(image, (2, 0, 1))
	image_mean = image_mean.reshape((3, 224, 224))
	image = image - image_mean.astype(float)
	image = np.squeeze(image)
	image = np.transpose(image, (1, 2, 0))

	return image


def getPoseFromTensor(tensor):
	predictedPose = []
	predictedPose.append(tensor[0][0][0])
	predictedPose.append(tensor[0][0][1])
	predictedPose.append(tensor[0][0][2])
	predictedPose.append(tensor[1][0][0])
	predictedPose.append(tensor[1][0][1])
	predictedPose.append(tensor[1][0][2])
	predictedPose.append(tensor[1][0][3])

	return np.asarray(predictedPose)

def error(predictedPose, groundtruth):

	q1 = predictedPose[3:7] / np.linalg.norm(predictedPose[3:7])
	q2 = groundtruth[3:7] / np.linalg.norm(groundtruth[3:7])
	d = abs(np.sum(np.multiply(q1,q2)))
	theta_error = 2 * np.arccos(d) * 180 / math.pi
	error_xyz = np.linalg.norm(predictedPose[0:3] - groundtruth[0:3])
	# print('error in position: {} m and orientation: {} degrees'.format(error_xyz, theta_error) )
	return error_xyz, theta_error

def load_image_mean(model_name):
	with open(model_name + '_image_mean.json','r') as inputfile:
		image_mean = json.load(inputfile)
	image_mean = np.array(image_mean)
	image_mean = np.reshape(image_mean, (224, 224, 3))
	# print('image mean values after loading: {}'.format(image_mean))
	return image_mean

def test(TEST_ALL_IMAGES):
# 	root_path = '/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/'
# 	model_name = '2017_09_04_22_37_train_kin_no_drift'
# 	model_path = root_path + 'model/' + model_name + '_posenet.h5'
	
	### NEW MODELS ###
	root_path = '/ssd/posenet_training_output/'
	model_name = '2018_11_09_00_00_train_train_data_training_largedataset_w_halloweendecor_rotated_BN_more_oct2018'
	model_path = root_path + 'weights/checkpoint_weights/' + model_name + '_posenet.h5'

	data_dict = readDataFromFile(root_path + 'training_data_info/{}_posenet_data.h5py'.format(model_name))
	
	# Test model
	model = posenet.create_posenet()
	if 0:
		model_json = model.to_json()
		with open(model_name + '_model_json_file.json', 'w') as f:
			f.write(model_json)
	
	graph = tf.get_default_graph()
	# load model weight
	model.load_weights(model_path)
	# get image mean value
	image_mean = data_dict['image_mean']
	# print('image mean values before loading: {}'.format(image_mean) )
	image_mean = image_mean.flatten()
	image_mean = image_mean.tolist()
	with open(model_name + '_image_mean.json','w') as outfile:
		json.dump(image_mean,outfile)

	image_mean = load_image_mean(model_name)
	# read file to get image file name and ground truth pose
	# test data path
	if TEST_ALL_IMAGES:
		test_data_directory = '/home/mitesh/Documents/keras_to_tensorflow/'
		fread = open(test_data_directory + 'test_images_new/groundTruth.csv', 'r')
	else:
		test_image_name = '/home/mitesh/Documents/keras_to_tensorflow/test_images_new/2300_image_save.png'
	

	## save results to file
	if TEST_ALL_IMAGES:
		fwrite = open('results' + model_name + '.csv', 'w')
		for string in fread:
			groundtruth = []
			temp = re.split("[  ' \n ,]+", string)
			# temp = string.split(',')
			img = cv2.imread(test_data_directory + temp[0])
			fwrite.write(temp[0]+',')
			for idx in range(1,len(temp)-1):
				groundtruth.append(float(temp[idx]))
				fwrite.write(temp[idx] + ',')

			########### predict
			# Downsize the image and crop
			image = subtractMean(img, image_mean)

			images = np.empty((1, 224, 224, 3), dtype=float)
			images[0] = image

			# This 'with' statement is to address a bug with keras/tensorflow. It
			# doesn't have any functional effect.
			with graph.as_default():
				robot_pose = model.predict(images, batch_size=1, verbose=0)
				groundtruth = np.asarray(groundtruth)
				predictedPose = getPoseFromTensor(robot_pose)
				for pose in predictedPose:
					value = "%.5f" % pose
					fwrite.write(value + ',')
				error_xyz, theta_error = error(predictedPose, groundtruth)
				fwrite.write(str(error_xyz) + ',')
				fwrite.write(str(theta_error) + '\n')
			print('img: {}, predicted pose is: {}, ground truth is: {}, position error: {} meters ,orientation error: {} degree'.format(temp[0], predictedPose, np.asarray(groundtruth), error_xyz, theta_error ) )
		fwrite.close()
	else:
		img = cv2.imread(test_image_name)
		image = subtractMean(img, image_mean)
		images = np.empty((1, 224, 224, 3), dtype=float)
		images[0] = image
		with graph.as_default():
			robot_pose = model.predict(images, batch_size=1, verbose=0)
			predictedPose = getPoseFromTensor(robot_pose)
		print('img: {}, predicted pose is: {}'.format(test_image_name, predictedPose) )
			
			# print('for img: {}, position error: {} meters and orientation error: {}'.format(temp[0], error_xyz, theta_error))
	

if __name__ == "__main__":
	test(True)


