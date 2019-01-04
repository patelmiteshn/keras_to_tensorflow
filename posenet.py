from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten
from keras.layers import merge, Reshape, Activation, BatchNormalization, concatenate
from keras.utils.conv_utils import convert_kernel
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import h5py
import math


beta = None


# def setBeta():
# 	global beta
# 	beta = run_pipeline.TrainingHyperParams.beta

	
# def euc_loss3x(y_true, y_pred):
# 	lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
# 	return 1 * lx


# def euc_loss3q(y_true, y_pred):
# 	lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
# 	return beta * lq

def create_posenet_sample(weights_path=None, tune=False):
	with tf.device('/gpu:0'):
		input = Input(shape=(224, 224, 3))

		conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu',  name='conv1')(input)
		pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
		norm1 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm1')(pool1)
		reduction2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='reduction2')(norm1)
		conv2 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='conv2')(reduction2)
		norm2 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm2')(conv2)
		pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
		icp1_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp1_reduction1')(pool2)
		icp1_out1 = Conv2D(128, (3, 3), padding='same', activation='relu',   name='icp1_out1')(icp1_reduction1)
		icp1_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp1_reduction2')(pool2)
		icp1_out2 = Conv2D(32, (5, 5), padding='same', activation='relu',   name='icp1_out2')(icp1_reduction2)
		icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)
		icp1_out3 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp1_out3')(icp1_pool)
		icp1_out0 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp1_out0')(pool2)
		icp2_in = concatenate(inputs=[icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

		cls3_fc1_flat = Flatten()(icp2_in)
		cls3_fc1_pose = Dense(100, activation='relu', name='cls3_fc1_poset')(cls3_fc1_flat)
		dropout3 = Dropout(0.5)(cls3_fc1_pose)
		cls3_fc_pose_all = Dense(7, name='cls3_fc_pose_all')(dropout3)
		posenet = Model(outputs=cls3_fc_pose_all, inputs=input)
		# cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyzt')(dropout3)
		# cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqrt')(dropout3)

		# posenet = Model(
		#     outputs=[cls3_fc_pose_xyz, cls3_fc_pose_wpqr], inputs=input)

	if tune:
		if weights_path:
			weights_data = np.load(weights_path, encoding='latin1').item()
			# weights_data = np.load(weights_path).item()
			for layer in posenet.layers:
				if layer.name in weights_data.keys():
					print('weights for layer : {} loaded'.format(layer.name))
					layer_weights = weights_data[layer.name]
					layer.set_weights((layer_weights['weights'], layer_weights['biases']))
					# print("FINISHED SETTING THE WEIGHTS!")
	return posenet


def create_posenet(weights_path=None, tune=False):
	# creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
	#####################################################################
	# MITESH
	# https://www.tensorflow.org/tutorials/using_gpu 
	# /gpu:0 should be used
	# https://github.com/kentsommer/keras-posenet/issues/1 : as per this note it will pick up GPU is there is one
	#####################################################################
	with tf.device('/gpu:0'):
		input = Input(shape=(224, 224, 3))

		conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu',  name='conv1')(input)
		pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
		norm1 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm1')(pool1)
		reduction2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='reduction2')(norm1)
		conv2 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='conv2')(reduction2)
		norm2 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm2')(conv2)
		pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
		icp1_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp1_reduction1')(pool2)
		icp1_out1 = Conv2D(128, (3, 3), padding='same', activation='relu',   name='icp1_out1')(icp1_reduction1)
		icp1_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp1_reduction2')(pool2)
		icp1_out2 = Conv2D(32, (5, 5), padding='same', activation='relu',   name='icp1_out2')(icp1_reduction2)
		icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)
		icp1_out3 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp1_out3')(icp1_pool)
		icp1_out0 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp1_out0')(pool2)
		icp2_in = concatenate(inputs=[icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')
		
		### modified by mitesh.. Adding batch normalization
		icp2_reduction1_cov2 = Conv2D(128, (1, 1), padding='same', name='icp2_reduction1')(icp2_in)
		norm3 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm3')(icp2_reduction1_cov2)
		icp2_reduction1 = Activation('relu')(norm3)
		
		icp2_out1_cov2 = Conv2D(192, (3, 3), padding='same', name='icp2_out1')(icp2_reduction1)
		norm4 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm4')(icp2_out1_cov2)
		icp2_out1 = Activation('relu')(norm4)
		
		icp2_reduction2_cov2 = Conv2D(32, (1, 1), padding='same', name='icp2_reduction2')(icp2_in)
		norm5 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm5')(icp2_reduction2_cov2)
		icp2_reduction2 = Activation('relu')(norm5)

		icp2_out2_cov2 = Conv2D(96, (5, 5), padding='same', name='icp2_out2')(icp2_reduction2)
		norm6 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm6')(icp2_out2_cov2)
		icp2_out2 = Activation('relu')(norm6)
		
		icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
		icp2_out3_cov2 = Conv2D(64, (1, 1), padding='same',  name='icp2_out3')(icp2_pool)
		norm7 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm7')(icp2_out3_cov2)
		icp2_out3 = Activation('relu')(norm7)
		
		icp2_out0_cov2 = Conv2D(128, (1, 1), padding='same', name='icp2_out0')(icp2_in)
		norm8 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm8')(icp2_out0_cov2)
		icp2_out0 = Activation('relu')(norm8)
		
		icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

		
		icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
		icp3_reduction1_cov2 = Conv2D(96, (1, 1), padding='same', name='icp3_reduction1')(icp3_in)
		norm9 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm9')(icp3_reduction1_cov2)
		icp3_reduction1 = Activation('relu')(norm9)
		
		icp3_out1_cov2 = Conv2D(208, (3, 3), padding='same', name='icp3_out1')(icp3_reduction1)
		norm10 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm10')(icp3_out1_cov2)
		icp3_out1 = Activation('relu')(norm10)
		
		icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp3_reduction2')(icp3_in)
		icp3_out2_cov2 = Conv2D(48, (5, 5), padding='same', name='icp3_out2')(icp3_reduction2)
		norm11 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm11')(icp3_out2_cov2)
		icp3_out2 = Activation('relu')(norm11)
		
		icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
		icp3_out3_cov2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp3_out3')(icp3_pool)
		norm12 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm12')(icp3_out3_cov2)
		icp3_out3 = Activation('relu')(norm12)
		
		icp3_out0_cov2 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp3_out0')(icp3_in)
		norm13 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm13')(icp3_out0_cov2)
		icp3_out0 = Activation('relu')(norm13)
		
		icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

# 		icp2_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_reduction1')(icp2_in)
# 		icp2_out1 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='icp2_out1')(icp2_reduction1)
# 		icp2_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp2_reduction2')(icp2_in)
# 		icp2_out2 = Conv2D(96, (5, 5), padding='same', activation='relu',   name='icp2_out2')(icp2_reduction2)
# 		icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
# 		icp2_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp2_out3')(icp2_pool)
# 		icp2_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_out0')(icp2_in)
# 		icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

# 		icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
# 		icp3_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp3_reduction1')(icp3_in)
# 		icp3_out1 = Conv2D(208, (3, 3), padding='same', activation='relu',   name='icp3_out1')(icp3_reduction1)
# 		icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp3_reduction2')(icp3_in)
# 		icp3_out2 = Conv2D(48, (5, 5), padding='same', activation='relu',   name='icp3_out2')(icp3_reduction2)
# 		icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
# 		icp3_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp3_out3')(icp3_pool)
# 		icp3_out0 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp3_out0')(icp3_in)
# 		icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

		icp4_reduction1 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp4_reduction1')(icp3_out)
		icp4_out1_cov2 = Conv2D(224, (3, 3), padding='same',  name='icp4_out1')(icp4_reduction1)
		norm14 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm14')(icp4_out1_cov2)
		icp4_out1 = Activation('relu')(norm14)
		
		
		icp4_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp4_reduction2')(icp3_out)
		icp4_out2_cov2 = Conv2D(64, (5, 5), padding='same', name='icp4_out2')(icp4_reduction2)
		norm15 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm15')(icp4_out2_cov2)
		icp4_out2 = Activation('relu')(norm15)
		
		icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)
		icp4_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp4_out3')(icp4_pool)
		icp4_out0_cov2 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp4_out0')(icp3_out)
		norm16 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm16')(icp4_out0_cov2)
		icp4_out0 = Activation('relu')(norm16)
		
		icp4_out = concatenate(inputs=[icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

		icp5_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_reduction1')(icp4_out)
		icp5_out1 = Conv2D(256, (3, 3), padding='same', activation='relu',   name='icp5_out1')(icp5_reduction1)
		icp5_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp5_reduction2')(icp4_out)
		icp5_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp5_out2')(icp5_reduction2)
		icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)
		icp5_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp5_out3')(icp5_pool)
		icp5_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_out0')(icp4_out)
		icp5_out = concatenate(inputs=[icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

		icp6_reduction1 = Conv2D(144, (1, 1), padding='same', activation='relu',   name='icp6_reduction1')(icp5_out)
		icp6_out1 = Conv2D(288, (3, 3), padding='same', activation='relu',   name='icp6_out1')(icp6_reduction1)
		icp6_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp6_reduction2')(icp5_out)
		icp6_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp6_out2')(icp6_reduction2)
		icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)
		icp6_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp6_out3')(icp6_pool)
		icp6_out0 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp6_out0')(icp5_out)
		icp6_out = concatenate(inputs=[icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

		icp7_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp7_reduction1')(icp6_out)
		icp7_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp7_out1')(icp7_reduction1)
		icp7_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp7_reduction2')(icp6_out)
		icp7_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp7_out2')(icp7_reduction2)
		icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)
		icp7_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp7_out3')(icp7_pool)
		icp7_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp7_out0')(icp6_out)
		icp7_out = concatenate(inputs=[icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

		icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)
		icp8_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp8_reduction1')(icp8_in)
		icp8_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp8_out1')(icp8_reduction1)
		icp8_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp8_reduction2')(icp8_in)
		icp8_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp8_out2')(icp8_reduction2)
		icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)
		icp8_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp8_out3')(icp8_pool)
		icp8_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp8_out0')(icp8_in)
		icp8_out = concatenate(inputs=[icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

		icp9_reduction1 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp9_reduction1')(icp8_out)
		icp9_out1 = Conv2D(384, (3, 3), padding='same', activation='relu',   name='icp9_out1')(icp9_reduction1)
		icp9_reduction2 = Conv2D(48, (1, 1), padding='same', activation='relu',   name='icp9_reduction2')(icp8_out)
		icp9_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp9_out2')(icp9_reduction2)
		icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)
		icp9_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp9_out3')(icp9_pool)
		icp9_out0 = Conv2D(384, (1, 1), padding='same', activation='relu',   name='icp9_out0')(icp8_out)
		icp9_out = concatenate(inputs=[icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

		cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)
		cls3_fc1_flat = Flatten()(cls3_pool)
		cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
		dropout3 = Dropout(0.5)(cls3_fc1_pose)
		cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(dropout3)
		cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(dropout3)

		posenet = Model(
			outputs=[cls3_fc_pose_xyz, cls3_fc_pose_wpqr], inputs=input)

	if tune:
		if weights_path:
			weights_data = np.load(weights_path, encoding='latin1').item()
			# weights_data = np.load(weights_path).item()
			for layer in posenet.layers:
				if layer.name in weights_data.keys():
					layer_weights = weights_data[layer.name]
					layer.set_weights((layer_weights['weights'], layer_weights['biases']))
					#print("FINISHED SETTING THE WEIGHTS!")
	return posenet


def create_posenet_2d(weights_path=None, tune=False):
	
	# creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
	#####################################################################
	# MITESH
	# https://www.tensorflow.org/tutorials/using_gpu 
	# /gpu:0 should be used
	# https://github.com/kentsommer/keras-posenet/issues/1 : as per this note it will pick up GPU is there is one
	#####################################################################
	with tf.device('/gpu:0'):
		input = Input(shape=(224, 224, 3))

		conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu',  name='conv1')(input)
		pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
		norm1 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm1')(pool1)
		reduction2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='reduction2')(norm1)
		conv2 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='conv2')(reduction2)
		norm2 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm2')(conv2)
		pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
		icp1_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp1_reduction1')(pool2)
		icp1_out1 = Conv2D(128, (3, 3), padding='same', activation='relu',   name='icp1_out1')(icp1_reduction1)
		icp1_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp1_reduction2')(pool2)
		icp1_out2 = Conv2D(32, (5, 5), padding='same', activation='relu',   name='icp1_out2')(icp1_reduction2)
		icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)
		icp1_out3 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp1_out3')(icp1_pool)
		icp1_out0 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp1_out0')(pool2)
		icp2_in = concatenate(inputs=[icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')
		
		### modified by mitesh.. Adding batch normalization
		icp2_reduction1_cov2 = Conv2D(128, (1, 1), padding='same', name='icp2_reduction1')(icp2_in)
		norm3 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm3')(icp2_reduction1_cov2)
		icp2_reduction1 = Activation('relu')(norm3)
		
		icp2_out1_cov2 = Conv2D(192, (3, 3), padding='same', name='icp2_out1')(icp2_reduction1)
		norm4 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm4')(icp2_out1_cov2)
		icp2_out1 = Activation('relu')(norm4)
		
		icp2_reduction2_cov2 = Conv2D(32, (1, 1), padding='same', name='icp2_reduction2')(icp2_in)
		norm5 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm5')(icp2_reduction2_cov2)
		icp2_reduction2 = Activation('relu')(norm5)

		icp2_out2_cov2 = Conv2D(96, (5, 5), padding='same', name='icp2_out2')(icp2_reduction2)
		norm6 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm6')(icp2_out2_cov2)
		icp2_out2 = Activation('relu')(norm6)
		
		icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
		icp2_out3_cov2 = Conv2D(64, (1, 1), padding='same',  name='icp2_out3')(icp2_pool)
		norm7 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm7')(icp2_out3_cov2)
		icp2_out3 = Activation('relu')(norm7)
		
		icp2_out0_cov2 = Conv2D(128, (1, 1), padding='same', name='icp2_out0')(icp2_in)
		norm8 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm8')(icp2_out0_cov2)
		icp2_out0 = Activation('relu')(norm8)
		
		icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

		
		icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
		icp3_reduction1_cov2 = Conv2D(96, (1, 1), padding='same', name='icp3_reduction1')(icp3_in)
		norm9 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm9')(icp3_reduction1_cov2)
		icp3_reduction1 = Activation('relu')(norm9)
		
		icp3_out1_cov2 = Conv2D(208, (3, 3), padding='same', name='icp3_out1')(icp3_reduction1)
		norm10 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm10')(icp3_out1_cov2)
		icp3_out1 = Activation('relu')(norm10)
		
		icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp3_reduction2')(icp3_in)
		icp3_out2_cov2 = Conv2D(48, (5, 5), padding='same', name='icp3_out2')(icp3_reduction2)
		norm11 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm11')(icp3_out2_cov2)
		icp3_out2 = Activation('relu')(norm11)
		
		icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
		icp3_out3_cov2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp3_out3')(icp3_pool)
		norm12 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm12')(icp3_out3_cov2)
		icp3_out3 = Activation('relu')(norm12)
		
		icp3_out0_cov2 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp3_out0')(icp3_in)
		norm13 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm13')(icp3_out0_cov2)
		icp3_out0 = Activation('relu')(norm13)
		
		icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

# 		icp2_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_reduction1')(icp2_in)
# 		icp2_out1 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='icp2_out1')(icp2_reduction1)
# 		icp2_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp2_reduction2')(icp2_in)
# 		icp2_out2 = Conv2D(96, (5, 5), padding='same', activation='relu',   name='icp2_out2')(icp2_reduction2)
# 		icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
# 		icp2_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp2_out3')(icp2_pool)
# 		icp2_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_out0')(icp2_in)
# 		icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

# 		icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
# 		icp3_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp3_reduction1')(icp3_in)
# 		icp3_out1 = Conv2D(208, (3, 3), padding='same', activation='relu',   name='icp3_out1')(icp3_reduction1)
# 		icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp3_reduction2')(icp3_in)
# 		icp3_out2 = Conv2D(48, (5, 5), padding='same', activation='relu',   name='icp3_out2')(icp3_reduction2)
# 		icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
# 		icp3_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp3_out3')(icp3_pool)
# 		icp3_out0 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp3_out0')(icp3_in)
# 		icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

		icp4_reduction1 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp4_reduction1')(icp3_out)
		icp4_out1_cov2 = Conv2D(224, (3, 3), padding='same',  name='icp4_out1')(icp4_reduction1)
		norm14 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm14')(icp4_out1_cov2)
		icp4_out1 = Activation('relu')(norm14)
		
		
		icp4_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp4_reduction2')(icp3_out)
		icp4_out2_cov2 = Conv2D(64, (5, 5), padding='same', name='icp4_out2')(icp4_reduction2)
		norm15 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm15')(icp4_out2_cov2)
		icp4_out2 = Activation('relu')(norm15)
		
		icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)
		icp4_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp4_out3')(icp4_pool)
		icp4_out0_cov2 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp4_out0')(icp3_out)
		norm16 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm16')(icp4_out0_cov2)
		icp4_out0 = Activation('relu')(norm16)
		
		icp4_out = concatenate(inputs=[icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

		icp5_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_reduction1')(icp4_out)
		icp5_out1 = Conv2D(256, (3, 3), padding='same', activation='relu',   name='icp5_out1')(icp5_reduction1)
		icp5_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp5_reduction2')(icp4_out)
		icp5_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp5_out2')(icp5_reduction2)
		icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)
		icp5_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp5_out3')(icp5_pool)
		icp5_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_out0')(icp4_out)
		icp5_out = concatenate(inputs=[icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

		icp6_reduction1 = Conv2D(144, (1, 1), padding='same', activation='relu',   name='icp6_reduction1')(icp5_out)
		icp6_out1 = Conv2D(288, (3, 3), padding='same', activation='relu',   name='icp6_out1')(icp6_reduction1)
		icp6_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp6_reduction2')(icp5_out)
		icp6_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp6_out2')(icp6_reduction2)
		icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)
		icp6_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp6_out3')(icp6_pool)
		icp6_out0 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp6_out0')(icp5_out)
		icp6_out = concatenate(inputs=[icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

		icp7_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp7_reduction1')(icp6_out)
		icp7_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp7_out1')(icp7_reduction1)
		icp7_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp7_reduction2')(icp6_out)
		icp7_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp7_out2')(icp7_reduction2)
		icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)
		icp7_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp7_out3')(icp7_pool)
		icp7_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp7_out0')(icp6_out)
		icp7_out = concatenate(inputs=[icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

		icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)
		icp8_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp8_reduction1')(icp8_in)
		icp8_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp8_out1')(icp8_reduction1)
		icp8_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp8_reduction2')(icp8_in)
		icp8_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp8_out2')(icp8_reduction2)
		icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)
		icp8_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp8_out3')(icp8_pool)
		icp8_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp8_out0')(icp8_in)
		icp8_out = concatenate(inputs=[icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

		icp9_reduction1 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp9_reduction1')(icp8_out)
		icp9_out1 = Conv2D(384, (3, 3), padding='same', activation='relu',   name='icp9_out1')(icp9_reduction1)
		icp9_reduction2 = Conv2D(48, (1, 1), padding='same', activation='relu',   name='icp9_reduction2')(icp8_out)
		icp9_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp9_out2')(icp9_reduction2)
		icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)
		icp9_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp9_out3')(icp9_pool)
		icp9_out0 = Conv2D(384, (1, 1), padding='same', activation='relu',   name='icp9_out0')(icp8_out)
		icp9_out = concatenate(inputs=[icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')
		
		
# 	# creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
# 	#####################################################################
# 	# MITESH
# 	# https://www.tensorflow.org/tutorials/using_gpu 
# 	# /gpu:0 should be used
# 	# https://github.com/kentsommer/keras-posenet/issues/1 : as per this note it will pick up GPU is there is one
# 	#####################################################################
# 	with tf.device('/gpu:0'):
# 		input = Input(shape=(224, 224, 3))

# 		conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu',  name='conv1')(input)
# 		pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
# 		norm1 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm1')(pool1)
# 		reduction2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='reduction2')(norm1)
# 		conv2 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='conv2')(reduction2)
# 		norm2 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm2')(conv2)
# 		pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
# 		icp1_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp1_reduction1')(pool2)
# 		icp1_out1 = Conv2D(128, (3, 3), padding='same', activation='relu',   name='icp1_out1')(icp1_reduction1)
# 		icp1_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp1_reduction2')(pool2)
# 		icp1_out2 = Conv2D(32, (5, 5), padding='same', activation='relu',   name='icp1_out2')(icp1_reduction2)
# 		icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)
# 		icp1_out3 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp1_out3')(icp1_pool)
# 		icp1_out0 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp1_out0')(pool2)
# 		icp2_in = concatenate(inputs=[icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

# 		icp2_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_reduction1')(icp2_in)
# 		icp2_out1 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='icp2_out1')(icp2_reduction1)
# 		icp2_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp2_reduction2')(icp2_in)
# 		icp2_out2 = Conv2D(96, (5, 5), padding='same', activation='relu',   name='icp2_out2')(icp2_reduction2)
# 		icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
# 		icp2_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp2_out3')(icp2_pool)
# 		icp2_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_out0')(icp2_in)
# 		icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

# 		icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
# 		icp3_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp3_reduction1')(icp3_in)
# 		icp3_out1 = Conv2D(208, (3, 3), padding='same', activation='relu',   name='icp3_out1')(icp3_reduction1)
# 		icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp3_reduction2')(icp3_in)
# 		icp3_out2 = Conv2D(48, (5, 5), padding='same', activation='relu',   name='icp3_out2')(icp3_reduction2)
# 		icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
# 		icp3_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp3_out3')(icp3_pool)
# 		icp3_out0 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp3_out0')(icp3_in)
# 		icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

# 		icp4_reduction1 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp4_reduction1')(icp3_out)
# 		icp4_out1 = Conv2D(224, (3, 3), padding='same', activation='relu',   name='icp4_out1')(icp4_reduction1)
# 		icp4_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp4_reduction2')(icp3_out)
# 		icp4_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp4_out2')(icp4_reduction2)
# 		icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)
# 		icp4_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp4_out3')(icp4_pool)
# 		icp4_out0 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp4_out0')(icp3_out)
# 		icp4_out = concatenate(inputs=[icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

# 		icp5_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_reduction1')(icp4_out)
# 		icp5_out1 = Conv2D(256, (3, 3), padding='same', activation='relu',   name='icp5_out1')(icp5_reduction1)
# 		icp5_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp5_reduction2')(icp4_out)
# 		icp5_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp5_out2')(icp5_reduction2)
# 		icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)
# 		icp5_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp5_out3')(icp5_pool)
# 		icp5_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_out0')(icp4_out)
# 		icp5_out = concatenate(inputs=[icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

# 		icp6_reduction1 = Conv2D(144, (1, 1), padding='same', activation='relu',   name='icp6_reduction1')(icp5_out)
# 		icp6_out1 = Conv2D(288, (3, 3), padding='same', activation='relu',   name='icp6_out1')(icp6_reduction1)
# 		icp6_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp6_reduction2')(icp5_out)
# 		icp6_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp6_out2')(icp6_reduction2)
# 		icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)
# 		icp6_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp6_out3')(icp6_pool)
# 		icp6_out0 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp6_out0')(icp5_out)
# 		icp6_out = concatenate(inputs=[icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

# 		icp7_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp7_reduction1')(icp6_out)
# 		icp7_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp7_out1')(icp7_reduction1)
# 		icp7_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp7_reduction2')(icp6_out)
# 		icp7_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp7_out2')(icp7_reduction2)
# 		icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)
# 		icp7_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp7_out3')(icp7_pool)
# 		icp7_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp7_out0')(icp6_out)
# 		icp7_out = concatenate(inputs=[icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

# 		icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)
# 		icp8_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp8_reduction1')(icp8_in)
# 		icp8_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp8_out1')(icp8_reduction1)
# 		icp8_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp8_reduction2')(icp8_in)
# 		icp8_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp8_out2')(icp8_reduction2)
# 		icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)
# 		icp8_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp8_out3')(icp8_pool)
# 		icp8_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp8_out0')(icp8_in)
# 		icp8_out = concatenate(inputs=[icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

# 		icp9_reduction1 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp9_reduction1')(icp8_out)
# 		icp9_out1 = Conv2D(384, (3, 3), padding='same', activation='relu',   name='icp9_out1')(icp9_reduction1)
# 		icp9_reduction2 = Conv2D(48, (1, 1), padding='same', activation='relu',   name='icp9_reduction2')(icp8_out)
# 		icp9_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp9_out2')(icp9_reduction2)
# 		icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)
# 		icp9_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp9_out3')(icp9_pool)
# 		icp9_out0 = Conv2D(384, (1, 1), padding='same', activation='relu',   name='icp9_out0')(icp8_out)
# 		icp9_out = concatenate(inputs=[icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

		cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)
		cls3_fc1_flat = Flatten()(cls3_pool)
		cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
		dropout3 = Dropout(0.5)(cls3_fc1_pose)
		cls3_fc_pose_xy = Dense(2, name='cls3_fc_pose_xy')(dropout3)
		cls3_fc_pose_yaw = Dense(1, name='cls3_fc_pose_yaw')(dropout3)

		posenet = Model(
			outputs=[cls3_fc_pose_xy, cls3_fc_pose_yaw], inputs=input)

	if tune:
		if weights_path:
			weights_data = np.load(weights_path, encoding='latin1').item()
			# weights_data = np.load(weights_path).item()
			for layer in posenet.layers:
				if layer.name in weights_data.keys():
					layer_weights = weights_data[layer.name]
					layer.set_weights((layer_weights['weights'], layer_weights['biases']))
					#print("FINISHED SETTING THE WEIGHTS!")
	return posenet


if __name__ == "__main__":
	print("Please run either test.py or train.py to evaluate or fine-tune the network!")
