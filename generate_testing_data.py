'''
visualize images of tfrecord and h5py
'''
import sys
import numpy as np
import cv2
import pickle
# %matplotlib inline
# import matplotlib.pyplot as plt
import random
import h5py


input_data_file = '/ssd/data/fxpal_dataset/posenetDataset/h5py/03_rotated.h5py'

f = h5py.File(input_data_file, "r")
number_of_data = len(f['images'])
print('number of images: {}'.format(number_of_data))

fwrite = open('test_images_new/groundTruth.csv', 'w')
for idx in range(0, number_of_data):
    if idx %100 == 0: #1 or idx == 499 or idx == 998:
        flat_image = f['images'][idx]
        augmented_image = np.reshape(flat_image, (224,224,3)) 
        str = 'test_images_new/' + ('%02d_image_save.png' % idx)
        fwrite.write(str)
        fwrite.write(',')
#         temp.append(str)
#         print('saving data: {}'.format(str))
#         print('label: {}'.format(np.squeeze(f['poses'][idx])))
        for idx, val in enumerate(f['poses'][idx]):
#             temp.append(val)
            value = "%.5f" % val
            fwrite.write(value)
            if idx == 6:
                fwrite.write('\n')
            else:
                fwrite.write(',')
        
        cv2.imwrite(str, augmented_image)
# f.close()
fwrite.close()
