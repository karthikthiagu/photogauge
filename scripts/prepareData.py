import os
import cv2
import h5py
import numpy as np

images = {0 : [], 1 : []}
for folder in os.listdir('./data/processed'):
    print folder
    folder_path = os.path.join('./data/processed', folder)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, 0)
        if 'bad' in folder:
            images[0].append(image)
        if 'good' in folder:
            images[1].append(image)

np.random.seed(0)
np.random.shuffle(images[0])
np.random.shuffle(images[1])

train = {0 : [], 1 : []}
valid = {0 : [], 1 : []}
test =  {0 : [], 1 : []}

for i in [0, 1]:
    train[i] += images[i][ : 100]
    valid[i] += images[i][100 : 150]
    test[i]  += images[i][150 : 250]  

train_images = np.array(train[0] + train[1], np.float32).reshape(-1, 1, 300, 300) / 255.0
train_labels = np.concatenate([np.zeros(100), np.ones(100)])
print np.min(train_images), np.max(train_images)
print train_images.shape
print train_labels.shape

valid_images = np.array(valid[0] + valid[1], np.float32).reshape(-1, 1, 300, 300) / 255.0
valid_labels = np.concatenate([np.zeros(50), np.ones(50)])
print np.min(valid_images), np.max(valid_images)
print valid_images.shape
print valid_labels.shape

test_images = np.array(test[0] + test[1], np.float32).reshape(-1, 1, 300, 300) / 255.0
test_labels = np.concatenate([np.zeros(100), np.ones(100)])
print np.min(test_images), np.max(test_images)
print test_images.shape
print test_labels.shape

train_file = h5py.File('./data/input/train.h5', 'w')
train_file.create_dataset(data = train_images, dtype = train_images.dtype, shape = train_images.shape, name = 'images')
train_file.create_dataset(data = train_labels, dtype = train_labels.dtype, shape = train_labels.shape, name = 'labels')

valid_file = h5py.File('./data/input/valid.h5', 'w')
valid_file.create_dataset(data = valid_images, dtype = valid_images.dtype, shape = valid_images.shape, name = 'images')
valid_file.create_dataset(data = valid_labels, dtype = valid_labels.dtype, shape = valid_labels.shape, name = 'labels')

test_file = h5py.File('./data/input/test.h5', 'w')
test_file.create_dataset(data = test_images, dtype = test_images.dtype, shape = test_images.shape, name = 'images')
test_file.create_dataset(data = test_labels, dtype = test_labels.dtype, shape = test_labels.shape, name = 'labels')

