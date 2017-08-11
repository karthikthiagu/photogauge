import os
import json
import h5py
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = '3'
set_session(tf.Session(config=config))

from keras.layers import Input, Reshape, Flatten, Dropout, Dense, Conv2D, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint

def getModel(mname, wname, finetune = False):
    input_img = Input(shape = (1, 300, 300))

    if finetune == True:
        json_file = open(mname, 'r')
        loaded_json_model = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_json_model)
        classifier.load_weights(wname)
    else:
        x = Conv2D(16, (3, 3), strides = (2, 2), padding = 'same', data_format = 'channels_first', name = 'conv1')(input_img)
	x = Activation('relu')(x)
	x = Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', data_format = 'channels_first', name = 'conv2')(x)
	x = Activation('relu')(x)
	x = Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', data_format = 'channels_first', name = 'conv3')(x)
	x = Activation('relu')(x)
	x = Conv2D(16, (3, 3), strides = (2, 2), padding = 'same', data_format = 'channels_first', name = 'conv4')(x)
	x = Activation('relu')(x)
	x = Conv2D(8, (3, 3), strides = (2, 2), padding = 'same', data_format = 'channels_first', name = 'conv5')(x)
        x = Flatten(name = 'flatten')(x)
	x = Dense(100, activation = 'relu', name = 'fc-1')(x)
        x = Dense(50, activation = 'relu', name = 'fc-2')(x)
        y = Dense(1, activation = 'sigmoid', name = 'sigmoid')(x)
        classifier = Model(input_img, y)

    adam = Adam(lr = 0.001)
    classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    print classifier.summary() 
    print 'Model has %s parameters' % classifier.count_params()
    print '-----end-----'
    return classifier

def getDataStatistics(data):
    labels = data['labels'][:]
    zero, one = 0, 0
    for label in labels:
        if label == 0:
            zero += 1
        if label == 1:
            one += 1
    print 'number of points in "0" = %s, number of points in "1" = %s' % (zero, one)
    print 'total number of points = %s' % str(zero + one)

def getData(trainname, validname, testname):
    print '-----getData(%s, %s)-----' % (trainname, testname)
    train = h5py.File(trainname, 'r')
    #getDataStatistics(train)
    valid = h5py.File(validname, 'r')
    print valid['images'][:].shape, valid['labels'][:].shape
    #getDataStatistics(valid)
    test =  h5py.File(testname, 'r')
    #getDataStatistics(test)

    print '-----end-----'
    return ((train['images'][:], train['labels'][:]), (valid['images'][:], valid['labels'][:]), (test['images'][:], test['labels'][:]))

def trainModelOnData(train_data, valid_data, model):
    print '-----trainModelOnData(data, model)-----'
    train_images, train_labels = train_data
    valid_images, valid_labels = valid_data
    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20),\
		 ModelCheckpoint('./models/karthik/weights_updated.h5', monitor = 'val_loss', save_best_only = True, save_weights_only = True)]
    model.fit(train_images, train_labels,\
                    epochs = 100,\
                    batch_size = 32,\
                    shuffle = True,\
                    validation_data = (valid_images, valid_labels),\
		    callbacks = callbacks)
    print '-----end-----'
    return model

def saveModel(model, mname, wname):
    print '-----saveModel(model, %s, %s)-----' % (mname, wname)
    model_json = model.to_json()
    with open(mname, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(wname)
    print '-----end-----'

def loadModel(mname, wname):
    json_file = open(mname, 'r')
    loaded_json_model = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_json_model)
    classifier.load_weights(wname)
    return classifier
    print '-----loadModel(model, %s, %s)----' % (mname, wname)

def evaluate(data, model):
    print '-----getPrediction(data, model)-----'
    print '-----end-----'
    images, labels = data
    result = model.evaluate(images, labels, batch_size = labels.shape[0])
    print result

def test_on_batch(data, model):
    images, labels = data
    result = model.evaluate(images, labels)
    print result

def confusionMatrix(labels, predictions):
    conmat = np.zeros((2, 2))
    for i in range(labels.shape[0]):
        conmat[int(labels[i]), int(predictions[i])] += 1
    print '##########'
    print '0\t' + '\t'.join([str(i) for i in range(2)])
    for i in range(2):
        row = []
        for j in range(2):
            row.append(str(int(conmat[i, j])))
        print str(i) + '\t' + '\t'.join(row)

def manualEvaluationBinary(data, model):
    images, labels = data
    predictions = model.predict(images).flatten()
    probabilities = predictions.copy()
    size = predictions.shape[0]
    for i in range(size):
        if predictions[i] < 0.5:
            predictions[i] = 0
        else:
            predictions[i] = 1
    accuracy = np.sum(np.equal(labels, predictions)) / float(predictions.shape[0])
    # for class "0"
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(size):
        if (predictions[i] == labels[i]) and predictions[i] == 0:
            tp += 1
        elif (predictions[i] == labels[i]) and predictions[i] != 0:
            tn += 1
        elif (predictions[i] != labels[i]) and predictions[i] == 0:
            fp += 1
        elif (predictions[i] != labels[i]) and predictions[i] == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision*recall / (precision + recall)
    print 'number of elements in class "0" : %s' % str(tp + fn)
    print 'number of elements in class "1" : %s' % str(fp + tn)
    print 'total class size : %s' % size
    print 'For class "0" : accuracy = %s, precision = %s, recall = %s, fscore = %s' % (accuracy, precision, recall, fscore)
    confusionMatrix(labels, predictions)

def train(train_data, valid_data):
    classifier = getModel('./models/karthik/model.json',\
                          './models/karthik/weights.h5',\
			   finetune = False)
    classifier = trainModelOnData(train_data, valid_data, classifier)
    saveModel(classifier, './models/karthik/model_updated.json',\
                          './models/karthik/weights_final.h5')

def test(data, mname, wname):
    images, labels = data[0], data[1]
    classifier = loadModel(mname, wname)
    adam = Adam()
    classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    evaluate([images, labels], classifier)
    manualEvaluationBinary([images, labels], classifier)

def projectData(data, mname, wname, layer, fname):
    images, labels = data[0], data[1]
    classifier = loadModel(mname, wname)
    feature_extractor = Model(input = classifier.input, output = classifier.get_layer(layer).output)
    features = feature_extractor.predict(images)
    print features.shape
    h5file = h5py.File(fname, 'w')
    h5file.create_dataset('features', shape = features.shape, dtype = features.dtype, data = features)
    h5file.create_dataset('labels', shape = labels.shape, dtype = labels.dtype, data = labels)   
    h5file.close()

if __name__ == '__main__':
    train_data, valid_data, test_data = getData('data/input/train.h5',\
                                                'data/input/valid.h5',\
                                                'data/input/test.h5')

    train(train_data, valid_data)

    test(test_data,\
	 './models/karthik/model_updated.json',\
	 './models/karthik/weights_updated.h5')
    projectData(test_data,\
	 './models/karthik/model_updated.json',\
	 './models/karthik/weights_updated.h5',\
         'fc-1',\
         './models/karthik/features.h5')

