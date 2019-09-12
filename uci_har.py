######## Human Activity Recognition Using RNNs on Accelerometer and Gyroscopic Data #########
#
# Author: Austin Taylor
# Date: 2/6/19
# Description: 
# This program loads the UCI Human Activity Recognition Using Smartphones Data Set, fits the data to an RNN model, and evaluates the model

## This code has been modified from Jason Brownlee's example at
## https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/


# Imports
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
from models import Models
 
# Load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# Load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded
 
# Load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
        
# Load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
	print('X train shape: {}, y train shape: {}'.format(trainX.shape, trainy.shape))
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
	print('X test shape: {}, y test shape: {}'.format(testX.shape, testy.shape))
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print('After one hot encoding, X train shape: {}, y train shape: {}, X test shape: {}, y test shape: {}'.format(trainX.shape, trainy.shape, testX.shape, testy.shape))
	return trainX, trainy, testX, testy

# Fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	epochs, batch_size = 15, 64
	verbose, n_steps, n_length = 0, 4, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = 'lstm'
	if model == 'cnnlstm':
		trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
		testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	elif model == 'convlstm':
		trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
		testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
	model = Models(model, n_timesteps, n_features, n_outputs, n_steps, n_length)
	model.model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
	print('Accuracy: {:.4f}'.format(accuracy))

# Execute the program
def main():
    trainX, trainy, testX, testy = load_dataset()
    evaluate_model(trainX, trainy, testX, testy)

if __name__ == '__main__':
    main()
