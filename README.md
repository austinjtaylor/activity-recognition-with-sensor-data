# Human Activity Recognition (HAR) using LSTM RNNs on accelerometer and gyroscopic data
This repository shows how to classify human activities from sequences of accelerometer and gyroscopic data from a smartphone using various types of Long Short-Term Memory (LSTM) recurrent neural networks.

## Get the data
The dataset can be downloaded from [Human Activity Recognition Using Smartphones Data Set, UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

Click here for the direct link: [UCI HAR Dataset.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

Unzip all files into a new directory in your current working directory. You should have a folder titled `UCI HAR Dataset`.

Download the `uci_har.py` and `models.py` files from this repository and move them into the directory containing the `UCI HAR Dataset` folder.

## Usage
The `models.py` contains implementations of a standard LSTM, a Convolutional Neural Network (CNN) that feeds into an LSTM, and a Convolutional LSTM. The difference between the CNN-LSTM and the ConvLSTM is that the CNN-LSTM uses CNN layers for feature extraction on input data and feeds the extracted features into an LSTM layer to support sequence prediction, while the ConvLSTM uses convolutions directly as part of reading input into the LSTM units themselves.

Change the `model_type` on line 74 in `uci_har.py` to 'lstm', 'cnnlstm', or 'convlstm' and run the program. You should see an output similar to the following:
```
Using TensorFlow backend.
X train shape: (7352, 128, 9), y train shape: (7352, 1)
X test shape: (2947, 128, 9), y test shape: (2947, 1)
After one hot encoding, X train shape: (7352, 128, 9), y train shape: (7352, 6), X test shape: (2947, 128, 9), y test shape: (2947, 6)
Using LSTM
Accuracy: 0.9006
```
Change the number of lstm units, number of dense units, number of filters, and sizes of filters in the `models.py` file to experiment with various model architectures, and change the number of epochs, batch size, n_steps, and n_length in the `uci_har.py` file to experiment with various training configurations.
