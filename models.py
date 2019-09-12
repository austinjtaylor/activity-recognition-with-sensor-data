# A collection of RNN models we'll use to classify the data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D

class Models():
    def __init__(self, model, n_timesteps, n_features, n_outputs, n_steps, n_length):
        #set defaults
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.n_length = n_length

        #get the appropriate model
        if model == 'lstm':
            print("Using LSTM")
            self.input_shape = (n_timesteps, n_features)
            self.model = self.lstm()
        elif model == 'cnnlstm':
            print("Using CNN-LSTM")
            self.input_shape = (None, n_length, n_features)
            self.model = self.cnnlstm()
        elif model == 'convlstm':
            print("Using ConvLSTM")
            self.input_shape = (n_steps, 1, n_length, n_features)
            self.model = self.convlstm()
        else:
            print("Unknown model")
            sys.exit()

        #compile the network
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

    def lstm(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        return model

    def cnnlstm(self):
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        return model

    def convlstm(self):
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        return model