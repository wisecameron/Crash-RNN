import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd


class CrashRNN():
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def create_model(self):
        model = Sequential()
        
        model.add(LSTM(128, input_shape = (self.x_train.shape[1:]),
                       activation = 'relu',
                       return_sequences = True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(128, activation = 'relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation = 'relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(10, activation = 'softmax'))
        
        opt = tf.keras.optimizers.Adam(learning_rate = 1e-3, decay = 1e-5)
        
        model.compile(loss = 'sparse_categorical_crossentropy',
                      optimizer = opt,
                      metrics = ['accuracy'])
        
        model.fit(self.x_train, self.y_train,
                  epochs = 4, validation_data = (self.x_test, self.y_test))
        score = model.evaluate(self.x_test, self.y_test)
        predictions = model.predict(self.x_test)
        model.save("fixed_model")
        predictions = np.argmax(predictions, axis = 1)
        label = np.argmax(self.y_test, axis = 1)
        count =0
        for a in predictions:
            if (a== 1):
                count+=1
        print(count, " total len == ", len(predictions))
        
        
        