###########################################
#########                         #########
#####                                 #####
###             Hello World!            ###
###          Author: efecanxrd          ###
#####                                 #####
#########                         #########
###########################################

#Modüllerimizi tanımlayalım
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

#The Model
class DeepLearningModel():
    def __init__(self, data: pd.DataFrame, Y_var: str,lag: int,LSTM_layer_depth: int,epochs=int,batch_size=256,train_test_split=0): 
        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        X, Y = [], []
        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, Y         

    def create_data_for_NN(
        self,
        use_last_n=None
        ):
        y = self.data[self.Y_var].tolist()
        if use_last_n is not None:
            y = y[-use_last_n:]
        X, Y = self.create_X_Y(y, self.lag)
        X_train = X
        X_test = []
        Y_train = Y
        Y_test = []
        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)
            X_train = X[:(len(X) - index)]
            X_test = X[-index:]     
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]
        return X_train, X_test, Y_train, Y_test

    def LSTModel(self):
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()
        model = Sequential()
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        keras_dict = {'x': X_train,'y': Y_train,'batch_size': self.batch_size,'epochs': self.epochs,'shuffle': False}
        if self.train_test_split > 0:
            keras_dict.update({'validation_data': (X_test, Y_test)})
        model.fit(**keras_dict)
        self.model = model
        return model

    def predict(self) -> list:
        yhat = []
        if(self.train_test_split > 0):
            _, X_test, _, _ = self.create_data_for_NN()        
            yhat = [y[0] for y in self.model.predict(X_test)]
        return yhat

    def predict_n_ahead(self, n_ahead: int):
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)        
        yhat = []
        for _ in range(n_ahead):
            fc = self.model.predict(X)
            yhat.append(fc)
            X = np.append(X, fc)
            X = np.delete(X, 0)
            X = np.reshape(X, (1, len(X), 1))
        return yhat    
