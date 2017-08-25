import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import math


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    number_of_pairs = len(series) - window_size 
    print("Number of pairs: ", number_of_pairs)
    for i in range(number_of_pairs):
        print(i, i + window_size)
        X += [series[i:i + window_size]]
        print(i + window_size)
        y += [series[i + window_size]]
   
    
    
    print("Length of X vector", len(X))
    print(X[0:1])
    
    print("Length of y vector", len(y))
    print(y[0:1])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    model = Sequential()
    # layer 1: LSTM 
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # layer 2: Fully connected
    model.add(Dense(1))
    return model 
    
   


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    vocab = sorted(set(text))
    punctuation = ['!', ',', '.', ':', ';', '?']
    wanted = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    wanted += punctuation
    unwanted = set(vocab) - set(wanted) 
    
   
    for u in unwanted:
        text = text.replace(u,' ')
    
    
    
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    number_of_pairs = math.ceil(len(text)/step_size)
    
    for i in range(number_of_pairs -window_size):
        s = i * step_size
        inputs += [text[s:s+window_size]]
        outputs += [text[s+window_size]]
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    
    model = Sequential()
    
    
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    
    
    
    return model
    
    
    
    
    