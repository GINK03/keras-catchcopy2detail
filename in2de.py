from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, dot, multiply, concatenate, add
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import json
import glob
import copy
import os
import re

WIDTH       = 16000+1
inputs_1    = Input( shape=(15, WIDTH) ) 
encoded     = Bi( GRU(512) )(inputs_1)
encoder     = Model(inputs_1, encoded)
att_1       = RepeatVector(15)(encoded)

inputs_2    = Input( shape=(15, WIDTH) )
encoded_2   = Bi( GRU(512) )(inputs_2)
att_2       = RepeatVector(15)(encoded_2)

conc        = concatenate( [att_1, att_2] )

conced      = Bi( GRU(512, return_sequences=False) )( conc )
next_term   = Dense(16000+1, activation='softmax')( conced )

in2de       = Model([inputs_1, inputs_2], next_term)
in2de.compile(optimizer=Adam(), loss='categorical_crossentropy')


def train():
  for name in sorted( glob.glob('dataset/*.pkl') ):
    X1s, X2s, Ys = pickle.loads( open(name, 'rb').read() ) 
    in2de.fit( [X1s, X2s], Ys, epochs=100 )
  ...
def predict():
  ...
if __name__ == '__main__':
  if '--make_dataset' in sys.argv:
    make_dataset()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
