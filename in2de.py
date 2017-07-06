from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, dot, multiply, concatenate, add, Activation
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
import utils
import time
WIDTH       = 16000+1
inputs_1    = Input( shape=(15, WIDTH) ) 
encoded_1   = Bi( GRU(256, kernel_initializer='lecun_uniform', activation='selu', return_sequences=True) )(inputs_1)
att_1       = TD( Dense(256, kernel_initializer='lecun_uniform', activation='selu') )( encoded_1 )

inputs_2    = Input( shape=(15, WIDTH) )
encoded_2   = Bi( GRU(256, kernel_initializer='lecun_uniform',activation='selu', return_sequences=True) )(inputs_2)
att_2       = TD( Dense(256, kernel_initializer='lecun_uniform', activation='selu') )( encoded_2 )

conc        = concatenate( [att_1, att_2] )

conced      = Bi( GRU(768, kernel_initializer='lecun_uniform', activation='selu', return_sequences=True) )( conc )
conced      = TD( Dense(768, kernel_initializer='lecun_uniform', activation='selu') )( conced )
conced      = Flatten()( conced )
next_term   = Dense(16000+1, activation='softmax')( conced )

in2de       = Model([inputs_1, inputs_2], next_term)
in2de.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
  with open('logs/loss_%s.log'%now, 'a+') as f:
    f.write('%s\n'%str(buff))
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )

def train():
  count = 0
  try:
    to_load = sorted(glob.glob('models/*.h5') ).pop() 
    in2de.load_weights( to_load )
    count = int( re.search(r'(\d{1,})', to_load).group(1) )
  except Exception as e:
    print( e )
  while True:
    for name in sorted( glob.glob(os.getenv("HOME") + '/sda/dataset/*.pkl') ):
      print('will deal this data', name)
      print('now count is', count)
      X1s, X2s, Ys = pickle.loads( open(name, 'rb').read() ) 
      inner_loop = 0
      while True:
        in2de.fit( [X1s, X2s], Ys, epochs=1, validation_split=0.1,callbacks=[batch_callback] )
        print(buff)
        if count < 10 and buff['loss'] < 1.25:
          break
        if count < 20 and buff['loss'] < 1.00:
          break
        if count < 30 and buff['loss'] < 0.80:
          break
        if count < 40 and buff['loss'] < 0.50:
          break
        if buff['loss'] < 0.30:
          break
        if inner_loop > 50:
          break
        inner_loop += 1
      pr = in2de.predict( [X1s, X2s] )
      utils.recover(X1s.tolist(), X2s.tolist(), pr.tolist()) 
      if count%5 == 0:
        in2de.save_weights('models/%09d.h5'%count)
      count += 1

def predict():
  to_load = sorted(glob.glob('models/*.h5') ).pop() 
  in2de.load_weights( to_load )
  
  for name in sorted( glob.glob('/home/gimpei/sda/dataset/*.pkl') ):
    print('will deal this data', name)
    print('now count is', count)
    X1s, X2s, Ys = pickle.loads( open(name, 'rb').read() ) 
    pr = in2de.predict( [X1s, X2s] )
    utils.recover(X1s.tolist(), X2s.tolist(), pr.tolist()) 

if __name__ == '__main__':
  if '--make_dataset' in sys.argv:
    make_dataset()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
