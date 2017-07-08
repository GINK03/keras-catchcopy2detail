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
from keras.layers.core     import Dropout 
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
import concurrent.futures
import threading 
WIDTH       = 16000+1
ACTIVATOR   = 'selu'
DO          = Dropout(0.1)
inputs_1    = Input( shape=(15, WIDTH) ) 
encoded_1   = Bi( GRU(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )(inputs_1)
att_1       = TD( Dense(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( encoded_1 )

inputs_2    = Input( shape=(5, WIDTH) )
encoded_2   = Bi( GRU(256, kernel_initializer='lecun_uniform',activation=ACTIVATOR, return_sequences=True) )(inputs_2)
att_2       = TD( Dense(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( encoded_2 )

conc        = DO( concatenate( [att_1, att_2], axis=1 ) )

conced      = Bi( GRU(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( conc )
conced      = TD( Dense(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( conced )
conced      = DO( Flatten()( conced ) )
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


DATASET_POOL = []
def loader():
  while True:
    for name in glob.glob('dataset/*.pkl') :
      while True:
        if len( DATASET_POOL ) >= 1: 
          time.sleep(1.0)
        else:
          break
        
      print('loading data...', name)
      X1s, X2s, Ys = pickle.loads( open(name, 'rb').read() ) 
      X1s          = np.array( [ x.todense() for x in X1s ] )
      X2s          = np.array( [ x.todense() for x in X2s ] )
      print( X2s.shape )
      Ys           = np.reshape( np.array( [ y.todense() for y in Ys  ] ), (2000, 16001) )
      DATASET_POOL.append( (X1s, X2s, Ys, name) )
      print('finish recover from sparse...', name)

def train():
  t = threading.Thread(target=loader, args=())
  t.start()
  count = 0
  try:
    to_load = sorted( glob.glob('models/*.h5') ).pop() 
    in2de.load_weights( to_load )
    count = int( re.search(r'(\d{1,})', to_load).group(1) )
  except Exception as e:
    print( e )
  while True:
    if DATASET_POOL == []:
      print('no buffers so delay some seconds')
      time.sleep(10.)
    else:
      X1s, X2s, Ys, name = DATASET_POOL.pop(0)
      print('will deal this data', name)
      print('now count is', count)
      inner_loop = 0
      while True:
        in2de.fit( [X1s, X2s], Ys, epochs=1, validation_split=0.02,callbacks=[batch_callback] )
        print(buff)
        if count < 10 and buff['loss'] < 3.25:
          break
        if count < 20 and buff['loss'] < 3.00:
          break
        if count < 30 and buff['loss'] < 2.80:
          break
        if count < 40 and buff['loss'] < 2.40:
          break
        if buff['loss'] < 2.00:
          break
        if inner_loop > 30:
          break
        inner_loop += 1
      if count%5 == 0:
        pr = in2de.predict( [X1s, X2s] )
        utils.recover(X1s.tolist()[:100], X2s.tolist()[:100], pr.tolist()[:100]) 
        in2de.save_weights('models/%09d.h5'%count)
      count += 1

def predict():
  to_load = sorted(glob.glob('models/*.h5') ).pop() 
  in2de.load_weights( to_load )
  
  for name in sorted( glob.glob('dataset/*.pkl') ):
    print('will deal this data', name)
    X1s, X2s, Ys = pickle.loads( open(name, 'rb').read() ) 
    X1s          = np.array( [ x.todense() for x in X1s ] )
    X2s          = np.array( [ x.todense() for x in X2s ] )
    Ys           = np.reshape( np.array( [ y.todense() for y in Ys  ] ), (2000, 16001) )
    X1s, X2s, Ys = map(lambda x:x.tolist(), [X1s, X2s, Ys])
    for x1, x2, y in zip(X1s, X2s, Ys):
      """" start to loop """
      x1, x2, y = map(lambda x:np.array([x]) , [x1, x2, y] )
      print('\ntitle', utils.recover_hint_one(x1.tolist()[0]) )
      for i in range(10):
        pr = in2de.predict( [x1, x2] )
        utils.recover(x1.tolist(), x2.tolist(), pr.tolist(), end='') 
        x2          = x2.tolist()[0]
        x2.pop(0)
        pr          = max([(i,w) for i,w in enumerate(pr.tolist()[0])], key=lambda x:x[1])
        base        = [0.0 for i in range(16001)]
        base[pr[0]] = 1.0
        pr          = base
        x2.append( pr )
        x2          = np.array([x2])
    print("split")

if __name__ == '__main__':
  if '--make_dataset' in sys.argv:
    make_dataset()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
