from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply, concatenate, add
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re


timesteps   = 100
inputs_1    = Input( shape=(timesteps, 1024*3)) 
encoded     = LSTM(512)(inputs_1)
encoder     = Model(inputs_1, encoded)

x           = RepeatVector(25)(encoded)
inputs_2    = Input( shape=(25, 1024*3) )
conc        = concatenate( [x, inputs_2] )
x           = Bi(LSTM(512, return_sequences=False))( conc )
single      = Dense(1024*3, activation='softmax')(x)

c2d         = Model([inputs_1, inputs_2], single)
c2d.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)

def train():
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  title_dataset = pickle.loads( open("dataset/title_dataset.pkl", "rb").read() )
  xss1 = []
  xss2 = []
  yss  = []

  for e, (title, dataset) in enumerate(title_dataset.items()):
    print( e, title )
    if e > 5 :
      break
    for di, (context, ans) in enumerate(dataset):
      if di > 220:
        break
      xs1 = [ [0.]*(1024*3) for _ in range(100) ] 
      xs2 = [ [0.]*(1024*3) for _ in range(25) ] 
      ys  =   [0.]*(1024*3)
      for e,c in enumerate(list(title)):
        if c_i.get(c) is not None:
          xs1[e][c_i[c]] = 1.
      for e,c in enumerate(context):
        if c_i.get(c) is not None:
          xs2[e][c_i[c]] = 1.
      if c_i.get(ans) is None:
        continue
      
      ys[c_i[ans]] = 1.

      xss1.append( list(reversed(xs1)) )
      xss2.append( xs2 )
      yss.append( ys ) 
  Xs1  = np.array( xss1 )
  Xs2  = np.array( xss2 )
  Ys   = np.array( yss )
  if '--resume' in sys.argv:
    model = sorted( glob.glob("models/*.h5") ).pop(0)
    print("loaded model is ", model)
    c2d.load_weights(model)

  for i in range(2000):
    print_callback = LambdaCallback(on_epoch_end=callbacks)
    batch_size = random.randint( 32, 64 )
    random_optim = random.choice( [Adam(), SGD(), RMSprop()] )
    print( random_optim )
    c2d.optimizer = random_optim
    c2d.fit( [Xs1, Xs2], Ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    if i%5 == 0:
      c2d.save("models/%9f_%09d.h5"%(buff['loss'], i))
      print("saved ..")
      print("logs...", buff )

def predict():
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c           = { i:c for c,i in c_i.items() }
  title_dataset = pickle.loads( open("dataset/title_dataset.pkl", "rb").read() )
  xss1  = []
  xss2  = []
  yss   = []
  xssrs = []
  for e, (title, dataset) in enumerate(title_dataset.items()):
    print( e, title )
    if e > 5 :
      break
    for di, (context, ans) in enumerate(dataset):
      if di > 220:
        break
      xs1 = [ [0.]*(1024*3) for _ in range(100) ] 
      xs2 = [ [0.]*(1024*3) for _ in range(25) ] 
      ys  =   [0.]*(1024*3)
      for e,c in enumerate(list(title)):
        if c_i.get(c) is not None:
          xs1[e][c_i[c]] = 1.
      for e,c in enumerate(context):
        if c_i.get(c) is not None:
          xs2[e][c_i[c]] = 1.
      if c_i.get(ans) is None:
        continue
      try:
        ys[c_i[ans]] = 1.
      except IndexError as e:
        print(e)
       
      xssrs.append( (title, "".join(context), ans) )
      xss1.append( xs1 )
      xss2.append( xs2 )
      yss.append( ys ) 
  Xs1  = np.array( xss1 )
  Xs2  = np.array( xss2 )
  Ys   = np.array( yss )

  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  c2d.load_weights(model)
  for xrs, xs1, xs2 in zip(xssrs, Xs1, Xs2):
    ps = c2d.predict( [ np.array([xs1]), np.array([xs2]) ] )
    print( len(ps.tolist()) )
    print( len(ps.tolist()[0]) )
    ips = [(i,p) for i, p in enumerate(ps.tolist()[0])]
    ip  = max(ips, key=lambda x:x[1])
    i, p = ip
    print( xrs )
    print(ip, i_c[i])
  print( Xs1.shape )
  print( Xs2.shape )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
