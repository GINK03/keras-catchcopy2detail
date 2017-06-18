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
import glob
import copy
import os
import re


timesteps   = 100
inputs_1    = Input( shape=(timesteps, 1024*3)) 
encoded     = LSTM(512)(inputs_1)
encoder     = Model(inputs_1, encoded)

att         = RepeatVector(25)(encoded)
inputs_2    = Input( shape=(25, 1024*3) )
#conc        = concatenate( [att, inputs_2] )
conc        = inputs_2
conced      = LSTM(512, return_sequences=False)( conc )
shot        = Dense(1024*3, activation='softmax')( conced )

#c2d         = Model([inputs_1, inputs_2], shot)
c2d         = Model( inputs_2, shot)
c2d.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)

def train():
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c           = { i:c for c,i in c_i.items() }
  xss1 = []
  xss2 = []
  yss  = []
  contexts = []
  for e, filename in enumerate(glob.glob("dataset/*.pkl")):
    if "c_i.pkl" in filename:
      continue
    title   = re.search(r"/(.*?).pkl", filename).group(1)
    dataset = pickle.loads( open(filename, "rb").read() )
    print( e, title )
    if e > 7 :
      break
    for di, (context, ans) in enumerate(dataset):
      if di > 220:
        break
      xs1 = [ [0.]*(1024*3) for _ in range(100) ] 
      xs2 = [ [0.]*(1024*3) for _ in range(25) ] 
      ys  =   [0.]*(1024*3)
      if c_i.get(ans) is None:
        continue
      for e,c in enumerate(list(title)):
        if c_i.get(c) is not None:
          xs1[e][c_i[c]] = 1.
      for e,c in enumerate(context):
        if c_i.get(c) is not None:
          xs2[e][c_i[c]] = 1.
      
      ys[c_i[ans]] = 1.

      xss1.append( list(reversed(xs1)) )
      xss2.append( xs2 )
      yss.append( ys ) 
      contexts.append( ("".join(context), ans) )
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
    #c2d.fit( [Xs1, Xs2], Ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    c2d.fit( Xs2, Ys,  shuffle=False, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    if i%5 == 0:
      c2d.save("models/%9f_%09d.h5"%(buff['loss'], i))
      print("saved ..")
      print("logs...", buff )
    ps = c2d.predict( Xs2[:5] ).tolist()
    for cs, p in zip(contexts[:5], ps):
      ips = [(i,_p) for i, _p in enumerate(p)]
      ip  = max(ips, key=lambda x:x[1])
      i, p = ip
      print( cs )
      print(ip, i_c[i])

def predict():
  class PD:
    def __init__(self):
      self.xs1 = []
      self.xs2 = []
      self.title   = ""
      self.context = ""
      self.ans     = ""
    def getX(self):
      return [ np.array([self.xs1]), np.array([self.xs2]) ] 
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c           = { i:c for c,i in c_i.items() }
  title_dataset = pickle.loads( open("dataset/title_dataset.pkl", "rb").read() )
  pds = []
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
      if c_i.get(ans) is None:
        continue
      for e,c in enumerate(list(title)):
        if c_i.get(c) is not None:
          xs1[e][c_i[c]] = 1.
      for e,c in enumerate(context):
        if c_i.get(c) is not None:
          xs2[e][c_i[c]] = 1.

      pd         = PD()
      pd.xs1     = xs1
      pd.xs2     = xs2
      pd.title   = title
      pd.context = context
      pd.ans     = ans
      pds.append( pd )


  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  c2d.load_weights(model)
  for e, pd in enumerate( pds ):
    p = c2d.predict( pd.getX() ).tolist()[0]
    ips = [(i,_p) for i, _p in enumerate(p)]
    ip  = max(ips, key=lambda x:x[1])
    i, p = ip
    print( pd.context )
    print(ip, i_c[i])
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
