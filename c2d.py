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


inputs_1    = Input( shape=(100, 1024*3)) 
encoded     = GRU(256)(inputs_1)
encoder     = Model(inputs_1, encoded)
att_1       = RepeatVector(25)(encoded)

inputs_2    = Input( shape=(25, 1024*3) )
encoded_2   = GRU(256)(inputs_2)
att_2       = RepeatVector(25)(encoded_2)

conc        = concatenate( [att_1, att_2] )

conced      = GRU(512, return_sequences=False)( conc )
shot        = Dense(1024*3, activation='softmax')( conced )

c2d         = Model([inputs_1, inputs_2], shot)
c2d.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)

class PD:
  def __init__(self):
    self.xs1 = []
    self.xs2 = []
    self.ys  = []
    self.title   = ""
    self.context = ""
    self.ans     = ""
  def getX(self):
    return [ np.array([self.xs1]), np.array([self.xs2]) ] 

def make_dataset():
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c           = { i:c for c,i in c_i.items() }
  files         = glob.glob("dataset/*.pkl")
  random.shuffle( files ) 
  for ef, filename in enumerate(files):
    if "c_i.pkl" in filename:
      continue
    title   = re.search(r"/(.*?).pkl", filename).group(1)
    dataset = pickle.loads( open(filename, "rb").read() )
    print( ef, title )
    #if ef > 30 :
    #  break
    for di, (context, ans) in enumerate(dataset):
      if di > 350:
        break
      key       = "{title}_{num}".format(title=title, num=di)
      xs1 = [ [0.]*(1024*3) for _ in range(100) ] 
      xs2 = [ [0.]*(1024*3) for _ in range(25) ] 
      ys  =   [0.]*(1024*3)
      if c_i.get(ans) is None:
        continue
      for e,c in enumerate(list(title)):
        try:
          if c_i.get(c) is not None:
            xs1[e][c_i[c]] = 1.
        except IndexError as e:
          print( e )
          continue
      for e,c in enumerate(context):
        try:
          if c_i.get(c) is not None:
            xs2[e][c_i[c]] = 1.
        except IndexError as e:
          print( e )
          continue
      
      ys[c_i[ans]] = 1.

      key       = "{title}_{num}".format(title=title, num=di)
      print( ef, key )
      pd        = PD()
      pd.xs1    = np.array( list(reversed(xs1)) )
      pd.xs2    = np.array( xs2 )
      pd.ys     = np.array( ys )
      pd.title  = title
      pd.context= "".join(context)
      pd.ans    = ans
      open("/mnt/sda/train_dataset/{}.pkl".format( key ), "wb").write( pickle.dumps(pd) )

def train():
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c           = { i:c for c,i in c_i.items() }
  keys          = []
  files         = glob.glob('/mnt/sda/train_dataset/*.pkl')
  for ed, key in enumerate(files):
    keys.append( key )
  random.shuffle( keys ) 
  xss1 = []
  xss2 = []
  yss  = []
  contexts = []

  for CROP in range(0, len(keys), 1024*5):
    for ek, key in enumerate( keys[CROP:CROP+1024*5] ):
      print( ek )
      try:
        pd = pickle.loads( open(key, "rb").read() )
      except EOFError as e:
        print( e )
        continue
      xss1.append( pd.xs1 )
      xss2.append( pd.xs2 )
      yss.append( pd.ys )
      contexts.append( (pd.title, pd.context, pd.ans)  ) 
    Xs1  = np.array( xss1 )
    Xs2  = np.array( xss2 )
    Ys   = np.array( yss )
    
    """ startインデックス """
    I          = 0
    model      = "start point"
    epoch_rate = json.loads( open("epoch_rate.json").read() )
    try:
      #if '--resume' in sys.argv:
      model = sorted( glob.glob("models/*.h5") ).pop()
      I = int( re.search( r"/(.*?)_", model).group(1) )
      print("loaded model is ", model)
      c2d.load_weights(model)
    except IndexError as e:
      print( e )
      

    delta = random.randint(15,20)
    ind   = 0
    for ind in range(I, I+delta):
      print_callback = LambdaCallback(on_epoch_end=callbacks)
      batch_size = random.randint( 32, 64 )
      lr           = epoch_rate["%d"%ind]
      #random_optim = random.choice( [Adam(lr), SGD(lr*10.), RMSprop(lr)] )
      random_optim = random.choice( [Adam(), SGD(), RMSprop()] )
      print( "optimizer", random_optim )
      print( "learning_rate base", lr )
      print( "now dealing ", model )
      c2d.optimizer = random_optim
      c2d.fit( [Xs1, Xs2], Ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )

      #c2d.fit( Xs2, Ys,  shuffle=False, batch_size=batch_size, epochs=1, callbacks=[print_callback] )

      """ サンプリング """
      ps = c2d.predict( [Xs1[:10], Xs2[:10]] ).tolist()
      for cs, p in zip(contexts[:10], ps):
        ips = [(i,_p) for i, _p in enumerate(p)]
        ip  = max(ips, key=lambda x:x[1])
        i, p = ip
        print( cs )
        print(ip, i_c[i])
    c2d.save("models/%09d_%09f.h5"%(ind, buff['loss']))
    print("saved ..")
    print("logs...", buff )

def predict():
  c_i           = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c           = { i:c for c,i in c_i.items() }
  title_dataset = pickle.loads( open("dataset/title_dataset.pkl", "rb").read() )
  pds = []
  for e, filename in enumerate(glob.glob("dataset/*.pkl")):
    if "c_i.pkl" in filename:
      continue
    title   = re.search(r"/(.*?).pkl", filename).group(1)
    dataset = pickle.loads( open(filename, "rb").read() )
    print( e, title )
    if e > 300 :
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
  if '--make_dataset' in sys.argv:
    make_dataset()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
