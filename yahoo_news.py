import glob
import sys
import pickle
import MeCab
import numpy as np
# step2 : frequencyを計算
def frequency():
  m = MeCab.Tagger('-Owakati')
  term_freq = {}
  for name in glob.glob("../pkls/*.pkl"):
    print( name )
    for context in pickle.loads( open(name, 'rb').read() ):
      for term in m.parse( context ).strip().split() :
        if term_freq.get( term ) is None : 
          term_freq[term] = 0
        term_freq[term] += 1
  open('term_freq.pkl', 'wb').write( pickle.dumps(term_freq) )

# 単語を制限して、8000語程度のボキャブラリに収める
def distinct_term():
  term_freq = pickle.loads( open('term_freq.pkl', 'rb').read() )
 
  dterm_freq = {}
  for e, (term, freq) in enumerate( sorted( term_freq.items(), key=lambda x:x[1]*-1 ) ):
    dterm_freq[term] = freq
    if e >= 16000-1:
      break

  for e, (term, freq) in enumerate( dterm_freq.items() ):
    print( e, term, freq )

  dterm_index = {}
  for e, (term, freq) in enumerate( dterm_freq.items() ):
    dterm_index[term] =  e
  open('dterm_index.pkl', 'wb').write( pickle.dumps(dterm_index) )
  


# step3 : データ・セット作成する
def step3():
  dterm_index = pickle.loads( open('dterm_index.pkl', 'rb').read() )
  xxx_index   = len(dterm_index)
  m = MeCab.Tagger('-Owakati')
  data_buff = []; counter = 0
  WINDOW = 15
  for eg, name in enumerate( glob.glob('../pkls/*') ):
    print( eg, name )

    
    for ep, context in enumerate( pickle.loads( open(name, 'rb').read() )):
       terms = m.parse(context).strip().split()
       heads = list( map(lambda x:x if dterm_index.get(x) is not None else 'XXX', terms[:WINDOW]) )
       terms = list( map(lambda x:x if dterm_index.get(x) is not None else 'XXX', terms[WINDOW:]) )
       
       for i in range(0, len(terms)-WINDOW, 1):
         print(terms[i:i+WINDOW], terms[i+WINDOW] )
         head_id = list( map(lambda x:dterm_index[x]  if dterm_index.get(x) is not None else xxx_index, heads ) )
         term_id = list( map(lambda x:dterm_index[x] if dterm_index.get(x) is not None else xxx_index, terms[i:i+WINDOW]) )
         ans_id  = dterm_index[terms[i+WINDOW]] if dterm_index.get(terms[i+WINDOW]) else xxx_index
         X1 = np.zeros( (WINDOW, 16000+1) )
         X2 = np.zeros( (WINDOW, 16000+1) )
         Y  = np.zeros( (16000+1) )
         for i, hi in enumerate(head_id):
           X1[i, hi] = 1.0
         for i, ti in enumerate(term_id):
           X2[i, ti] = 1.0
         Y[ans_id] = 1.0
         #print(head_id, term_id, ans_id)
         data_buff.append( (X1, X2, Y) )
         if len(data_buff) >= 1000:
           X1s = np.array( list( map(lambda x:x[0], data_buff) ) )
           X2s = np.array( list( map(lambda x:x[1], data_buff) ) )
           Ys  = np.array( list( map(lambda x:x[2], data_buff) ) )

           open('dataset/%09d.pkl'%counter, 'wb').write( pickle.dumps( (X1s, X2s, Ys) ) )
           # reset
           data_buff = []; counter += 1

if __name__ == '__main__':
  if '--step1' in sys.argv:
    frequency()
  if '--step2' in sys.argv:
    distinct_term()
  if '--step3' in sys.argv:
    step3()
