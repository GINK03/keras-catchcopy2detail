
import pickle
fterm_index = pickle.loads( open('dterm_index.pkl', 'rb').read() )
index_fterm = { index:fterm for fterm, index in fterm_index.items() }
def recover(x1s, x2s, prs):
  
  for e, (xs, pr) in enumerate(zip(x2s, prs)):
    if e > 10:
      break
    text = ''
    for x in xs:
      index, weight = max([(i,x) for i,x in enumerate(x)], key=lambda x:x[1])
      text += index_fterm[index] if index_fterm.get(index) is not None else 'XXX'
    index, weight = max([(i,x) for i,x in enumerate(pr)], key=lambda x:x[1])
    predict = index_fterm[index] if index_fterm.get(index) is not None else 'XXX'

    print(text, predict)

    

if __name__ == '__main__':
  import glob
  import os
  import math
  for name in glob.glob('./dataset/000000000.pkl'):
    x1s, x2s, ys = pickle.loads( open(name, 'rb').read() )
    recover(x1s, x2s, ys)
