import glob
import sys
import pickle
# step1 : タイトル長を計算する
# 結果: 100字ぐらいからでいいと思う
def step1():
  for name in glob.glob("../output/*/*"):
    title = name.split('/').pop()
    print( len(title) )

# step2 : 次元制限
def step2():
  DIM = 1024*3
  c_f = {}
  for name in glob.glob("../output/*/*"):
    title = name.split('/').pop()
    #print( title )
    for c in list( title ):
      if c_f.get(c) is None:
        c_f[c] = 0
      c_f[c] += 1
  c_i = {}
  for e, (c, f) in enumerate(sorted( c_f.items(), key=lambda x:x[1]*-1)):
    if e >= DIM:
      break
    print( e, c, f )
    c_i[c] = e
  open("dataset/c_i.pkl", "wb").write( pickle.dumps( c_i ) ) 

# step3 : データ・セット作成する
def step3():
  WINDOW = 25
  title_dataset = {}
  for e, name in enumerate( glob.glob("../output/*/*") ):
    title = name.split('/').pop()
    print( e, title )
    if e > 50000:
      break
    with open(name) as f:
      """" head, tail padding """
      chars = list("H" * WINDOW + f.read() + "❄️" )
      for i in range( len(chars) - WINDOW ):
        #print( title, chars[i:i+WINDOW], chars[i+WINDOW] )
        if title_dataset.get( title ) is None:
          title_dataset[title] = []
        title_dataset[title].append( (chars[i:i+WINDOW], chars[i+WINDOW]) )  
  open("dataset/title_dataset.pkl", "wb").write( pickle.dumps( title_dataset ) )
if __name__ == '__main__':
  if '--step1' in sys.argv:
    step1()
  if '--step2' in sys.argv:
    step2()
  if '--step3' in sys.argv:
    step3()
