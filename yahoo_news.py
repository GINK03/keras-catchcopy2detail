import glob
import sys

# step1 : タイトル長を計算する
# 結果: 100字ぐらいからでいいと思う
def step1():
  for name in glob.glob("../output/*/*"):
    title = name.split('/').pop()
    print( len(title) )

# step2 : 次元制限
def step2():
  c_f = {}
  for name in glob.glob("../output/*/*"):
    title = name.split('/').pop()
    for c in list( title ):
      if c_f.get(c) is None:
        c_f[c] = 0
      c_f[c] += 1
  for e, (c, f) in enumerate(sorted( c_f.items(), key=lambda x:x[1]*-1)):
    print( e, c, f )
if __name__ == '__main__':
  if '--step1' in sys.argv:
    step1()
  if '--step2' in sys.argv:
    step2()
