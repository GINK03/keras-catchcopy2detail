
import os
import sys

if '-1' in sys.argv:
  for i in range(3000):
    os.system("CUDA_VISIBLE_DEVICES=1 python3 c2d.py --train --resume")

if '-0' in sys.argv:
  for i in range(3000):
    os.system("CUDA_VISIBLE_DEVICES=0 python3 c2d.py --train --resume")
