
import os
import sys

for i in range(3000):
  os.system("CUDA_VISIBLE_DEVICES=1 python3 c2d.py --train --resume")
