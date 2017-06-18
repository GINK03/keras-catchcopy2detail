import json
DEF = 0.001
WID = 2000

RAT = DEF/WID

epoch_rate = {}
for i in range(WID):
  epoch_rate[i] = DEF - RAT*i
  print(i , DEF -  RAT*i )
open("epoch_rate.json", "w").write( json.dumps(epoch_rate) )
