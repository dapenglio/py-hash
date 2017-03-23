import pandas as pd
import time
import mmh3
from dateutil import parser

#exec( open('a.py').read())

t=time.time()
#df = pd.read_csv('partm.csv', delimiter=';')
df = pd.read_csv('members.csv', delimiter=';')
print('loaded {} members'.format(len(df)))
print('used {} seconds'.format(time.time()-t))

t=time.time()
modified_month = [0] * 12
weekday = [0] * 7
bucket = [0] * 16
ms = []
ds = []
hs = []
bs = []
valueShift = 1 << 31
for index, row in df.iterrows():
  dt = parser.parse(row['modified_date'])
  m = dt.month
  modified_month[m-1] = modified_month[m-1] + 1
  ms.append(m)
  d = dt.weekday()
  weekday[d] = weekday[d] + 1
  ds.append(d)
  h = mmh3.hash(str(row['member_id'])) + valueShift
  hs.append(h)
  b = h % 16
  bucket[b-1] = bucket[b-1] + 1
  bs.append(b)
df['month'] = ms
df['weekday'] = ds
df['hash'] = hs
df['bucket'] = bs
print('used {} seconds'.format(time.time()-t))
print('modified_month:{}'.format(modified_month))
print('weekday:{}'.format(weekday))
print('bucket:{}'.format(bucket))

t=time.time()
print('used {} seconds'.format(time.time()-t))

t=time.time()
print('used {} seconds'.format(time.time()-t))
