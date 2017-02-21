import pandas as pd
import time
from dateutil import parser
import mmh3

#bucket_sizes = [5, 10, 15, 20, 25, 30]
bucket_sizes = [25, 50, 75, 100]
durations = list( range(1, 15))
#bucket_sizes = [5, 10]
#durations = list( range(1, 3))

def hash(i):
  valueShift = 1 << 31
  #return mmh3.hash(str(row['member_id'])) + valueShift
  return mmh3.hash(str(i)) + valueShift


t = time.time()
def stats(df, bucket_size):
  bs=[0] * bucket_size
  sums=[0.0] * bucket_size
  for index, row in df.iterrows():
    member_id = row['member_id']
    h = hash(member_id)
    bucket = h % bucket_size
    bs[bucket] = bs[bucket] + 1
    sums[bucket] = sums[bucket] + row['subtotal']
  print(sums)
  print(bs)
  diffPercent = (max(sums) - min(sums)) / min(sums) * 100
  print('{}: {:.2f}%'.format(len(df), diffPercent))
  return diffPercent

x = []
y = []
z = []
df = pd.read_csv('orders.csv', nrows=2000000)
for b in bucket_sizes:
  zs = []
  for d in durations:
    df1 = df[750000 : 750000 + 25000 * d ] # about 25K orders a day
    r = stats(df1, b)
    zs.append(r)
  x.append( [b] * len(durations))
  y.append( durations.copy())
  z.append( zs)

print('loading used {} seconds'.format(time.time()-t))
print(x)
print(y)
print(z)

t = time.time()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(x, y, z)
plt.show()

print('used {} seconds'.format(time.time()-t))

t = time.time()
print('used {} seconds'.format(time.time()-t))

t = time.time()
print('used {} seconds'.format(time.time()-t))

t = time.time()
print('used {} seconds'.format(time.time()-t))

t = time.time()
print('used {} seconds'.format(time.time()-t))
