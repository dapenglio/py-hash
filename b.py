import sys
import mmh3
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

def hashIntoBuckets(bucketNo, maxId):
  startTime = time.time()
  valueShift = 1 << 31 # added to hash value to bring it positive
  buckets = [[] for x in range(bucketNo)]
  for i in range(maxId):
    hash = mmh3.hash(str(i)) + valueShift
    bucket = hash % bucketNo
    #print('{}:{}:bucket:{}'.format(i, hash, bucket))
    buckets[bucket].append(i)
    #print('{}:{}:bucket:{}, afterwards length {}'.format(i, hash, bucket, len(buckets[bucket])))
  #for bucketIndex in range(bucketNo):
  #  print('bucket {} has {} elements'.format(bucketIndex, len(buckets[bucketIndex])))
  counts = [len(x) for x in buckets]
  minV = min(counts)
  maxV = max(counts)
  median = np.median(counts)
  mean = np.mean(counts)
  diffPercent = (maxV-minV)*100/minV
  print('bucketNo:{} maxId:{} [min, median, mean, max] = [{}, {}, {}, {}]; difference: {}%'
    .format(bucketNo, maxId, minV, median, mean, maxV, round(diffPercent, 2)))
  endTime = time.time()
  print('whole calculation finished in {} seconds'.format(endTime - startTime))
  #drawCounts(counts)
  return minV, maxV, median, mean, diffPercent

def drawCounts(counts):
  plt.plot(counts)
  plt.title('bucket sizes (total: {})'.format(maxId))
  plt.xlabel('buckets')
  plt.ylabel('counts')
  plt.ylim(ymin=0)
  plt.show()

if __name__ == '__main__':
  #df = pd.DataFrame(index=['10M', '50M', '100M'])
  df = pd.DataFrame(index=['1K', '5K', '10K', '50K', '100K', '500K', '1M'])
  bucketSizes = [2, 4, 8, 16]
  for bucketNo in bucketSizes:
    diffPercents = []
    #for maxId in [10*1000*1000, 50*1000*1000, 100*1000*1000]:
    for maxId in [1*1000, 5*1000, 10*1000, 50*1000, 100*1000, 500*1000, 1000*1000]:
      v = hashIntoBuckets(bucketNo, maxId)
      diffPercents.append(v[4])
      print('bucket:{} maxId:{}, results:{}'.format(bucketNo, maxId, v))
    df[str(bucketNo) + ' buckets'] = diffPercents
  df.plot()
  plt.xlabel('number of ids')
  plt.ylabel('diff percents')
  plt.ylim(ymin=0)
  #plt.show()
  start = df.index[0]
  end = df.index[len(df.index) - 1]
  maxBucketSize = bucketSizes[len(bucketSizes) - 1]
  plt.savefig(start + '-' + end + '-' + str(maxBucketSize) + '.png')

def unused():
  fig, ax = plt.subplots()
  plt.plot(diffPercents)
  #plt.xticks(['1K', '10K', '100K', '1M'])
  #ax.set_xticklabels(['1K', '10K', '100K', '1M', '10M'])
  ax.set_xticklabels(['1K', '10K', '100K', '1M'])
  plt.ylim(ymin=0)
  plt.show()
