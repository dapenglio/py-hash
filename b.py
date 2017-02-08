import sys
import murmurhash as mmh
import mmh3
import matplotlib.pyplot as plt


if __name__ == '__main__':
  valueShift = 1 << 31 # added to hash value to bring it positive
  bucketNo = 128 # better to be power of 2
  maxId = 10 * 1000 * 1000
  if len(sys.argv) > 1: bucketNo = int(sys.argv[1])
  if len(sys.argv) > 2: maxId = int(sys.argv[2])
  buckets = [[] for x in range(bucketNo)]
  for i in range(maxId):
    hash = mmh3.hash(str(i)) + valueShift
    bucket = hash % bucketNo
    #print('{}:{}:bucket:{}'.format(i, hash, bucket))
    buckets[bucket].append(i)
    #print('{}:{}:bucket:{}, afterwards length {}'.format(i, hash, bucket, len(buckets[bucket])))
  for bucketIndex in range(bucketNo):
    print('bucket {} has {} elements'.format(bucketIndex, len(buckets[bucketIndex])))
  counts = [len(x) for x in buckets]
  plt.plot(counts)
  plt.title('bucket sizes')
  plt.xlabel('buckets')
  plt.ylabel('counts')
  plt.show()
