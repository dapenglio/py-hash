import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t=time.time()
#df=pd.read_csv('mors.csv') # per Member Orders, Returns, and Sold (net_value)
#df.index = df.member_id
#del df['member_id']
#df = df [(df.member_status == 'active') & (df.returnmoneyratio <= 1) & (df.returnmoneyratio >= 0)]
print('loaded data in ', time.time()-t);

def sum_per_netValue():
 #index = list( range(10, 110, 10));
 index = list( range(7, 101, 2));
 dfr = pd.DataFrame(index=index);
 sum = []
 pb = -1
 for i in index:
  b = np.percentile(df.net_value, i)
  df_ = df[(df.net_value <= b) & (df.net_value > pb)]
  print('{} percentile, bar {}, pb {}, member count: {}'.format(i, b, pb, len(df_)))
  pb = b;
  sum.append(np.sum(df_.net_value))
 dfr['sum of segment'] = sum
 print('loaded data in ', time.time()-t);
 dfr.plot()
 plt.xlabel('net value percentile')
 plt.ylabel('total market value of each segment')
 plt.show()


def returnMoneyRatio_per_netValue():
 #index = list( range(10, 110, 10));
 index = list( range(11, 101, 2));
 dfr = pd.DataFrame(index=index);
 #minrr = []
 meanrr = []
 rr = []
 #maxrr = []
 p10=[]
 p20=[]
 p30=[]
 p40=[]
 p50=[]
 p60=[]
 p70=[]
 p80=[]
 p90=[]
 pb = -1
 for i in index:
  b = np.percentile(df.net_value, i)
  df_ = df[(df.net_value <= b) & (df.net_value > pb)]
  print('{} percentile, bar {}, pb {}, member count: {}'.format(i, b, pb, len(df_)))
  pb = b;
  #(min_, mean_, max_) = (np.min(df_.returnmoneyratio), np.mean(df_.returnmoneyratio), np.max(df_.returnmoneyratio))
  #(min_, mean_, max_) = (np.percentile(df_.returnmoneyratio, 25), np.mean(df_.returnmoneyratio), np.percentile(df_.returnmoneyratio, 75))
  #print('{} [{}, {}, {}]'.format(i, min_, mean_, max_))
  #minrr.append(min_)
  #meanrr.append(mean_)
  #maxrr.append(max_)
  meanrr.append(np.mean(df_.returnmoneyratio))
  rr.append(sum(df_.returnsubtotal) / sum(df_.ordersubtotal))
  p10.append(np.percentile(df_.returnmoneyratio, 10))
  p20.append(np.percentile(df_.returnmoneyratio, 20))
  p30.append(np.percentile(df_.returnmoneyratio, 30))
  p40.append(np.percentile(df_.returnmoneyratio, 40))
  p50.append(np.percentile(df_.returnmoneyratio, 50))
  p60.append(np.percentile(df_.returnmoneyratio, 60))
  p70.append(np.percentile(df_.returnmoneyratio, 70))
  p80.append(np.percentile(df_.returnmoneyratio, 80))
  p90.append(np.percentile(df_.returnmoneyratio, 90))
 #dfr['25%'] = minrr
 dfr['mean'] = meanrr
 dfr['totalrr'] = rr 
 #dfr['75%'] = maxrr
 dfr['90%'] = p90
 dfr['80%'] = p80
 dfr['70%'] = p70
 dfr['60%'] = p60
 dfr['50%'] = p50
 dfr['40%'] = p40
 dfr['30%'] = p30
 dfr['20%'] = p20
 dfr['10%'] = p10
 print('loaded data in ', time.time()-t);
 plt.cla()
 dfr.plot()
 plt.xlabel('net value percentile')
 #plt.ylabel('[25%, mean, 75%] of returned money ratio')
 plt.ylabel('[10%..90%] and mean of returned money ratio')
 plt.show()


def createBigImages():
 print('{} members contributed 0.'.format(len(df[df.net_value == 0]))) #`292387
 
 plt.cla()
 plt.hist(df[(df.net_value>0) & (df.net_value <= 1000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.1-1K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>1000) & (df.net_value <= 2000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.1K-2K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>2000) & (df.net_value <= 5000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.2K-5K.png')
 
 
 plt.cla()
 plt.hist(df[(df.net_value>5000) & (df.net_value <= 10000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.5K-10K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>2000) & (df.net_value <= 10000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.2K-10K.png')
 
 
 plt.cla()
 plt.hist(df[(df.net_value>10000) & (df.net_value <= 25000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.10K-25K.png')
 
 
 plt.cla()
 plt.hist(df[(df.net_value>25000) & (df.net_value <= 50000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.25K-50K.png')
 
 
 plt.cla()
 plt.hist(df[(df.net_value>10000) & (df.net_value <= 50000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.10K-50K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>50000) & (df.net_value <= 100000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.50K-100K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>100000) & (df.net_value <= 300000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.100K-300K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>300000) & (df.net_value <= 600000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.300K-600K.png')
 
 plt.cla()
 plt.hist(df[(df.net_value>600000) & (df.net_value <= 10000000)].net_value, bins=100);
 plt.title('Histogram of net_value')
 plt.xlabel('$')
 plt.savefig('mors.600K-10M.png')

def kmean_on_return():
 rs=df[['returnsubtotal']].copy()
 for k in range(3,7):
  kmeans=KMeans(n_clusters=k)
  p=kmeans.fit_predict(rs.values)
  column = 't' + str(k)
  rs[column]=p
  ts=[]
  for g in range(0, k):
   mi = min( rs[ rs[column]==g].returnsubtotal)
   ma = max( rs[ rs[column]==g].returnsubtotal)
   ts.append(mi)
   ts.append(ma)
  ts.sort()
  print(column)
  print(ts)

def kmean_on_nv():
 nv = 'net_value'
 rs=df[[nv]].copy()
 for k in range(3,7):
  kmeans=KMeans(n_clusters=k)
  p=kmeans.fit_predict(rs.values)
  column = 't' + str(k)
  rs[column]=p
  ts=[]
  for g in range(0, k):
   mi = min( rs[ rs[column]==g][nv])
   ma = max( rs[ rs[column]==g][nv])
   ts.append(mi)
   ts.append(ma)
  #ts.sort()
  print(column)
  print(ts)

def segments():
 totalmember=float(len(df))
 totalvalue=sum(df.net_value)
 #bars=list( range(100, 2100, 100))
 bars=list( range(2000, 21000, 1000))
 countratios=[]
 valueratios=[]
 for b in bars:
  t=df[ df.net_value>=b]
  countratios.append( len(t) / totalmember)
  valueratios.append( sum(t.net_value) / totalvalue)
 sdf=pd.DataFrame()
 sdf['#member']=countratios
 sdf['net_value']=valueratios
 sdf.index=bars
 sdf.to_csv('member_segment.csv')
 sdf.plot()
 plt.xlabel('$spend')
 plt.ylabel('%')
 plt.show()


def nv_rmr_mesh():
 '''
 x=[]
 y=[]
 z=[]
 for i in range(0, 10):
  for j in range(0, 10):
   x.append(i)
   y.append(j)
   z.append(10)
 fig = plt.figure()
 ax=fig.gca(projection='3d')
 #ax.plot_wireframe(x, y, z)
 ax.scatter(x, y, z)
 plt.show()
 return

 for net_value vs. returnmoneyratio, show the member# mesh
 '''
 nvs = []
 rmrs = []
 counts = []
 nvr = list( range(000, 20001, 100))
 rmrr = list( range(0, 11, 1))
 for nvi, nv in enumerate(nvr):
  if nvi == 0: continue
  nvs.append([nv] * (len(rmrr) -1))
  rmrrow=[]
  countrow=[]
  for rmri, rmr in enumerate(rmrr):
   if rmri < 1: continue
   rmrrow.append(rmr)
   pnv = nvr[nvi - 1]
   prmr = rmrr[rmri - 1]
   df_ = df[ (df.net_value >= pnv) & (df.net_value < nv) & (df.returnmoneyratio >= prmr/10.0) & (df.returnmoneyratio < rmr/10.0)]
   count = len(df_)
   #count = sum(df_.net_value)
   #counts.append(count)
   countrow.append(count)
   print("[%d, %d), [%d, %d) -- %d" % (pnv, nv, prmr, rmr, count));
  rmrs.append(rmrrow)
  counts.append(countrow)
 fig = plt.figure()
 ax=fig.gca(projection='3d')
 ax.plot_wireframe(nvs, rmrs, counts)
 #ax.plot_surface(nvs, rmrs, counts)
 #ax.scatter(nvs, rmrs, counts)
 plt.xlabel('net_value')
 plt.ylabel('/10 returnmoneyratio')
 #plt.title('total net value')
 plt.title('member count')
 plt.show()
#plt.savefig('diff-' + str(startInd) + '.png')

def per_orderno():
 ons = list(range(0, 101, 10))
 index=[]
 count=[]
 totalnv=[]
 for i, o in enumerate(ons):
  if i==0: continue
  d_=df[(df.orderno>ons[i-1]) & (df.orderno<=o)]
  index.append(i)
  count.append(len(d_))
  totalnv.append(np.sum(d_.net_value))
 d_=df[df.orderno>100]
 index.append(111)
 count.append(len(d_))
 totalnv.append(np.sum(d_.net_value))
 dfpo = pd.DataFrame(index=index)
 dfpo['count']=count
 dfpo['totalnv']=totalnv
 dfpo[['totalnv']].plot()
 #dfpo[['count']].plot()
 plt.xlabel('order no')
 plt.show()


### main ###
t=time.time()
#createBigImages()

#sum_per_netValue()
#returnMoneyRatio_per_netValue();
#nv_rmr_mesh();
per_orderno();

print('finished in ', (time.time()-t));

#rs.to_csv('rs_labeled.csv')
#print('Totall used time: ', time.time()-t)


'''
import datetime
import mysql.connector

cnx = mysql.connector.connect(user='scott', database='employees')
cursor = cnx.cursor()

query = ("SELECT first_name, last_name, hire_date FROM employees "
         "WHERE hire_date BETWEEN %s AND %s")

	 hire_start = datetime.date(1999, 1, 1)
	 hire_end = datetime.date(1999, 12, 31)

	 cursor.execute(query, (hire_start, hire_end))

	 for (first_name, last_name, hire_date) in cursor:
	   print("{}, {} was hired on {:%d %b %Y}".format(
	       last_name, first_name, hire_date))

	       cursor.close()
	       cnx.close()
'''
