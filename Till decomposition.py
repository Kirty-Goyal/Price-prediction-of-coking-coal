
# coding: utf-8

# # filling the missing data 

# In[2]:


#missing data filled
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

df = pd.read_csv('desu.csv')
df_null = df['DEXUSUK'].isnull()
df_null[df_null==True].count() #'0' means No Null values
df[df.DEXUSUK=='.']
df[df.DEXUSUK=='.'].count()
df_missing = df.DEXUSUK[df.DEXUSUK=='.']
missing_indexes = np.array(df_missing.index).astype(int)

for i in missing_indexes:
    if df.DEXUSUK[i+5]=='.':
        df.DEXUSUK[i] = (float(df.DEXUSUK[i-1]) + float(df.DEXUSUK[i+6]))/2
    elif df.DEXUSUK[i+4]=='.':
        df.DEXUSUK[i] = (float(df.DEXUSUK[i-1]) + float(df.DEXUSUK[i+5]))/2
    elif df.DEXUSUK[i+3]=='.':
        df.DEXUSUK[i] = (float(df.DEXUSUK[i-1]) + float(df.DEXUSUK[i+4]))/2
    elif df.DEXUSUK[i+2]=='.':
        df.DEXUSUK[i] = (float(df.DEXUSUK[i-1]) + float(df.DEXUSUK[i+3]))/2
    elif df.DEXUSUK[i+1]=='.':
        df.DEXUSUK[i] = (float(df.DEXUSUK[i-1]) + float(df.DEXUSUK[i+2]))/2
    else :
        df.DEXUSUK[i] = (float(df.DEXUSUK[i-1]) + float(df.DEXUSUK[i+1]))/2
# check for missing values

df.DATE = pd.to_datetime(df.DATE)
df.DEXUSUK = pd.to_numeric(df.DEXUSUK)
df.set_index('DATE',inplace=True)

df.plot(y='DEXUSUK',figsize=(15,5))


# In[3]:


df.index = pd.to_datetime(df.index)


# In[4]:


df.index


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


#missing dates filled
idx = pd.date_range('1971-01-01', '2018-05-10')
ts=df['DEXUSUK']
s = pd.Series(ts)
s.index = pd.DatetimeIndex(s.index)

s = s.reindex(idx, fill_value="NaN")
s = s.astype(float)

s = s.interpolate(method='linear', axis=0).ffill().bfill()
print(s)


# In[8]:


s.describe()


# In[9]:


df1=s['1971-01']


# In[10]:


df1.describe()


# In[11]:


plt.figure(figsize=(20,10))
plt.plot(s)
plt.tick_params(axis='both', which='major', labelsize=30)


# # Converting into month-wise data

# In[12]:


mnth=s.groupby( pd.Grouper( freq='M')).mean()
plt.figure(figsize=(50,20))
plt.xlabel('date',size=20)
plt.ylabel('price',size=20)
plt.plot(mnth)



# In[13]:


monthn=mnth


# In[14]:


#PIVOT TABLE
plt.figure(figsize=(30,10))
plt.xlabel('date',size=20)
plt.ylabel('price',size=20)
mnth=mnth.to_frame()

mnth.name = 'DEXUSUK'
mnth['month'] = mnth.index.month
mnth['Year'] = mnth.index.year
piv = pd.pivot_table(mnth, index=['month'],columns=['Year'], values=['DEXUSUK'])

piv.plot()
plt.gca().legend_.remove()


piv


# In[15]:


monthn=mnth


# In[16]:


plt.figure(figsize=(100,10))
plt.xlabel('date',size=20)
plt.ylabel('price',size=20)

mnth.name = 'DEXUSUK'
mnth['month'] = mnth.index.month
mnth['Year'] = mnth.index.year
piv1 = pd.pivot_table(mnth, index=['Year'],columns=['month'], values=['DEXUSUK'])

piv1.plot()


piv1


# In[17]:


corr1 = piv1.corr(method='pearson', min_periods=1)
print(corr1)


# In[18]:


#heatmap for correlation between different months
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# plot the heatmap
sns.heatmap(corr1, 
        xticklabels=corr1.columns,
        yticklabels=corr1.columns)
plt.xlabel('Months',size=20)
plt.ylabel('Months',size=20)


# In[42]:


mnth1=s.groupby( pd.Grouper( freq='M')).mean()
plt.figure(figsize=(30,10))
plt.xlabel('Date',size=30)
plt.ylabel('Price',size=30)
mnth1.plot()
plt.tick_params(axis='x', which='major', labelsize=3)
plt.tick_params(axis='y', which='major', labelsize=30)


# In[20]:


x=mnth1.values


# ### Mann-Kendall Test

# In[21]:


from scipy.stats import norm

n = len(x)
s2 = 0
for k in range(n-1):
   for j in range(k+1, n):
       s2 += np.sign(x[j] - x[k])

   # calculate the unique data
unique_x = np.unique(x)
g = len(unique_x)

   # calculate the var(s)
if n == g:  # there is no tie
   var_s = (n*(n-1)*(2*n+5))/18
else:  # there are some ties in data
   tp = np.zeros(unique_x.shape)
   for i in range(len(unique_x)):
       tp[i]= sum(x== unique_x[i])
   var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

if s2 > 0:
   z = (s2 - 1)/np.sqrt(var_s)
elif s2 < 0:
   z = (s2 + 1)/np.sqrt(var_s)
else: # s == 0:
   z = 0

   # calculate the p_value
p = 2*(1-norm.cdf(abs(z)))  # two tail test
h = abs(z) > norm.ppf(1-0.05/2)

if (z < 0) and h:
   trend = 'decreasing'
elif (z > 0) and h:
    trend = 'increasing'
else:
   trend = 'no trend'

print(trend)
print(h)
print(p)
print(z)



#       Seasonal Decomposition

# In[22]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(mnth1)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(26,10))
plt.subplot(411)
plt.plot(mnth1, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')


# In[23]:


plt.figure(figsize=(30,6))
plt.plot(residual,label='Residual',color='r')
plt.legend(loc='best', prop={'size':20})
plt.tick_params(axis='both', which='major', labelsize=30)

