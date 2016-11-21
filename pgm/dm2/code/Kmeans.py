
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt 


# In[3]:

data = np.loadtxt('../classification_data_HWK2/EMGaussian.data')


# In[37]:

def kmeans(data,n_clusters,n_iterations):
    means = np.random.rand(n_clusters,data.shape[1])
    
    for N in range(0,n_iterations):
        
        cluster_assignement = []
        for k,x in enumerate(data):
            cluster_assignement.append(np.argmin([np.linalg.norm(x-m) for m in means]))
            
        for k,mu in enumerate(means):
            x_idx = [i for i,x in enumerate(cluster_assignement) if k == x]
            means[k] = 1./(len(x_idx))*sum([data[i] for i in x_idx])
                           
    return cluster_assignement, means
                        


# In[47]:

n_cluster = 5
assignements ,means  = kmeans(data,n_cluster,100)


# In[48]:

LABEL_COLOR_MAP = {0 : 'r',1 : 'b', 2: 'g',3:'c',4:'m',5:'y',6:'k'}
colors = [LABEL_COLOR_MAP[k] for k in assignements]

plt.figure()
plt.scatter(data[:,0],data[:,1],color = colors, marker='+')
plt.scatter(means[:,0],means[:,1], marker=">")


# In[ ]:



