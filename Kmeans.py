
# coding: utf-8

# In[121]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# In[122]:


X = np.random.rand(100,2)


# In[123]:


X[0]


# In[124]:


plt.scatter(X[:,0],X[:,1], s=10)


# In[125]:


from sklearn.cluster import KMeans


# In[126]:


clf=KMeans(n_clusters=3)


# In[127]:


clf.fit(X)


# In[128]:


clf.labels_


# In[129]:


plt.scatter(X[:,0], X[:,1], c=clf.labels_)


# In[130]:


print(clf.labels_)


# In[131]:


Y = np.random.rand(100,3)
for i in range(10):
    print(Y[i])
print("---")
#print(Y[:,2])


# In[132]:


from mpl_toolkits.mplot3d import Axes3D
fig= plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(Y[:,0], Y[:,1], Y[:,2])
plt.show()


# In[136]:


clf=KMeans(n_clusters=3)
clf.fit(Y)


# In[137]:


ax.scatter(Y[:,0], Y[:,1], Y[:,2])
plt.show()


# In[147]:


fig1 = plt.figure(figsize=(15,15))
ax=fig1.add_subplot(111,projection='3d')
ax.scatter(Y[:,0], Y[:,1], Y[:,2],s=50,c=clf.labels_)
plt.show()


# In[119]:




