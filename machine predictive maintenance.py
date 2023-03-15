#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("predictive_maintenance.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.columns


# In[9]:


df.nunique()


# In[10]:


df['Type'].unique()


# In[11]:


df['Failure Type'].unique()


# In[12]:


new_df=df.drop(['UDI', 'Product ID', 'Type', 'Target'],axis=1)


# In[13]:


new_df


# In[14]:


new_df.dtypes


# In[15]:


plt.hist(new_df['Air temperature [K]'],bins=20)
plt.xlabel('temperature')
plt.ylabel('frequency')
plt.title('Distribution of Air Temperature')
plt.show()


# In[16]:


plt.hist(new_df['Rotational speed [rpm]'],bins=20)
plt.xlabel('speed')
plt.ylabel('frequency')
plt.title('Distribution of Rotational speed')
plt.show()


# In[17]:


plt.boxplot(new_df['Process temperature [K]'])
plt.title('Boxplot of Process temperature')
plt.ylabel('Temperature')
plt.show()


# In[18]:


plt.boxplot(new_df['Torque [Nm]'])
plt.title('Boxplot of Torque')
plt.ylabel('Nm')
plt.show()


# In[19]:


plt.scatter(new_df['Air temperature [K]'],new_df['Torque [Nm]'])
plt.xlabel('Air temperature')
plt.ylabel('Torque')
plt.title('Air temperature vs Torque')
plt.show()


# In[20]:


plt.scatter(new_df['Process temperature [K]'],new_df['Tool wear [min]'])
plt.xlabel('Process temperature')
plt.ylabel('Tool wear')
plt.title('Process temperature vs Tool wear')
plt.show()


# In[21]:


corr=new_df.corr()
sns.heatmap(corr,annot=True)
plt.title('Corelation Matrix')
plt.show()


# In[22]:


sns.pairplot(new_df,hue='Tool wear [min]')
plt.show()


# In[23]:


sns.pairplot(new_df,hue='Failure Type')
plt.show()


# Finally, we can group the data by the failure variable and calculate the mean of the other variables for each group:

# In[24]:


grouped=new_df.groupby('Failure Type').mean()
grouped


#  let's plot some histograms to check the distribution of the numerical features
#  This will give us an idea of the range of values in each feature, as well as any skewness or outliers

# In[25]:


new_df.hist(bins=50,figsize=(20,15))
plt.show()


# let's plot some boxplots to check for any outliers
# This will help us identify any extreme values that may need to be treated or removed from the dataset

# In[26]:


new_df.plot(kind='box', subplots=True, layout=(4,4), figsize=(15,15))
plt.show()


# Calculate rolling statistics such as the mean, standard deviation, and maximum of sensor readings over a certain window of time. This can help capture trends in the data and detect anomalies.

# In[27]:


window_size=10
new_df['temp_mean'] = new_df['Air temperature [K]'].rolling(window_size).mean()
new_df['temp_std'] = new_df['Air temperature [K]'].rolling(window_size).std()
new_df['temp_max'] = new_df['Air temperature [K]'].rolling(window_size).max()


# In[28]:


window_size=10
new_df['temp_mean'] = new_df['Process temperature [K]'].rolling(window_size).mean()
new_df['temp_std'] = new_df['Process temperature [K]'].rolling(window_size).std()
new_df['temp_max'] = new_df['Process temperature [K]'].rolling(window_size).max()


# In[29]:


window_size=10
new_df['rot_mean'] = new_df['Rotational speed [rpm]'].rolling(window_size).mean()
new_df['rot_std'] = new_df['Rotational speed [rpm]'].rolling(window_size).std()
new_df['rot_max'] = new_df['Rotational speed [rpm]'].rolling(window_size).max()


# In[30]:


window_size=10
new_df['tor_mean'] = new_df['Torque [Nm]'].rolling(window_size).mean()
new_df['tor_std'] = new_df['Torque [Nm]'].rolling(window_size).std()
new_df['tor_max'] = new_df['Torque [Nm]'].rolling(window_size).max()


# In[33]:


window_size=10
new_df['tool_mean'] = new_df['Tool wear [min]'].rolling(window_size).mean()
new_df['tool_std'] = new_df['Tool wear [min]'].rolling(window_size).std()
new_df['tool_max'] = new_df['Tool wear [min]'].rolling(window_size).max()


# Create new features by combining existing features. For example, we can multiply the temperature and torque readings to capture their interaction.

# In[34]:


new_df['temp_torque_interaction']=new_df['Air temperature [K]']*new_df['Torque [Nm]']


# In[35]:


new_df['temp_torque_interaction']


# In[36]:


new_df.dropna(inplace=True)


# In[37]:


# Split the data into features and labels
X = new_df.drop('Failure Type', axis=1)
y =new_df['Failure Type']


# we'll split our dataset into training and testing sets. We'll use 80% of the data for training and 20% for testing.

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# we'll preprocess our data by scaling the features using a standard scaler.

# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


scaler=StandardScaler()


# In[42]:


X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[44]:


model=LogisticRegression()


# In[45]:


model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
precision=precision_score(y_test, y_pred,average='macro')
print(f'Precision: {precision:.4f}')
recall=recall_score(y_test, y_pred,average='macro')
print(f'Recall: {recall:.4f}')
f1=f1_score(y_test, y_pred,average='macro')
print(f'F1: {f1:.4f}')


# we can train a machine learning model on our training data. Here, we'll use a Random Forest Classifier as an example.

# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train_scaled,y_train)


# we'll evaluate our model on the testing data by calculating its accuracy, precision, recall, and F1 score.

# In[48]:


y_pred=rf.predict(X_test_scaled)


# In[49]:


accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
precision=precision_score(y_test, y_pred,average='macro')
print(f'Precision: {precision:.4f}')
recall=recall_score(y_test, y_pred,average='macro')
print(f'Recall: {recall:.4f}')
f1=f1_score(y_test, y_pred,average='macro')
print(f'F1: {f1:.4f}')


# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


clf=DecisionTreeClassifier()


# In[52]:


clf.fit(X_train_scaled,y_train)


# In[53]:


y_pred=clf.predict(X_test_scaled)


# In[54]:


accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
precision=precision_score(y_test, y_pred,average='macro')
print(f'Precision: {precision:.4f}')
recall=recall_score(y_test, y_pred,average='macro')
print(f'Recall: {recall:.4f}')
f1=f1_score(y_test, y_pred,average='macro')
print(f'F1: {f1:.4f}')


# In[58]:


from sklearn.metrics import confusion_matrix


# In[70]:


cm=confusion_matrix(y_test,y_pred)


# In[71]:


print(cm)


# In[72]:


from sklearn.metrics import classification_report


# In[75]:


print(classification_report(y_test,y_pred))


# In[ ]:




