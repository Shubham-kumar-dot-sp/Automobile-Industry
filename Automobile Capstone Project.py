#!/usr/bin/env python
# coding: utf-8

# # Capstone Project-Automobile Industry

# About: There is an automobile company fom India which wants to enter the 
#  Indian used-car market by setting up their company.

# Data Preparation and Data Cleaning

# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"E:\SNU B3 - CapstoneProject_UsedCarsData\CapstoneProject_UsedCarsData\UserCarData.csv")

dataset.head() 


# In[95]:


dataset.info() 


# In[96]:


dataset.shape


# In[97]:


dataset.head()


# In[159]:


dataset[['selling_price','km_driven','mileage','engine','max_power'].describe()


# Inspection based on value counts, number of occurencea and visualistaion:
# 

# In[99]:


dataset['name'].value_counts().to_frame()


# In[100]:


var = dataset.groupby('name').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('name')
ax1.set_ylabel('selling_price')
ax1.set_title("name Vs selling_price")
var.plot(kind='bar')


# In[101]:


var = dataset.groupby('name').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('name')
ax1.set_ylabel('selling_price')
ax1.set_title("name Vs selling_price")
var.plot(kind='bar')


# In[102]:


dataset['year'].value_counts()


# In[103]:


var = dataset.groupby('year').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('year')
ax1.set_ylabel('selling_price')
ax1.set_title("year Vs selling_price")
var.plot(kind='bar')


# In[104]:


dataset['Region'].value_counts()


# In[105]:


var = dataset.groupby('Region').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Region')
ax1.set_ylabel('selling_price')
ax1.set_title("Region Vs selling_price")
var.plot(kind='bar')


# In[106]:


dataset['State or Province'].value_counts()


# In[107]:


var = dataset.groupby('State or Province').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('State or Province')
ax1.set_ylabel('selling_price')
ax1.set_title("State or Province Vs selling_price")
var.plot(kind='bar')


# In[108]:


dataset['City'].value_counts()


# In[109]:


dataset['fuel'].value_counts()


# In[110]:


var = dataset.groupby('fuel').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('fuel')
ax1.set_ylabel('selling_price')
ax1.set_title("Fuel Vs selling_price")
var.plot(kind='bar')


# In[112]:


dataset['seller_type'].value_counts()


# In[113]:


var = dataset.groupby('seller_type').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('seller_type')
ax1.set_ylabel('selling_price')
ax1.set_title("seller_type Vs selling_price")
var.plot(kind='bar')


# In[114]:


dataset['transmission'].value_counts()


# In[115]:


var = dataset.groupby('transmission').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('transmission')
ax1.set_ylabel('selling_price')
ax1.set_title("transmission Vs selling_price")
var.plot(kind='bar')


# In[116]:


dataset['owner'].value_counts()


# In[117]:


var = dataset.groupby('owner').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('owner')
ax1.set_ylabel('selling_price')
ax1.set_title("owner Vs selling_price")
var.plot(kind='bar')


# In[118]:


dataset['seats'].value_counts()


# In[119]:


var = dataset.groupby('seats').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('seats')
ax1.set_ylabel('selling_price')
ax1.set_title("seats Vs selling_price")
var.plot(kind='bar')


# In[120]:


dataset['sold'].value_counts()


# In[121]:


var = dataset.groupby('sold').selling_price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('sold')
ax1.set_ylabel('selling_price')
ax1.set_title("sold Vs selling_price")
var.plot(kind='bar')


# In[122]:



dataset.plot(x='km_driven',y='selling_price',kind='scatter')


# In[123]:


dataset.plot(x='mileage',y='selling_price',kind='scatter')


# In[124]:


dataset.corr()
sns.heatmap(dataset.corr())


# In[125]:



dataset.corr() 
pd.set_option('display.max_columns',50)
dataset.corr()


# In[126]:


dataset=pd.get_dummies(dataset, columns=['Region'])


# In[130]:



dataset.groupby('Region')['sold_Y'].value_counts().to_frame()


# In[127]:


dataset=pd.get_dummies(dataset, columns=['fuel'])


# In[128]:


dataset=pd.get_dummies(dataset, columns=['sold'])


# In[129]:



dataset.groupby('name')['sold_Y'].value_counts().to_frame()


# In[132]:


dataset=pd.get_dummies(dataset, columns=['seller_type'])


# In[133]:


dataset=pd.get_dummies(dataset, columns=['transmission'])


# In[135]:


dataset=pd.get_dummies(dataset, columns=['owner'])


# In[136]:


dataset=pd.get_dummies(dataset, columns=['seats'])


# In[137]:


dataset=pd.get_dummies(dataset, columns=['name'])


# In[138]:


dataset=pd.get_dummies(dataset, columns=['year'])


# In[139]:


dataset.head()


# In[140]:


dataset.columns


# In[141]:


dataset.describe()


# In[142]:


dataset['region_sum']=dataset['Region_Central']+dataset['Region_East']+dataset['Region_South']+dataset['Region_West']
dataset['fuel_sum']=dataset['fuel_Diesel']+dataset['fuel_LPG']+ dataset['fuel_Petrol']+dataset['fuel_CNG']
dataset['seller_type_sum']=dataset['seller_type_Dealer']+dataset['seller_type_Individual']+dataset['seller_type_Trustmark_Dealer']
dataset['transmission_sum']=dataset['transmission_Automatic']+dataset['transmission_Manual']
dataset['owner_sum']=dataset['owner_First_Owner']+dataset['owner_Fourth_Above_Owner']+dataset['owner_Second_Owner']+dataset['owner_Test_Drive_Car']+dataset['owner_Third_Owner']
dataset['sold_sum']=dataset['sold_N']+dataset['sold_Y']
dataset['seats_sum']=dataset['seats_2']+dataset['seats_4']+dataset['seats_5']+dataset['seats_6']+dataset['seats_7']+dataset['seats_8']+dataset['seats_9']+dataset['seats_10']+dataset['seats_14']

dataset['brand_sum']=dataset['name_Audi']+dataset['name_BMW']
+dataset['name_Ambassador']+dataset['name_Ashok']+dataset['name_Fiat']
+dataset['name_Force']+dataset['name_Ford']+dataset['name_Honda']
+dataset['name_Hyundai']+dataset['name_Isuzu']+dataset['name_Jaguar']
+dataset['name_Jeep']+dataset['name_Kia']+dataset['name_Land']
+dataset['name_Lexus']+dataset['name_MG']+dataset['name_Mahindra']
+dataset['name_Maruti']+dataset['name_Mercedes']+dataset['name_Mitsubishi']
+dataset['name_Nissan']+dataset['name_Opel']+dataset['name_Renault']
+dataset['name_Skoda']+dataset['name_Tata']+dataset['name_Toyota']
+dataset['name_Volkswagen']+dataset['name_Volvo']

dataset['year_sum']=dataset['year_1994']+dataset['year_1995']
+dataset['year_1996']+dataset['year_1997']+dataset['year_1998']
+dataset['year_1999']+dataset['year_2000']+dataset['year_2001']
+dataset['year_2002']+dataset['year_2003']+dataset['year_2004']
+dataset['year_2005']+dataset['year_2006']+dataset['year_2007']
+dataset['year_2008']+dataset['year_2009']+dataset['year_2010']
+dataset['year_2011']+dataset['year_2012']+dataset['year_2013']
+dataset['year_2014']+dataset['year_2015']+dataset['year_2016']
+dataset['year_2017']+dataset['year_2018']+dataset['year_2019']
+dataset['year_2020']



# In[143]:


dataset.columns


# In[144]:


dataset


# In[145]:


dataset.corr()
sns.heatmap(dataset.corr())


# In[146]:



print("correlation Matrix: ")
print(dataset.corr())


# REGRESSION:

# In[147]:



import statsmodels.formula.api as smf

reg=smf.ols("selling_price~km_driven+year_sum+fuel_sum+region_sum+sold_sum+brand_sum+seats_sum+owner_sum+transmission_sum+seller_type_sum+mileage+engine+max_power",data=dataset)
results_summation=reg.fit()
results_summation.summary
print(results_summation.summary())


# running regression after removing insignificant variables

# In[148]:



import statsmodels.formula.api as smf

reg=smf.ols("selling_price~km_driven+fuel_sum+region_sum+sold_sum+brand_sum+seats_sum+owner_sum+transmission_sum+seller_type_sum+mileage+engine+max_power",data=dataset)
results_summation_sig=reg.fit()
results_summation_sig.summary
print(results_summation_sig.summary())


#  running regression including each dummy variables

# In[149]:



import statsmodels.formula.api as smf

reg=smf.ols("selling_price~name_Audi+name_BMW+name_Ambassador+name_Ashok+name_Fiat+name_Force+name_Ford+name_Honda+name_Hyundai+name_Isuzu+name_Jaguar+name_Jeep+name_Kia+name_Land+name_Lexus+name_MG+name_Mahindra+name_Maruti+name_Mercedes+name_Mitsubishi+name_Nissan+name_Opel+name_Renault+name_Skoda+name_Tata+name_Toyota+name_Volkswagen+name_Volvo+year_1994+year_1995+year_1996+year_1997+year_1998+year_1999+year_2000+year_2001+year_2002+year_2003+year_2004+year_2005+year_2006+year_2007+year_2008+year_2009+year_2010+year_2011+year_2012+year_2013+year_2014+year_2015+year_2016+year_2017+year_2018+year_2019+year_2020+seats_2+seats_4+seats_5+seats_6+seats_7+seats_14+seats_8+seats_9+seats_10+owner_First_Owner+owner_Fourth_Above_Owner+owner_Second_Owner+owner_Test_Drive_Car+owner_Third_Owner+transmission_Automatic+transmission_Manual+seller_type_Dealer+seller_type_Individual+seller_type_Trustmark_Dealer+fuel_Diesel+fuel_LPG+fuel_Petrol+fuel_CNG+Region_Central+Region_East+Region_South+Region_West+km_driven+mileage+engine+max_power+sold_N+sold_Y",data=dataset)
results_dummy=reg.fit()
results_dummy.summary
print(results_dummy.summary())


# validating results, see how good the model is pridected value vs actual value

# In[150]:



predictions=results_dummy.predict(dataset)
actuals=dataset['selling_price']
plt.plot(actuals,"b")
plt.plot(predictions,"r")


#  running regression excluding each insignificant dummy variables

# In[151]:



import statsmodels.formula.api as smf

reg=smf.ols("selling_price~name_Audi+name_BMW+name_Hyundai+name_Isuzu+name_Jaguar+name_Jeep+name_Kia+name_Land+name_Lexus+name_MG+name_Maruti+name_Mercedes+name_Tata+name_Toyota+name_Volvo+year_2003+year_2004+year_2005+year_2006+year_2007+year_2008+year_2009+year_2010+year_2011+year_2014+year_2015+year_2016+year_2017+year_2018+year_2019+year_2020+seats_4+seats_6+owner_First_Owner+owner_Fourth_Above_Owner+owner_Second_Owner+owner_Test_Drive_Car+owner_Third_Owner+transmission_Automatic+seller_type_Dealer+fuel_Diesel+fuel_LPG+fuel_Petrol+fuel_CNG+km_driven+engine+max_power+sold_Y",data=dataset)
results_dummy=reg.fit()
results_dummy.summary
print(results_dummy.summary())


# separating features from the target variable

# In[152]:



x_data=dataset[['name_Audi','name_BMW','name_Hyundai','name_Isuzu','name_Jaguar','name_Jeep','name_Kia','name_Land','name_Lexus','name_MG','name_Maruti','name_Mercedes','name_Tata','name_Toyota','name_Volvo','year_2003','year_2004','year_2005','year_2006','year_2007','year_2008','year_2009','year_2010','year_2011','year_2014','year_2015','year_2016','year_2017','year_2018','year_2019','year_2020','seats_4','seats_6','owner_First_Owner','owner_Fourth_Above_Owner','owner_Second_Owner','owner_Test_Drive_Car','owner_Third_Owner','transmission_Automatic','seller_type_Dealer','fuel_Diesel','fuel_LPG','fuel_Petrol','fuel_CNG','km_driven','engine','max_power','sold_Y']]
y_data=dataset['selling_price']


# In[53]:


x_data


# In[153]:


y_data


# In[154]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x_data,y_data,test_size=0.2,shuffle=False)


# In[155]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[156]:


log_clf=LogisticRegression(solver="lbfgs",random_state=42)
rnd_clf=RandomForestClassifier(n_estimators=100,random_state=42)
svm_clf=SVC(gamma='scale',random_state=42)
voting_clf=VotingClassifier(
      estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
      voting='hard'
)


# In[157]:


voting_clf.fit(x_train,y_train)


# In[59]:


from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))

