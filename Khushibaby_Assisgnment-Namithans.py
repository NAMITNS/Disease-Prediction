#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries & loading the data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


data1=pd.read_excel("Biostatistician_Challenge_Dataset.xlsx")


# In[3]:


data1.head(10)


# # Data understanding

# In[4]:


data1.shape


# In[5]:


data1.describe()


# In[6]:


data1.info()


# In[7]:


data1.isnull().sum()


# In[8]:


duplicate = data1[data1.duplicated()]
duplicate


# In[9]:


#there are no duplicates


# # Data cleaning

# In[10]:


data2=data1.copy()


# In[11]:


data2.drop('diseases', axis=1, inplace=True)


# In[12]:


#decoding string with their values


# In[13]:


data2['primary_cooking_fuel']=data2['primary_cooking_fuel'].replace(['Kerosene oil (kerosene)'],5)
data2['primary_transport']=data2['primary_transport'].replace(['Motorcycle (Two wheeler)'],1)
data2['primary_transport']=data2['primary_transport'].replace(['Car (Four wheeler)'],2)
data2['education_status']=data2['education_status'].replace(['Primary 1-5th'],2)  
data2['education_status']=data2['education_status'].replace(['Middle 6-8th'],3)
data2['primary_house_material']=data2['primary_house_material'].replace(['Brick and concrete house (Pakka)', 'Made of clay and slurry (Kachcha-Pakka)','Brick and concrete house (Teen Shed)','Brick and concrete house (Khaprail)'],[3,2,4,5])


# In[14]:


data2.head(10)


# In[15]:


#filling null values with mode & mean for (age)


# In[16]:


data2["primary_toilet"].fillna(data2["primary_toilet"].mode()[0], inplace=True)
data2["primary_drinking_water"].fillna(data2["primary_drinking_water"].mode()[0], inplace=True)
data2["toilet_usage_status"].fillna(data2["toilet_usage_status"].mode()[0], inplace=True)
data2["primary_transport"].fillna(data2["primary_transport"].mode()[0], inplace=True)
data2["primary_electricity"].fillna(data2["primary_electricity"].mode()[0], inplace=True)
data2["primary_house_material"].fillna(data2["primary_house_material"].mode()[0], inplace=True)
data2["primary_cooking_fuel"].fillna(data2["primary_cooking_fuel"].mode()[0], inplace=True)
data2["education_status"].fillna(data2["education_status"].mode()[0], inplace=True)
data2["occupation_status"].fillna(data2["occupation_status"].mode()[0], inplace=True)
data2["sdh_occupational_risk"].fillna(data2["sdh_occupational_risk"].mode()[0], inplace=True)
data2["age"].fillna(data2["age"].mean(), inplace=True)


# In[17]:


data2.isnull().sum()


# In[18]:



#finding mode values


# In[19]:


data2.primary_transport.mode()


# In[20]:


data2.primary_house_material.mode()


# In[21]:


data2.primary_cooking_fuel.mode()


# In[22]:


data2.education_status.mode()


# In[23]:


data2.education_status.value_counts()


# In[24]:


data2.occupation_status.mode()


# In[25]:


data2.occupation_status.value_counts()


# In[26]:


data2.primary_drinking_water.mode()


# In[27]:


data2.primary_toilet.mode()


# In[28]:


data2.toilet_usage_status.mode()


# In[29]:


data2.toilet_usage_status.value_counts()


# In[30]:


data2.primary_electricity.mode()


# In[31]:


data2.religion.mode()


# In[32]:


data2.caste.mode()


# In[33]:


data2["sdh_occupational_risk"].value_counts()


# In[34]:


#here education_status,occupation_status,sdh_occupational_risk,toilet_usage_status has 0 as the mode value. Hence considering next highest mode value and replacing it with zeroes


# In[35]:


#replaceing all zeros(i.e., missing values) with mode value


# In[36]:


data2['primary_cooking_fuel'].replace([0],1,inplace=True)
data2['primary_transport'].replace(['0'],1,inplace=True)
data2['primary_house_material'].replace(['0'],1,inplace=True)
data2["sdh_occupational_risk"].replace([0],9,inplace=True)
data2['primary_drinking_water'].replace([0.0],5.0,inplace=True)
data2['primary_toilet'].replace([0],6,inplace=True)
data2['primary_electricity'].replace([0],6,inplace=True)
data2['toilet_usage_status'].replace([0.0],1.0,inplace=True)
data2['education_status'].replace([0],2,inplace=True)
data2['occupation_status'].replace([0],5,inplace=True)
data2['religion'].replace([0],1,inplace=True)
data2['caste'].replace([0],3,inplace=True)


# In[37]:


#Replacing null values of tobacco consumption column


# In[38]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(data2)


# In[39]:


imputed_data = pd.DataFrame(imputer.transform(data2))
imputed_data.columns=data2.columns


# In[40]:


imputed_data.isnull().sum()


# In[41]:


#finally there are no null values


# In[42]:


imputed_data["age"]=imputed_data["age"].astype('int')
imputed_data["religion"]=imputed_data["religion"].astype('int')
imputed_data["caste"]=imputed_data["caste"].astype('int')
imputed_data["sdh_occupational_risk"]=imputed_data["sdh_occupational_risk"].astype('float')
imputed_data["education_status"]=imputed_data["education_status"].astype('float')
imputed_data["primary_drinking_water"]=imputed_data["primary_drinking_water"].astype('float')
imputed_data["primary_toilet"]=imputed_data["primary_toilet"].astype('float')
imputed_data["toilet_usage_status"]=imputed_data["toilet_usage_status"].astype('float')
imputed_data["primary_electricity"]=imputed_data["primary_electricity"].astype('float')


# In[43]:


#Removing the rows with gender='O'
imputed_data.drop(imputed_data[imputed_data['gender']=='O'].index,inplace=True)


# In[44]:


imputed_data.describe()


# In[45]:


data3=data1[['diseases','person_id']]
imputed_data=imputed_data.merge(data3, on= 'person_id', how='left')


# In[46]:


#removing person_id as it is not required for analysis
imputed_data.drop('person_id', axis=1, inplace=True)


# In[47]:


imputed_data.head(5)


# # EDA

# In[48]:


imputed_data.diseases.value_counts()


# In[49]:


#128934 patients has no disease, 3722 patients has disease


# In[50]:


male_with_disease = len(imputed_data[(imputed_data['diseases']==1) & (imputed_data['gender']=='M')])
female_with_disease=  len(imputed_data[(imputed_data['diseases']==1) & (imputed_data['gender']=='F')])
hasdisease_tot = len(imputed_data[imputed_data['diseases']==1])
print('Probability of Male patients having the disease:',(round(male_with_disease/hasdisease_tot,3)) )
print('Probability of FeMale patients having the disease:',(round(female_with_disease/hasdisease_tot,3)) )


# In[51]:


print('Ratio of Female to male patients having the disease:', female_with_disease/male_with_disease)


# In[52]:


#Female patients having disease are 1.09 times male pateints having disease


# In[53]:


#Distribution of sdh occupation risk
a=pd.crosstab(imputed_data["sdh_occupational_risk"],imputed_data["diseases"])
a


# In[54]:


#Among the patients having the disease, most of them are females


# In[55]:


water_dist=imputed_data.groupby("primary_drinking_water").diseases.agg(Count=(lambda x: (x==1).sum()))
water_dist.sort_values(by='Count')


# In[56]:


# people drinking water from Handpump/Tube well outside house, more likely got the disease


# In[57]:


toilet_dist=imputed_data.groupby("primary_toilet").diseases.agg(Count=(lambda x: (x==1).sum()))
toilet_dist.sort_values(by='Count')


# In[58]:


#people using toilet with water(primary_toilet=1) has got disease among people using other primary_toilet 


# In[59]:


imputed_data["primary_toilet"].value_counts()


# In[60]:


#Most of the people do not use toilet


# In[61]:


alcohol_cons=imputed_data.groupby("alcohol_consumption").diseases.agg(Count=(lambda x: (x==1).sum()))
alcohol_cons.sort_values(by='Count')


# In[62]:


#Most of the patients having disease do not consume alcohol


# In[63]:


tobacco_cons=imputed_data.groupby("tobacco_consumption").diseases.agg(Count=(lambda x: (x==1).sum()))
tobacco_cons.sort_values(by='Count')


# In[64]:


#Most of the patients having disease never consumed tobacco


# In[65]:


#Correlation between primary_toilet & drinking water
imputed_data[["primary_toilet", "primary_drinking_water"]].corr()


# # Visualisations of the data

# In[66]:


imputed_data.hist(figsize=(20,20))
plt.show


# In[67]:


sns.countplot(x='diseases', data=imputed_data)


# In[68]:


#visualising the features & their relation with the target(has disease or no disease)


# In[69]:


# Distribution of age 
sns.kdeplot(data=imputed_data, x="age", hue="diseases")
plt.show()


# In[70]:


#it is seen that kde plot of age of people having disease varies in same maner as the people without disease


# In[71]:


#Relation between age & gender
sns.catplot(data=imputed_data, x='gender', y="age",hue='diseases', kind='box')


# In[72]:


#it is seen that age has many outliers
#outlier treatment
imputed_data.age.quantile([0.1, 0.23,0.5,0.75,0.9,0.95])


# In[73]:


#removing all the rows having age values above 95th percentile
imputed_data.drop(imputed_data[imputed_data['age']>66].index,inplace=True)


# In[74]:


#Distribution of gender
plt.figure(figsize=[16,8])
sns.set(style="darkgrid")


fig,(ax1,ax2)=plt.subplots(2, sharex=True,gridspec_kw={"height_ratios":(.25,.5)})

sns.histplot(imputed_data[imputed_data['diseases']==1]["gender"],ax=ax1,color='red',bins=5, alpha=1)
sns.histplot(imputed_data[imputed_data['diseases']==0]["gender"],ax=ax2,color='blue',bins=5, alpha=0.5)

ax1.set(xlabel="Gender")
ax1.set_title("Gender-wise count of disease")

plt.show()


# In[75]:


sns.countplot(x="sdh_occupational_risk", hue="diseases", data=imputed_data, palette='rocket')
plt.show()


# In[76]:


#Major risk of occupation of people having disease & not having disease is 'none'


# In[77]:


sns.countplot(x="primary_cooking_fuel", hue="diseases", data=imputed_data, palette='rocket')
plt.show()


# In[78]:


#LPG is the cooking fuel widely used by patients having disease


# In[79]:


plt.figure(figsize=[16,8])
sns.set(style="darkgrid")


fig,ax1=plt.subplots(figsize=[16,8])

ax1.hist(imputed_data[imputed_data['diseases']==1]["primary_drinking_water"],color='red',bins=11, alpha=1)
ax1.hist(imputed_data[imputed_data['diseases']==0]["primary_drinking_water"],color='blue',bins=11, alpha=0.5)

ax1.set(xlabel="primary_drinking_water")
ax1.set_title("Effect of drinking water")

plt.show()


# In[80]:


g=pd.crosstab(imputed_data["alcohol_consumption"],imputed_data["diseases"])


# In[81]:


g.plot(kind='bar',stacked=True,color=['green','yellow'])


# In[82]:


sns.regplot(x="primary_toilet", y="primary_drinking_water", data=imputed_data)


# In[83]:


sns.regplot(x="primary_toilet", y="primary_electricity", data=imputed_data)


# In[84]:


#there is weak positive relation between primary_toilet & primary_electricity


# In[85]:


#Heatmap to find correlation


# In[86]:


plt.figure(figsize=(20,10))
sns.heatmap(imputed_data.corr(),annot=True, cmap='terrain')


# In[87]:


#we observe weak positive correlation between disease & age
#we observe weak negative correlation between disease & primary_toilet,primary_drinking_water,primary_electriccity


# In[88]:


imputed_data.head()


# In[89]:


#Training & testing the data
from sklearn.model_selection import train_test_split


# In[90]:


imputed_data['gender']=imputed_data['gender'].replace(['M'],0)
imputed_data['gender']=imputed_data['gender'].replace(['F'],1)
imputed_data['alcohol_consumption']=imputed_data['alcohol_consumption'].replace(['Yes'],1)
imputed_data['alcohol_consumption']=imputed_data['alcohol_consumption'].replace(['No'],0)
imputed_data['is_literate']=imputed_data['is_literate'].replace(['No'],0)
imputed_data['is_literate']=imputed_data['is_literate'].replace(['Yes'],1)


# In[91]:


#get dummy values for tobacco_consumption
list=['tobacco_consumption']


# In[92]:


imp_data=imputed_data.copy()


# In[93]:


imp_data=pd.get_dummies(imp_data,columns=list)


# In[94]:


imp_data.head()


# In[95]:


y2 = imp_data['diseases']
X2 = imp_data.drop("diseases",axis=1)


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state=40)


# In[97]:


y_test.value_counts()


# In[98]:


y_train.value_counts()


# In[99]:


#Scaling the data
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()


# In[100]:


X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)


# In[101]:


#Logistic Regression


# In[102]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[103]:


model1=lr.fit(X_train,y_train)


# In[104]:


y_pred=model1.predict(X_test)


# In[105]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# from sklearn.metrics import r2_score
# r2_score(y_test,y_pred)

# In[112]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[125]:


sns.heatmap(cm, annot=True, cmap='BuPu')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

score = accuracy_score(y_test, y_pred)
print("Accuracy of the Random Forest model is", score)
plt.show()


# In[116]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[117]:


#Random forest Classifier


# In[118]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[119]:


clf = RandomForestClassifier(n_estimators = 100) 
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)
score=metrics.accuracy_score(y_test, y_pred2)
print("Accuracy of Random forest classifier is", score)


# In[120]:


print(classification_report(y_test,y_pred2))


# In[121]:


cm2=confusion_matrix(y_test,y_pred2)
cm2


# In[122]:


plt.figure(figsize=(7,5))

ax = sns.heatmap(cm2/np.sum(cm2),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')


# In[123]:


from sklearn.tree import DecisionTreeClassifier
model3=DecisionTreeClassifier()
model3=model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)
score=accuracy_score(y_test,y_pred3)
print("Accuracy of the decision tree model is", score)


# In[124]:


cm3 = confusion_matrix(y_test, y_pred3)

plt.figure(figsize=(7,5))

ax = sns.heatmap(cm3/np.sum(cm3),fmt='.2%', annot=True, cmap='Greens')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No Disease','Disease'])
ax.yaxis.set_ticklabels(['No Disease','Disease'])

plt.show()


# In[ ]:




