#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("loan.csv")
df.head()


# 1.	Loan_ID - This refer to the unique identifier of the applicant's affirmed purchases
# 2.	Gender - This refers to either of the two main categories (male and female) into which applicants are divided on the basis of their reproductive functions
# 3.	Married - This refers to applicant being in a state of matrimony
# 4.	Dependents - This refres to persons who depends on the applicants for survival
# 5.	Education - This refers to number of years in which applicant received systematic instruction, especially at a school or university
# 6.	Self_Employed - This refers to applicant working for oneself as a freelancer or the owner of a business rather than for an employer
# 7.	Applicant Income - This refers to disposable income available for the applicant's use under State law.
# 8.	CoapplicantIncome - This refers to disposable income available for the people that participate in the loan application process alongside the main applicant use under State law.
# 9.	Loan_Amount - This refers to the amount of money an applicant owe at any given time.
# 10. Loan_Amount_Term - This refers to the duaration in which the loan is availed to the applicant
# 11. Credit History - This refers to a record of applicant's ability to repay debts and demonstrated responsibility in repaying them.
# 12. Property_Area - This refers to the total area within the boundaries of the property as set out in Schedule.
# 13. Loan_Status - This refres to whether applicant is eligible to be availed the Loan requested.
# 

# we have build a model that can predict loan will be approved or not

# In[3]:


df.shape


# There are 614 rows and 13 columns

# In[4]:


df.info()


#  There are 12 independent variables(Numerical)
#     
# 1. Loan_id
# 2. Gender
# 3. Married
# 4. Dependents
# 5. Education
# 6. self_Employed
# 7. ApplicationIncome
# 8. CoapplicantIncome
# 9. LoanAmount
# 10. Loan_Amount_Term
# 11. Credit_History
# 12. Property_Area
#         
# Target varible:
#     
# 1. Loan_Status(catagorical).this is classification prolem    

# In[5]:


df.isnull().sum()


# There are null values presnt in some columns
# 
# 1. catagorical columns null value filled with their mode
# 2. numerical columns null values filled with their mean

# # Handling Null Values

# In[6]:


# Gender
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
#Married
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
#Dependents
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
# Self_Employed
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
# Loan_Amount
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())
# Loan_Amount_Term
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
# Credit History
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[7]:


df


# In[8]:


# recheck the null values
df.isnull().sum()


# There is no missing values

# In[9]:


# drop unwanted columns
df.drop(['Loan_ID'],axis=1,inplace=True)


# In[10]:


df


# In[11]:


#checking unique values
cat_cols = df.select_dtypes(include="O").columns

for columns in cat_cols:
    print("Unique Values of:", columns,set(df[columns]))

    print("-"*50)


# In[12]:


df.describe()


# # Loan Analysis

# In[13]:


df.corr()


# In[14]:


sns.heatmap(df.corr(),annot=True)


# In[15]:


df.columns


# In[16]:


sns.countplot(df['Gender'])


# In[17]:


df['Gender'].value_counts()


# There are 502 are males and 112 are Females
# 
# most of the applicant are males

# In[18]:


# Married status
sns.countplot(df['Married'])


# In[19]:


df['Married'].value_counts()


# Most of the Applicant are Marries-->401
# 
# rest are unmarried-->213

# In[20]:


# applicant dependency
sns.countplot(df['Dependents'])


# In[21]:


df['Dependents'].value_counts()


# In[22]:


#Education
sns.countplot(df['Education'])


# In[23]:


df['Education'].value_counts()


# Graduate applicant-->480
# 
# not Graduate applicant--->134

# In[24]:


# self Employed
sns.countplot(df['Self_Employed'])


# In[25]:


df['Self_Employed'].value_counts()


# Applicants who are not self Employed--->532
# 
# Applicants who are  self Employed--->82

# In[26]:


# Loan Status
sns.countplot(df['Loan_Status'])


# In[27]:


df['Loan_Status'].value_counts()


# There are 422 applicants loan approved
# 
# there are 192 applicants loan have not been approved

# In[28]:


#Loan_Amount
sns.distplot(df['LoanAmount'])


# most of the applicant will apply for loan amount between 100 to 200

# # Bivariant Analysis

# In[29]:


# Relation between Gender and Loan Status
sns.countplot(df['Loan_Status'],hue=df['Gender'])


# In[30]:


pd.crosstab(df['Loan_Status'],df['Gender'])


# Total Females are 112
# 
# 1. 75 Females loan have approved
# 2. 37 Females loan have not approved
# 
# Total Males are 502
# 
# 1. 347 males loan have approved
# 2. 155 males loan have not approved

# In[31]:


# Relation between Education and Loan Status
sns.countplot(df['Education'],hue=df['Loan_Status'])


# In[32]:


sns.set_theme(style="whitegrid")
sns.barplot(df['Loan_Status'],df['LoanAmount'],hue=df['Education'])


# In[33]:


# Relation between loan_Status LoanAmount and Gender
sns.barplot(df['Loan_Status'],df['LoanAmount'],hue=df['Gender'])


# In[34]:


# Relation between Loan_Status and Loan_Aount_Term
sns.barplot(df['Loan_Status'],df['Loan_Amount_Term'])


# In[35]:


pd.crosstab(df['Loan_Status'],df['Loan_Amount_Term'])


# In[36]:


#Relationship between Loan_Status and ApplicantIncome
sns.barplot(df['Loan_Status'],df['ApplicantIncome'])


# In[37]:


sns.histplot(df['ApplicantIncome'],kde=True)


# Most of the applicant income between 5000 to 10000

# In[38]:


# Find CoapplicantIncome
sns.histplot(df['CoapplicantIncome'],kde=True)


# Most of the CoapplicantIncome 

# In[39]:


#Relation between Applicant income and Coapplicant income
sns.scatterplot(df['CoapplicantIncome'],df['ApplicantIncome'])


# This means coapplicant income is less than ApplicantIncome

# In[40]:


#Relationship between Loan_Status and CoapplicantIncome
sns.barplot(df['Loan_Status'],df['CoapplicantIncome'])


# We can see that lesser CoapplicantIncome has positive loan_status rather than Greater Coapllicant income has Negative loan_status

# In[41]:


#Relation between Applicant income and Coapplicant income with Loan_Status
plt.figure(figsize=(10,5))
sns.scatterplot(df['CoapplicantIncome'],df['ApplicantIncome'],hue=df['Loan_Status'])


# loan has approved of those Applicant whose income is lie between 0 to 10000 and also their Coapplicant
# 
# means  if Applicant income range is between 0 t0 10000 , so copapplicant income range is also same as main Applicant
# 
# then loan has been approved

# In[42]:


# Relation between Loan_Status and Credit_History
sns.barplot(df['Loan_Status'],df['Credit_History'])


#  with low credit history has negative impact on loan and higher credit history has positive impact on loan

# In[43]:


# Relation between Loan_Amount and property area
sns.barplot(df['Property_Area'],df['LoanAmount'])


# urban and semiurban have same LoanAmount but rural Property Area has high as comparison to both of this

# In[44]:


# Relation between Property_Area and Credit_History
sns.barplot(df['Property_Area'],df['Credit_History'])


# Almost all type of property areas has equal credit history but nsemiurban has little high

# In[45]:


# Relation between property area and loanstatus
sns.countplot(df['Property_Area'],hue=df['Loan_Status'])


# As above we can see credit history of semiurban is high so loan status is good as compare to both of them
# 
# 

# In[46]:


# Relation between property area and Gender
sns.countplot(df['Property_Area'],hue=df['Gender'])


# Male Applicant is more all areas , and semiurban has most of the applicant is male

# # Encoding the catagorical columns

# 1. Gender             
# 2.  Married               
# 3.  Education          
# 4.  Self_Employed 
# 5. Property_Area
# 6. Loan_Status

# In[47]:


df['Gender']=df['Gender'].map({'Female':0,'Male':1})
df['Loan_Status']=df['Loan_Status'].map({'N':0,'Y':1})
df['Married']=df['Married'].map({'No':0,'Yes':1})
df['Self_Employed']=df['Self_Employed'].map({'No':0,'Yes':1})
df['Property_Area']=df['Property_Area'].map({'Urban':2,'Rural':0,'Semiurban':1})


# In[48]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Education']=le.fit_transform(df['Education'])


# In[49]:


col=['Gender', 'Married', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']


# In[50]:


#Boxplot
for i in col:
    sns.boxplot(df[i])
    plt.show()


# In[51]:


# drop columns dependents
df=df.drop(['Dependents'],axis=1)


# In[52]:


from scipy.stats import zscore


# In[53]:


z=np.abs(zscore(df))


# In[54]:


z


# In[55]:


threshold=3
print(np.where(z>3))


# In[56]:


df1=df[(z<3).all(axis=1)]
df1


# In[57]:


# shape of data
print("shape of old data",df.shape)
print("shape of new data", df1.shape)


# In[58]:


print("loss of data percentage:",((df.shape[0]-df1.shape[0])/df.shape[0])*100)


# # Split data into x and y

# In[59]:


x=df1.iloc[:,:-1]
x


# In[60]:


y=df1.iloc[:,-1]


# In[61]:


y


# # Balanced the data

# In[62]:


from imblearn.over_sampling import SMOTE


# In[63]:


smt=SMOTE()
x,y=smt.fit_resample(x,y)


# In[64]:


y.value_counts()


# In[65]:


# Fearures Scaling
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
x=st.fit_transform(x)
x


# # splitting the data into train and test

# In[66]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[67]:


print(x_train.shape),print(y_train.shape),print(x_test.shape),print(y_test.shape)


# # import models

# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


# In[69]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[70]:


lg=LogisticRegression()
lg.fit(x_train,y_train)
predlg=lg.predict(x_test)
pred_trainlg=lg.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predlg))
print("Accuracy score on training data:", accuracy_score(y_train,pred_trainlg))
print("Confusion matrix:",confusion_matrix(y_test,predlg))
print("classification report:", classification_report(y_test,predlg))


# In[71]:


y_pred_prob=lg.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)


# In[72]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='logistic regression')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('logistic regression') 
plt.show()         


# In[73]:


auc_scorelg=roc_auc_score(y_test,lg.predict(x_test))
auc_scorelg


# In[74]:


dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
preddtc=dtc.predict(x_test)
pred_traindtc=dtc.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,preddtc))
print("Accuracy score on training data:", accuracy_score(y_train,pred_traindtc))
print("Confusion matrix:",confusion_matrix(y_test,preddtc))
print("classification report:", classification_report(y_test,preddtc))


# In[75]:


y_pred_prob=dtc.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='DecisionTreeClassifier')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('DecisionTreeClassifier') 
plt.show()         


# In[76]:


auc_scoredtc=roc_auc_score(y_test,dtc.predict(x_test))
auc_scoredtc


# In[77]:


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
predknn=knn.predict(x_test)
pred_trainknn=knn.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predknn))
print("Accuracy score on training data:", accuracy_score(y_train,pred_trainknn))
print("Confusion matrix:",confusion_matrix(y_test,predknn))
print("classification report:", classification_report(y_test,predknn))


# In[78]:


y_pred_prob=knn.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='KNeighborsClassifier')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('KNeighborsClassifier') 
plt.show()         


# In[79]:


auc_scoreknn=roc_auc_score(y_test,knn.predict(x_test))
auc_scoreknn


# In[80]:


gnb=GaussianNB()
gnb.fit(x_train,y_train)
predgnb=gnb.predict(x_test)
pred_traingnb=gnb.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predgnb))
print("Accuracy score on training data:", accuracy_score(y_train,pred_traingnb))
print("Confusion matrix:",confusion_matrix(y_test,predgnb))
print("classification report:", classification_report(y_test,predgnb))


# In[81]:


y_pred_prob=gnb.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='GaussianNB')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('GaussianNB') 
plt.show() 


# In[82]:


auc_scoregnb=roc_auc_score(y_test,gnb.predict(x_test))
auc_scoregnb


# # Parametric Tunning

# In[83]:


from sklearn.model_selection import GridSearchCV


# In[84]:


svm=SVC()
estimators={'kernel':['linear','poly','rbf'],'C':[1,10]}


# In[85]:


gsv=GridSearchCV(estimator=svm,param_grid=estimators)
gsv.fit(x,y)
print(gsv.best_params_)
print(gsv.best_score_)
print(gsv.best_estimator_)


# In[86]:


svm=SVC(kernel='rbf',C=1)
svm.fit(x_train,y_train)
predsvm=svm.predict(x_test)
pred_trainsvm=svm.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predsvm))
print("Accuracy score on training data:", accuracy_score(y_train,pred_trainsvm))
print("Confusion matrix:",confusion_matrix(y_test,predsvm))
print("classification report:", classification_report(y_test,predsvm))


# In[87]:


svm=SVC(probability=True)
svm.fit(x_train,y_train)
y_pred_prob=svm.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='SVC')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('SVC') 
plt.show() 


# In[88]:


auc_scoresvm=roc_auc_score(y_test,svm.predict(x_test))
auc_scoresvm


# In[89]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
predrf=rf.predict(x_test)
pred_trainrf=rf.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predrf))
print("Accuracy score on training data:", accuracy_score(y_train,pred_trainrf))
print("Confusion matrix:",confusion_matrix(y_test,predrf))
print("classification report:", classification_report(y_test,predrf))


# In[90]:


y_pred_prob=rf.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='RandomForestClassifier')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('RandomForestClassifier') 
plt.show() 


# In[91]:


auc_scorerf=roc_auc_score(y_test,rf.predict(x_test))
auc_scorerf


# In[92]:


ada=AdaBoostClassifier()
ada.fit(x_train,y_train)
predada=ada.predict(x_test)
pred_trainada=ada.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predada))
print("Accuracy score on training data:", accuracy_score(y_train,pred_trainada))
print("Confusion matrix:",confusion_matrix(y_test,predada))
print("classification report:", classification_report(y_test,predada))


# In[93]:


y_pred_prob=ada.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='AdaBoostClassifier')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('AdaBoostClassifier') 
plt.show() 


# In[94]:


auc_scoreada=roc_auc_score(y_test,ada.predict(x_test))
auc_scoreada


# In[95]:


gbr=GradientBoostingClassifier()
gbr.fit(x_train,y_train)
predgbr=gbr.predict(x_test)
pred_traingbr=gbr.predict(x_train)
print("Accuracy Score:",accuracy_score(y_test,predgbr))
print("Accuracy score on training data:", accuracy_score(y_train,pred_traingbr))
print("Confusion matrix:",confusion_matrix(y_test,predgbr))
print("classification report:", classification_report(y_test,predgbr))


# In[96]:


y_pred_prob=gbr.predict_proba(x_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='GradientBoostingClassifier')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('GradientBoostingClassifier') 
plt.show() 


# In[97]:


auc_scoregbr=roc_auc_score(y_test,gbr.predict(x_test))
auc_scoregbr


# # comparing Models

# In[98]:


models={"Models":['Logistic Regression','Decision Tree','SVC','GaussianNB','KNeighbors','RandomForest','AdaBoostClassifier',
                 'GradientBoostingClassifier'],"Score":[accuracy_score(y_test,predlg),accuracy_score(y_test,preddtc),
                  accuracy_score(y_test,predsvm),accuracy_score(y_test,predgnb),accuracy_score(y_test,predknn),accuracy_score(y_test,predrf),
                                                       accuracy_score(y_test,predada),accuracy_score(y_test,predgbr)],'auc_roc_score':[auc_scorelg,
                                                        auc_scoredtc,auc_scoresvm,auc_scoregnb,auc_scoreknn,auc_scorerf,auc_scoreada,auc_scoregbr]}


# In[99]:


df2=pd.DataFrame(models)
df2


# out of all models Random Forest model perform very well with score=83% and with best auc_score=83%

# # crossvalidation with best performing model

# In[100]:


score=cross_val_score(rf,x,y,cv=5)
print(score)
print(score.mean())
print(score.std())


# # save the best model

# In[101]:


import pickle
filename="RandomForestClassifier.pkl"
pickle.dump(rf,open(filename,'wb'))

load_model=pickle.load(open(filename,'rb'))
load_model.predict(x_test)


# In[102]:


conculusion=pd.DataFrame([load_model.predict(x_test)[:],(y_test)[:]],index=["Predicted","Original"])
conculusion


# In[103]:


def predval(p):
    p=p.reshape(1,-1)
    print(p.shape)
    predvalue=rf.predict(p)
    print(predvalue)


# In[104]:


p=np.array([1,1,0,1,5849,0.0,128.0,360.0,1.0,2])
predval(p)


# In[ ]:




