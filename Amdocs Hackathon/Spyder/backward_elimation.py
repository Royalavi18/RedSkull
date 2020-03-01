import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datasets=pd.read_csv('Churn/Customer_Churn.csv')

datasets['MultipleLines']=datasets['MultipleLines'].replace({'No phone service' : 'No'})
columns=['OnlineBackup','DeviceProtection','StreamingMovies','OnlineSecurity','StreamingTV','OnlineSecurity','TechSupport']

for i in columns:
    datasets[i]=datasets[i].replace({'No internet service' :'No'})

datasets['TotalCharges']=datasets['TotalCharges'].replace(" ",np.nan)

datasets=datasets[datasets['TotalCharges'].notnull()]

datasets=datasets.reset_index()[datasets.columns]

datasets['TotalCharges']=datasets['TotalCharges'].astype(float)

        
""" 
columns=['customerID' , 'gender', 'SeniorCitizen','Partner','Dependents','tenure' ,'PhoneService','MultipleLines','InternetService' ,'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn']
for i in range(0,len(datasets)):
    dic=[]
    for item in columns:
        if item=='SeniorCitizen':
            if datasets[item][i]==0:
                dic.append('No')
            else:
                dic.append('Yes')
        elif item=='tenure':
            dic.append(int(datasets[item][i]))
        else:
            dic.append(datasets[item][i])
    db.execute('insert into data(customerID , gender,SeniorCitizen,Partner,Dependents,tenure ,PhoneService,MultipleLines,InternetService ,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)' ,dic)
    db.commit()
"""



datasets['gender']=datasets['gender'].replace({'Male':1,'Female':0})
datasets['Churn']=datasets['Churn'].replace({'Yes':1,'No':0})

datasets=pd.get_dummies(datasets,columns=['Contract','Dependents','DeviceProtection','gender','InternetService',
                                          'MultipleLines','OnlineBackup','OnlineSecurity','PaperlessBilling' ,'Partner',
                                          'PaymentMethod','PhoneService','SeniorCitizen','StreamingMovies','StreamingTV',
                                          'TechSupport'],
                        drop_first=True)

from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()

columns=['tenure','MonthlyCharges','TotalCharges']

datasets[columns]=standardScaler.fit_transform(datasets[columns])

data=datasets.corr()

y=datasets['Churn']
X=datasets.drop(['Churn','customerID'],axis='columns')

import statsmodels.api  as sm

X= np.append(arr = np.ones((7032, 1)).astype(int),  
              values = X, axis = 1) 



X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())


X_opt=X[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,17,18,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,17,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,9,10,11,12,13,14,17,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,9,10,11,13,14,17,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,9,10,11,13,14,17,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())


X_opt=X[:,[0,1,3,4,5,9,10,11,13,14,17,20,21,22,23]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())
