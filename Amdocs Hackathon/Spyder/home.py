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


X=datasets.drop(['MonthlyCharges','Dependents_Yes','DeviceProtection_Yes','gender_1','OnlineBackup_Yes',
                'Partner_Yes','PhoneService_Yes' ],axis='columns')
X=X.drop(['Churn','customerID'],axis='columns')

X=X.drop(['PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check','PaymentMethod_Mailed check'],axis='columns')



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)


from sklearn.metrics  import accuracy_score

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()

log_model.fit(X_train,y_train)
y_pred1=log_model.predict(X_test)

print(round(accuracy_score(y_pred1,y_test)*100,2))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_model=DecisionTreeClassifier()

tree_model.fit(X_train,y_train)
y_pred2=tree_model.predict(X_test)

print(round(accuracy_score(y_pred2,y_test)*100,2))

#Support Vector Machine
from sklearn.svm import SVC
svc_model=SVC(kernel='rbf',random_state=50)

svc_model.fit(X_train,y_train)
y_pred3=svc_model.predict(X_test)

print(round(accuracy_score(y_pred3,y_test)*100,2))


#Random Forest 
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)

rf_model.fit(X_train,y_train)
y_pred4=rf_model.predict(X_test)

print(round(accuracy_score(y_pred4,y_test)*100,2))

#KNearert neighbours
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()

knn_model.fit(X_train,y_train)
y_pred5=knn_model.predict(X_test)

print(round(accuracy_score(y_pred5,y_test)*100,2))


from sklearn.metrics import confusion_matrix
cm_log=confusion_matrix(y_pred1,y_test)
cm_log

datasets['Probability_of_churn']=log_model.predict_proba(datasets[X_test.columns])[:,1]

# To get the weights of all the variables
weights = pd.Series(log_model.coef_[0],index=X.columns.values)
weights=weights.sort_values(ascending = False)



from sklearn.metrics import classification_report
report = classification_report(y_test, log_model.predict(X_test))

print(report)






