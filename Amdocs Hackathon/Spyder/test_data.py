import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datasets=pd.read_csv("")

import pickle
model=pickle.load(open('logistic_model.pickle','rb'))
columns=pickle.load(open('columns.pickle','wb'))


model.predict
dataset['Probability_of_churn']=model.predict_proba(dataset[dataset.columns])[:,1]

dataset=dataset.sort_values("Probability_of_churn", axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 

results=[]
for i in range(0,10):
    data=[]
    for item in columns:
        data.append(datasets[item][i])
    results.append(data)