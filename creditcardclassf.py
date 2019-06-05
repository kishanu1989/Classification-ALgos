import pandas as pd
import numpy as np

#Extracting Data
df_data = pd.read_csv('C:/Users/kibhattacharya/Desktop/GoTo Docs/ML/DataSets/creditcard.csv')
# print(df_data.head())

# print(df_data['Class'].value_counts())

######################################### Upsampling #########################################
from sklearn.utils import resample

df_majority = df_data[df_data.Class == 0]  # filter condition
df_minority = df_data[df_data.Class == 1]  # filter condition

df_minority_upsampled = resample(df_minority, replace=True, n_samples=284315, random_state=123)

df_upsampled = pd.concat([df_minority_upsampled, df_majority])

# Start test and training
X = df_upsampled.drop('Class', axis=1)
y = df_upsampled.Class

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg_fit = reg.fit(X, y)

reg_predict = reg.predict(X)

from sklearn.metrics import accuracy_score

accr = accuracy_score(y, reg_predict)
print("Accuracy After Upsampling:", accr)

from sklearn.metrics import roc_auc_score
prob_up = reg_fit.predict_proba(X)
prob_up = [p[1] for p in prob_up]
print("ROC Score for Upsampling",roc_auc_score(y,prob_up))
######################################### Downsampling #########################################

df_down_majority = df_data[df_data.Class == 0]
df_down_minority = df_data[df_data.Class == 1]

df_downsampled = resample(df_down_majority, replace=False, n_samples=492, random_state=123)

df_final_downsampled = pd.concat([df_downsampled,df_down_minority])

#model
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

X_down = df_final_downsampled.drop('Class',axis=1)
y_down = df_final_downsampled.Class

reg_down_fit = reg.fit(X_down,y_down)
reg_down_predict = reg.predict(X_down)

from sklearn.metrics import accuracy_score
reg_acc_down_score = accuracy_score(y_down,reg_down_predict)

print("Accuracy after Downsampling",reg_acc_down_score)
