import pandas as pd
import numpy as np
import seaborn as sns

df_raw_train = pd.read_csv('C:/Users/kibhattacharya/Desktop/GoTo Docs/ML/DataSets/Titanic/train.csv')
df_raw_test = pd.read_csv('C:/Users/kibhattacharya/Desktop/GoTo Docs/ML/DataSets/Titanic/test.csv')

# print(df_raw.isnull().sum())

####################### Data Pre-processing for training set #######################
# Gender
df_raw_train['Sex'] = [1 if b == 'male' else 0 for b in df_raw_train.Sex]

# Age
df_raw_train['Age'] = df_raw_train['Age'].fillna(df_raw_train['Age'].mean())

# Embarked
df_raw_train['Embarked'] = df_raw_train['Embarked'].fillna(df_raw_train['Embarked'].mode()[0])

#Extracting only the character from the Cabin and add it as a new feature
df_raw_train['Cabin'] = df_raw_train.Cabin.str.slice(0,1)
df_raw_train['Cabin'] = df_raw_train['Cabin'].fillna(df_raw_train['Cabin'].mode()[0])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_raw_train['Cabin'] = lb.fit_transform(df_raw_train['Cabin'])


# Removing the columns PassengerId,Name, Ticket and Fare
df_train = df_raw_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)

X_train = df_train.drop(columns=['Survived'], axis=1)
y_train = df_train.iloc[:, 0:1]
y_train = y_train.to_numpy()

# Using get_dummies() to encode the categorical variables
ohc_embarked = pd.get_dummies(X_train.Embarked)

# ohc_cabin_class = pd.get_dummies(X_train.Cabin_Class)

X_train = X_train.drop(columns=['Embarked'], axis=1)
X_train = pd.concat([X_train, ohc_embarked], axis=1, sort='False')

####################### Data Pre-processing for Test set #######################
df_raw_test['Sex'] = [1 if b == 'male' else 0 for b in df_raw_test.Sex]

# Age
df_raw_test['Age'] = df_raw_test['Age'].fillna(df_raw_test['Age'].mean())

# Embarked
df_raw_test['Embarked'] = df_raw_test['Embarked'].fillna(df_raw_test['Embarked'].mode()[0])

#Extracting only the character from the Cabin and add it as a new feature
df_raw_test['Cabin'] = df_raw_test.Cabin.str.slice(0,1)
df_raw_test['Cabin'] = df_raw_test['Cabin'].fillna(df_raw_test['Cabin'].mode()[0])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_raw_test['Cabin'] = lb.fit_transform(df_raw_test['Cabin'])

# Removing the columns Name, Ticket and Fare
df_test = df_raw_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)

X_test = df_test

# Using get_dummies() to encode the categorical variables
ohc_embarked_test = pd.get_dummies(X_test.Embarked)

X_test = X_test.drop(columns=['Embarked'], axis=1)
X_test = pd.concat([X_test, ohc_embarked_test], axis=1, sort='False')

# Using Standard Scalar to bring the features under same scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

####################### Creating Models #######################

#LOGISTIC REGRESSION - 77.033%
# from sklearn.linear_model import LogisticRegression
#
# lg = LogisticRegression(random_state=0)
# lg = lg.fit(X_train, y_train)
#
# y_pred = lg.predict(X_test)

# SUPPORT VECTOR - 76%
# from sklearn.svm import SVC
# svm = SVC(kernel='sigmoid',C = 0.030,random_state=0)
# svm = svm.fit(X_train,y_train)
# y_pred = svm.predict(X_test)

# STOCHASTIC GRADIENT - 59.8%
# from sklearn.linear_model import SGDClassifier
# sgd = SGDClassifier(loss = 'modified_huber',shuffle = True,random_state = 101)
# sgd.fit(X_train,y_train)
# y_pred = sgd.predict(X_test)

# RANDOM FOREST CLASSIFIER - 78.9%
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50, min_samples_leaf=100, random_state=0)
rfc = rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

# NAIVE BAYES - 74.162%
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB(var_smoothing=1e-9)
# nb = nb.fit(X_train,y_train)
# y_pred = nb.predict(X_test)

# KNN - 74.162%
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier(n_neighbors=20, p=2)
# knn = knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# Dumping the prediction to csv
df_pred = pd.DataFrame(y_pred, columns=['Survived'])
df_pred.index += 892
df_pred.to_csv('C:/Users/kibhattacharya/Desktop/GoTo Docs/ML/DataSets/Titanic/gender_submission_pred.csv', index='True',
               index_label='PassengerID')
