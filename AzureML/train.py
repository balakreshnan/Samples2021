# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import pandas as pd
import numpy as np
from azureml.core import Workspace, Dataset
from azureml.core import Dataset
from azureml.data.dataset_factory import DataType
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import sklearn as sk
import pandas as pd
# import seaborn as sn
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("In train.py")
print("As a data scientist, this is where I use my training code.")

parser = argparse.ArgumentParser("train")

parser.add_argument("--input_data", type=str, help="input data")
parser.add_argument("--output_train", type=str, help="output_train directory")

args = parser.parse_args()

print("Argument 1: %s" % args.input_data)
print("Argument 2: %s" % args.output_train)

if not (args.output_train is None):
    os.makedirs(args.output_train, exist_ok=True)
    print("%s created" % args.output_train)
    
web_path ='https://dprepdata.blob.core.windows.net/demo/Titanic.csv'
titanic_ds = Dataset.Tabular.from_delimited_files(path=web_path, set_column_types={'Survived': DataType.to_bool()})

# preview the first 3 rows of titanic_ds
#titanic_ds.take(3).to_pandas_dataframe()
    
#df = args.input_data.to_pandas_dataframe()

df = titanic_ds.to_pandas_dataframe()
df.head()

titanic_features = df.copy()
titanic_labels = titanic_features.pop('Survived')

df1 = pd.get_dummies(df)

y = df1['Survived']
X = df1
X = X.drop(columns=['Survived'])
X['Age'] = X['Age'].fillna(0)
X = X.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X.iloc[460:,:])
round(LR.score(X,y), 4)

y_pred = LR.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

print("roc_auc_score: ", roc_auc_score(y_test, y_pred))
print("f1 score: ", f1_score(y_test, y_pred))

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

clf.fit(X, y)

print(clf.predict(X_test))