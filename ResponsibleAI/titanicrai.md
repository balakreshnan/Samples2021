# Responsible AI Code sample for titanic data set

## How to implement Responsible AI features in titanic data set

## Intro

## Pre Requistie

- Azure account
- Azure storage
- Azure machine learning services

## Code Sample

- Using notebook code to perform responsible ai features
- Explainability
- Fairness
- and more

## library installation

```
!pip install raiwidgets==0.9.2
!pip install fairlearn==0.7.0
```

- Check the versions

```
!pip show fairlearn
!pip show raiwidgets
```

- Now the actual code
- Load the data from dataset or from file

```
from raiwidgets import ExplanationDashboard

from azureml.core import Dataset
from azureml.data.dataset_factory import DataType

# create a TabularDataset from a delimited file behind a public web url and convert column "Survived" to boolean
web_path ='https://dprepdata.blob.core.windows.net/demo/Titanic.csv'
titanic_ds = Dataset.Tabular.from_delimited_files(path=web_path, set_column_types={'Survived': DataType.to_bool()})

# preview the first 3 rows of titanic_ds
titanic_ds.take(3).to_pandas_dataframe()
from azureml.core import Workspace, Dataset

import pandas as pd
import numpy as np
```

```
df = pd.read_csv('Titanic.csv')
df.head()
```

```
df['id'] = df[['Name']].sum(axis=1).map(hash)
```

```
titanic_features = df.copy()
titanic_labels = titanic_features.pop('Survived')
```

```
df.drop('Name', axis=1, inplace=True)
```

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn as sk
```

```
df1 = pd.get_dummies(df)
```

```
y = df1['Survived']
```

```
X = df1
X = X.drop(columns=['Survived'])
X['Age'] = X['Age'].fillna(0)
X = X.dropna()
```

```
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn import preprocessing
```

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

```
import sklearn as sk
from sklearn.linear_model import LogisticRegression
```

```
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X.iloc[460:,:])
round(LR.score(X,y), 4)
```

```
y_pred = LR.predict(X_test)
```

```
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
```

```
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
```

```
from sklearn.datasets import load_breast_cancer
from sklearn import svm

# Explainers:
# 1. SHAP Tabular Explainer
#from interpret.ext.blackbox import TabularExplainer
from interpret.ext.blackbox import TabularExplainer
```

```
classes = X_train.columns.tolist()
```

```
explainer = TabularExplainer(LR, 
                             X_train, 
                             features=X.columns, 
                             classes=['Sex_male', 'Sex_female'])
```

```
from interpret.ext.blackbox import MimicExplainer

# you can use one of the following four interpretable models as a global surrogate to the black box model

from interpret.ext.glassbox import LGBMExplainableModel
from interpret.ext.glassbox import LinearExplainableModel
from interpret.ext.glassbox import SGDExplainableModel
from interpret.ext.glassbox import DecisionTreeExplainableModel

# "features" and "classes" fields are optional
# augment_data is optional and if true, oversamples the initialization examples to improve surrogate model accuracy to fit original model.  Useful for high-dimensional data where the number of rows is less than the number of columns. 
# max_num_of_augmentations is optional and defines max number of times we can increase the input data size.
# LGBMExplainableModel can be replaced with LinearExplainableModel, SGDExplainableModel, or DecisionTreeExplainableModel
explainer = MimicExplainer(LR, 
                           X_train, 
                           LGBMExplainableModel, 
                           augment_data=True, 
                           max_num_of_augmentations=10, 
                           features=X.columns, 
                           classes=['Sex_male', 'Sex_female'])
```

```
global_explanation = explainer.explain_global(X_test)
```

```
from raiwidgets import ExplanationDashboard

ExplanationDashboard(global_explanation, LR, dataset=X_test, true_y=y_test)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img1.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img2.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img3.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img4.jpg "Service Health")


```
# Sorted SHAP values
print('ranked global importance values: {}'.format(global_explanation.get_ranked_global_values()))
# Corresponding feature names
print('ranked global importance names: {}'.format(global_explanation.get_ranked_global_names()))
# Feature ranks (based on original order of features)
print('global importance rank: {}'.format(global_explanation.global_importance_rank))

# Note: Do not run this cell if using PFIExplainer, it does not support per class explanations
# Per class feature names
print('ranked per class feature names: {}'.format(global_explanation.get_ranked_per_class_names()))
# Per class feature importance values
print('ranked per class feature values: {}'.format(global_explanation.get_ranked_per_class_values()))
```

```
# Print out a dictionary that holds the sorted feature importance names and values
print('global importance rank: {}'.format(global_explanation.get_feature_importance_dict()))
```

```
sex = df['Sex']
sex.value_counts()
```

```
y_true = y_test
len(y_test)
len(y_pred)
```

```
sensitivefeatures = X_test[['Sex_male', 'Sex_female']]
sensitivefeatures
```

```
print ("Confusion Matrix:")
print (metrics.confusion_matrix(y_test, y_pred))
```

```
gm = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=sensitivefeatures)
print(gm.overall)
print(gm.by_group)
```

```
from fairlearn.metrics import selection_rate
sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitivefeatures)
```

- Fairness

```
from raiwidgets import FairnessDashboard

# A_test contains your sensitive features (e.g., age, binary gender)
# y_true contains ground truth labels
# y_pred contains prediction labels

FairnessDashboard(sensitive_features=sensitivefeatures,
                  y_true=y_test.tolist(),
                  y_pred=y_pred)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img5.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img6.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img7.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img8.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img9.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img10.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ResponsibleAI/images/img11.jpg "Service Health")
