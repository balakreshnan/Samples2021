# Azure ML - Timeseries using TSFresh

## How to run Tsfresh in Azure machine learning

## Pre requistie

- Azure Account
- Azure Machine learning services

## How to

- Log into ml.azure.com
- Select the workspace to use
- Create a compute cluster for my sample i use DS14_v2 one node
- Once running log into Jupyter lab
- Create a new text file called requirements.txt
- Copy the content below

```
requests>=2.9.1
numpy>=1.15.1
pandas>=0.25.0
scipy>=1.2.0
statsmodels>=0.9.0
patsy>=0.4.1
scikit-learn>=0.22.0
tqdm>=4.10.0
dask[dataframe]>=2.9.0
distributed>=2.11.0
matrixprofile>=1.1.10<2.0.0
stumpy>=1.7.2
```

- Now create a new ML notebook
- Call it install.ipynb
- This note book is to install requirements and tsfresh
- Run this command to install requirements

```
pip install -r requirements.txt
```

- Now install tsfresh

```
pip install tsfresh
```

- Now create a new notebook to test tsfresh with sklearn
- Call the notebook sklearn.ipynb
- Start the notebook coding
- Sample is also available here - https://github.com/blue-yonder/tsfresh

```
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute
```

- Load the sample data set

```
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
download_robot_execution_failures() 
df_ts, y = load_robot_execution_failures()
```

- Split the dataset

```
X = pd.DataFrame(index=y.index)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

- Setup sklearn pipeline

```
ppl = Pipeline([
        ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
        ('classifier', RandomForestClassifier())
      ])
```

```
ppl.set_params(augmenter__timeseries_container=df_ts);
```

- Pipeline fit
- This is where training happens

```
ppl.fit(X_train, y_train)
```

- Now predict the model

```
y_pred = ppl.predict(X_test)
```

- Print the output

```
print(classification_report(y_test, y_pred))
```

- display the features:

```
ppl.named_steps["augmenter"].feature_selector.relevant_features
```

```
df_ts_train = df_ts[df_ts["id"].isin(y_train.index)]
df_ts_test = df_ts[df_ts["id"].isin(y_test.index)]
```

- Train the model

```
ppl.set_params(augmenter__timeseries_container=df_ts_train);
ppl.fit(X_train, y_train);
```

- Now save the pickle file

```
import pickle
with open("pipeline.pkl", "wb") as f:
    pickle.dump(ppl, f)
```

- Predict using tsfresh features

```
ppl.set_params(augmenter__timeseries_container=df_ts_test);
y_pred = ppl.predict(X_test)
```

- Print the output

```
print(classification_report(y_test, y_pred))
```

