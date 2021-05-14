# Azure Machine Learning Notebook Code and run as pipeline

## Ability to run notebook code as Pipeline

## Prerequistie

- Azure Account
- Azure Machine learning
- Create a compute instance
- Create a compute cluster as cpu-cluster
- Select Standard D series version
- Create Train file to train the model
- Create a pipeline file to run the as pipeline

## Steps

## Create Train file as train.py

- Create a directory ./train_src
- Create a train.py
- Should be a python file not notebook

```
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
```

## Create Pipeline code

- Load the workspace config

```
import azureml.core
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()
```

- Get the default store information

```
# Default datastore 
def_data_store = ws.get_default_datastore()

# Get the blob storage associated with the workspace
def_blob_store = Datastore(ws, "workspaceblobstore")

# Get file storage associated with the workspace
def_file_store = Datastore(ws, "workspacefilestore")
```

- Create compute cluster

```
from azureml.core.compute import ComputeTarget, AmlCompute

compute_name = "cpu-cluster"
vm_size = "Standard_F16s_v2"
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # Standard_F16s_v2 is CPU-enabled
                                                                min_nodes=0,
                                                                max_nodes=4)
    # create the compute target
    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current cluster status, use the 'status' property
    print(compute_target.status.serialize())
```

- Load the package dependecies

```
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment 

aml_run_config = RunConfiguration()
# `compute_target` as defined in "Azure Machine Learning compute" section above
aml_run_config.target = compute_target

USE_CURATED_ENV = True
if USE_CURATED_ENV :
    curated_environment = Environment.get(workspace=ws, name="AzureML-Tutorial")
    aml_run_config.environment = curated_environment
else:
    aml_run_config.environment.python.user_managed_dependencies = False
    
    # Add some packages relied on by data prep step
    aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
        conda_packages=['pandas','scikit-learn','seaborn','tqdm'], 
        pip_packages=['azureml-sdk', 'azureml-dataprep[fuse,pandas]','seaborn','tqdm'], 
        pin_sdk_version=False)
```

- Load the data set

```
from azureml.core import Dataset
from azureml.data.dataset_factory import DataType

# create a TabularDataset from a delimited file behind a public web url and convert column "Survived" to boolean
web_path ='https://dprepdata.blob.core.windows.net/demo/Titanic.csv'
my_dataset = Dataset.Tabular.from_delimited_files(path=web_path, set_column_types={'Survived': DataType.to_bool()})
```

- set the dataset as input

```
from azureml.pipeline.steps import PythonScriptStep
dataprep_source_dir = "./dataprep_src"
#entry_point = "prepare.py"
# `my_dataset` as defined above
ds_input = my_dataset.as_named_input('input1')
```

- Setup output optional

```
from azureml.data import OutputFileDatasetConfig
from azureml.core import Workspace, Datastore

datastore = ws.get_default_datastore()

output_data1 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))
output_data_dataset = output_data1.register_on_complete(name = 'titanic_output_data')
```

- I am only creating single step

```
train_source_dir = "./train_src"
train_entry_point = "train.py"

training_results = OutputFileDatasetConfig(name = "training_results",
    destination = def_blob_store)

    
train_step = PythonScriptStep(
    script_name=train_entry_point,
    source_directory=train_source_dir,
    arguments=["--input_data", ds_input],
    compute_target=compute_target, # , "--training_results", training_results
    runconfig=aml_run_config,
    allow_reuse=True
)
```

- setup the pipeline config and assign

```
# list of steps to run (`compare_step` definition not shown)
compare_models = [train_step]

from azureml.pipeline.core import Pipeline

# Build the pipeline
pipeline1 = Pipeline(workspace=ws, steps=train_step)
```

- Validate the pipeline

```
pipeline1.validate()
print("Pipeline validation complete")
```

- Now time to submit the pipeline
- Wait for pipeline to finish

```
from azureml.core import Experiment

# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'Titanic_Pipeline_Notebook').submit(pipeline1)
pipeline_run1.wait_for_completion()
```