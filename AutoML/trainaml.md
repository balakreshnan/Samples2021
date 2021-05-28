# Automated ML training using Azure DevOps - CI/CD

## Use Azure DevOps to create CI/CD code to train a model using automated ML using sdk

## Steps

- Import libraries

```
import logging

# from matplotlib import pyplot as plt
import pandas as pd
import os

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
# from azureml.widgets import RunDetails
from sklearn.metrics import confusion_matrix

from azureml.core.authentication import ServicePrincipalAuthentication

import numpy as np
import itertools

import argparse 
import json
import os
```

- Print the AML version

```
print("This notebook was created using version 1.29.0 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")
```

- Get the arguments passed in command line

```
parse = argparse.ArgumentParser()
parse.add_argument("--tenantid")
parse.add_argument("--acclientid")
parse.add_argument("--accsecret")
    
args = parse.parse_args()
```

- Authenticate using service prinicipal
- all the information to authenticate is passed as command line arguments
- configured in azure DevOps

```
sp = ServicePrincipalAuthentication(tenant_id=args.tenantid, # tenantID
                                    service_principal_id=args.acclientid, # clientId
                                    service_principal_password=args.accsecret) # clientSecret
```

- Get workspace information

```
ws = Workspace.get(name="mlopsdev",
                   auth=sp,
                   subscription_id="c46a9435-c957-4e6c-a0f4-b9a597984773", resource_group="mlops")
```

- Now create a experiment name

```
# choose a name for experiment
experiment_name = 'automl-classification-ccard-remote'

experiment=Experiment(ws, experiment_name)

output = {}
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T

ws.get_details()
```

- Setup the compute clsuter

```
# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           max_nodes=6)
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
```
- Download the data set needed

```
data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/creditcard.csv"
dataset = Dataset.Tabular.from_delimited_files(data)
training_data, validation_data = dataset.random_split(percentage=0.8, seed=223)
label_column_name = 'Class'
```

- Set up AutoML config

```
automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": 'average_precision_score_weighted',
    "enable_early_stopping": True,
    "max_concurrent_iterations": 2, # This is a limit for testing purpose, please increase it as per cluster size
    "experiment_timeout_hours": 0.25, # This is a time limit for testing purposes, remove it for real use cases, this will drastically limit ablity to find the best model possible
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target = compute_target,
                             training_data = training_data,
                             label_column_name = label_column_name,
                             **automl_settings
                            )
```

- Submit the experiment

```
remote_run = experiment.submit(automl_config, show_output = False)

remote_run.wait_for_completion(show_output=True)
```

- Wait for the experiment to complete
- Validate the model

```
best_run, fitted_model = remote_run.get_output()
fitted_model

# convert the test data to dataframe
X_test_df = validation_data.drop_columns(columns=[label_column_name]).to_pandas_dataframe()
y_test_df = validation_data.keep_columns(columns=[label_column_name], validate=True).to_pandas_dataframe()

# call the predict functions on the model
y_pred = fitted_model.predict(X_test_df)
y_pred
```

## Azure DevOps

- Create a project called AMLOps
- Create a pipeine for build
- name is AutoMLOps
- Connect to your github job
- Code is available in https://github.com/balakreshnan/AMLcode/tree/main/AutomatedML

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops1.jpg "Service Health")

- Configure the agent

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops2.jpg "Service Health")

- Now configure Use Python

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops3.jpg "Service Health")

- Set the version as 3.6
- Configure the agend dependencies
- Install all the libraries needed to experiment to run
- Path - AutomatedML/agent_dependicy.sh

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops4.jpg "Service Health")

- Copy the data to build directory
- Target: $(Build.SourcesDirectory)

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops5.jpg "Service Health")

- Now configure the Automated ML run python code

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops6.jpg "Service Health")

- Script Path: AutomatedML/automlpipeline.py
- Argumetns: --tenantid $(tenatid) --acclientid $(acclientid) --accsecret $(accsecret)
- Once everything is set
- Click Save and queue
- Wait for experiment run to complete

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops7.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amldevops8.jpg "Service Health")

- Follow the same for inferencing code and then create release pipeline to deploy to other environments.