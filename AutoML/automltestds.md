# Azure Automated Machine learning using test data set to validate

## Azure Machine learning use Test data set to validate and combine input and predictions

## Use Case

- Run Automated ML mode
- Use Test data set to validate model
- Use separate test dataset which are data that model hasn't seen in training
- Save the predictions
- Combine the inpute test data set and prediction to make final output for validation

## Code

- use python version 3.6
- AutoML only works with this version

```
import logging

from matplotlib import pyplot as plt
import pandas as pd
import os

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
```

- Print the version

```
print("This notebook was created using version 1.29 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")
```

- at the time of testin version was 1.28.0
- Load the subscription information

```
ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'Titanic-automl_Test'

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
```

- Create a compute cluster

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_FS16_V2',
                                                           max_nodes=6)
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
```

- Load the data set

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'rgname'
workspace_name = 'workspacename'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='titanic_ds')
dataset.to_pandas_dataframe()
```

- Create training and test data set
- Also configure label column name

```
training_data, validation_data = dataset.random_split(percentage=0.8, seed=223)
label_column_name = 'Survived'
```

- Now configure the auto ml run

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
                             
                             # Use train/test split
                             test_size=0.2,
                             
                             **automl_settings
                            )
```

- Now run the experiment

```
remote_run = experiment.submit(automl_config, show_output = False)
```

- Show the automl run outputs

```
from azureml.widgets import RunDetails
RunDetails(remote_run).show()
```

- Wait for experiment to complete

```
remote_run.wait_for_completion(show_output=True)
```

- Get he best model

```
best_run, fitted_model = remote_run.get_output()
fitted_model
```

- now run the test run

```
test_run = next(best_run.get_children(type='automl.model_test'))
test_run.wait_for_completion(show_output=False, wait_post_processing=True)
test_run
```

- Print the test metrics

```
test_run_metrics = test_run.get_metrics()
for name, value in test_run_metrics.items():
    print(f"{name}: {value}")
```

- See the predicted output

```
test_run_details = test_run.get_details()
test_run_predictions = Dataset.get_by_id(ws, test_run_details['outputDatasets'][0]['identifier']['savedId'])
test_run_predictions.to_pandas_dataframe().head()
```

- Now load the test data set

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'c46a9435-c957-4e6c-a0f4-b9a597984773'
resource_group = 'mlops'
workspace_name = 'mlopsdev'

workspace = Workspace(subscription_id, resource_group, workspace_name)

validation_data = Dataset.get_by_name(workspace, name='titanictest')
validation_data.to_pandas_dataframe()
```

- Create features and label

```
# convert the test data to dataframe
X_test_df = validation_data.drop_columns(columns=[label_column_name]).to_pandas_dataframe()
y_test_df = validation_data.keep_columns(columns=[label_column_name], validate=True).to_pandas_dataframe()
```

- run the prediction from above model

```
# call the predict functions on the model
y_pred = fitted_model.predict(X_test_df)
y_pred
```

- Print the confusion matrix

```
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

cf =confusion_matrix(y_test_df.values,y_pred)
plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
class_labels = ['False','True']
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks,class_labels)
plt.yticks([-0.5,0,1,1.5],['','False','True',''])
# plotting text value inside cells
thresh = cf.max() / 2.
for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
plt.show()
```

- print model metrics

```
from azureml.train.automl.model_proxy import ModelProxy

model_proxy = ModelProxy(best_run)
predictions, test_run_metrics = model_proxy.test(validation_data)

print(predictions.to_pandas_dataframe().head())
pd.DataFrame.from_dict(test_run_metrics, orient='index', columns=['Value'])
```

- New approach
- get the predictions

```
df = validation_data.to_pandas_dataframe()
```

```
predictions_df = predictions.to_pandas_dataframe()
```

```
result = pd.concat(df, predictions_df, axis=1, join="inner")
```

- get the predictions - Only to get the test run for model training run

```
import pandas as pd

test_run.download_file("predictions/predictions.csv")
predictions_df = pd.read_csv("predictions.csv")
```

- convert to pandas data frame

```
df = dataset.to_pandas_dataframe()
```

- Now combine the test data set with predicted output

```
result = pd.concat([df, predictions_df], axis=1, join="inner")
```