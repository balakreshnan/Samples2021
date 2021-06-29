# Azure Automated ML run - Exising run combine test and predicted output

## Combine Predicted output to test data set provide (private preview)

## Code

- Code in python
- Load test data set

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'c46a9435-c957-4e6c-a0f4-b9a597984773'
resource_group = 'mlops'
workspace_name = 'mlopsdev'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='titanictest')
dataset.to_pandas_dataframe()
```

- set the experiment run

```
from azureml.core.experiment import Experiment
experiment = Experiment(workspace=workspace, name='Titanic-automl')
```

- Get the autoML run details
- Run id is the run shows in experiment run in UI

```
# If you need to retrieve a run that already started, use the following code
from azureml.train.automl.run import AutoMLRun
remote_run = AutoMLRun(experiment = experiment, run_id = 'AutoML_d1646d14-f8d6-4172-8cd5-d6d9bcb0228e')
```

- get the best run

```
best_run, fitted_model = remote_run.get_output()
test_run = next(best_run.get_children(type='automl.model_test'))
test_run.wait_for_completion(show_output=False, wait_post_processing=True)
```

- Print the metrics

```
# Get test metrics
test_run_metrics = test_run.get_metrics()
for name, value in test_run_metrics.items():
    print(f"{name}: {value}")
```

- get the predicted values

```
import pandas as pd

test_run.download_file("predictions/predictions.csv")
predictions_df = pd.read_csv("predictions.csv")
```

- convert test dataset to pandas

```
df = dataset.to_pandas_dataframe()
```

- now merge the 2 data set each row to each row

```
result = pd.concat([df, predictions_df], axis=1, join="inner")
result
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/testdataset1.jpg "Service Health")